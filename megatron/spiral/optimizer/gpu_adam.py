# Most of the code here has been copied from:
#   https://github.com/NVIDIA/apex/blob/741bdf50825a97664db08574981962d66436d16a/apex/optimizers/fused_adam.py
#   https://github.com/sail-sg/zero-bubble-pipeline-parallelism
# with some modifications.

import torch
import nvtx
from apex.multi_tensor_apply import multi_tensor_applier
from collections import defaultdict

from megatron import get_args
from megatron.spiral.initialize import get_thunder_cuda_manager


class SpiralGPUAdam(torch.optim.Optimizer):

    """Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0., amsgrad=False, set_grad_none=True,
                 capturable=False, master_weights=False):

        if amsgrad:
            raise RuntimeError('SpiralGPUAdam does not support the AMSGrad variant.')
        if capturable or master_weights:
            raise RuntimeError('SpiralGPUAdam does not support catureable or master_weights.')
        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(SpiralGPUAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        # For unscale in mixed precision
        self.half_precision = get_args().fp16
        self.inv_scale = 0
        self.params_have_main_grad = True
        self.step_event = None

        self.local_device = torch.cuda.current_device()
        self.remote_device = torch.device("cpu")
        self.cpu_state = defaultdict(dict)
        self.numel_threshold = get_args().hidden_size * get_args().hidden_size

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('SpiralGPUAdam requires cuda extensions')

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(SpiralGPUAdam, self).zero_grad()

    def _unscale_grads_and_check_inf(self):
        fp32_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if self.params_have_main_grad and hasattr(p, 'main_grad'):
                    if p.main_grad is not None:
                        p.main_grad = p.main_grad.float()
                        fp32_grads.append(p.main_grad.data)
                else:
                    if p.grad is not None:
                        p.grad = p.grad.float()
                        fp32_grads.append(p.grad.data)

        found_inf = torch.cuda.FloatTensor([0.0])
        inv_scale = torch.cuda.FloatTensor([self.inv_scale])
        torch._amp_foreach_non_finite_check_and_unscale_(fp32_grads, found_inf, inv_scale)

        return found_inf.item() > 0

    def init_cpu_state(self, p):
        cpu_state = self.cpu_state[p]
        if len(cpu_state) == 0:
            cpu_state['exp_avg'] = torch.empty(p.data.shape, device=self.remote_device, dtype=torch.float32, pin_memory=True)
            cpu_state['exp_avg_sq'] = torch.empty(p.data.shape, device=self.remote_device, dtype=torch.float32, pin_memory=True)
            cpu_state['fp32_param'] = torch.empty(p.data.shape, device=self.remote_device, dtype=torch.float32, pin_memory=True)
            cpu_state['fp32_grad'] = torch.empty(p.data.shape, device=self.remote_device, dtype=torch.float32, pin_memory=True)

    @nvtx.annotate("offload_state", color="pink")
    def offload_state(self, p):
        state = self.state[p]
        cpu_state = self.cpu_state[p]
        cpu_state['exp_avg'].copy_(state['exp_avg'].data, non_blocking=True)
        cpu_state['exp_avg_sq'].copy_(state['exp_avg_sq'].data, non_blocking=True)
        cpu_state['fp32_param'].copy_(state['fp32_param'].data, non_blocking=True)
        cpu_state['fp32_grad'].copy_(state['fp32_grad'].data, non_blocking=True)

    @nvtx.annotate("free_state", color="purple")
    def free_state(self, p):
        state = self.state[p]
        state['exp_avg'] = None
        state['exp_avg_sq'] = None
        state['fp32_param'] = None
        state['fp32_grad'] = None

    @nvtx.annotate("fetch_state", color="green")
    def fetch_state(self, p, grad_temp=False):
        state = self.state[p]
        cpu_state = self.cpu_state[p]
        state['exp_avg'] = cpu_state['exp_avg'].to(device=self.local_device, non_blocking=True)
        state['exp_avg_sq'] = cpu_state['exp_avg_sq'].to(device=self.local_device, non_blocking=True)
        state['fp32_param'] = cpu_state['fp32_param'].to(device=self.local_device, non_blocking=True)

        if grad_temp:
            state['fp32_grad'] = cpu_state['fp32_grad'].to(device=self.local_device, non_blocking=True)

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('SpiralGPUAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        loss = None
        if closure is not None:
            loss = closure()

        # Unscale fp32 grads and check for inf/nan
        if self.half_precision:
            self.found_inf = self._unscale_grads_and_check_inf()
            if self.found_inf:
                return loss

        prefetch_events = []
        compute_events = []

        # Fetch optimizer state
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    state = self.state[p]
                    if not len(state) == 0:
                        self.fetch_state(p)
                        event = torch.cuda.Event()
                        event.record()
                        prefetch_events.append(event)

        # Parameter update
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            for group in self.param_groups:
                if len(group['params']) == 0:
                    continue
                dtype = group['params'][0].dtype
                bias_correction = 1 if group['bias_correction'] else 0
                beta1, beta2 = group['betas']

                # assume same step across group now to simplify things
                # per parameter step can be easily support by making it tensor, or pass list into kernel
                if 'step' in group:
                    group['step'] += 1
                else:
                    group['step'] = 1

                for i, p in enumerate(group['params']):
                    # create lists for multi-tensor apply
                    p_16 = []
                    g_32, p_32, m_32, v_32 = [], [], [], []

                    if self.params_have_main_grad and hasattr(p, 'main_grad'):
                        grad = p.main_grad
                    else:
                        grad = p.grad

                    if grad is None:
                        continue
                    if grad.data.is_sparse:
                        raise RuntimeError('SpiralGPUAdam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data).float()
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                        # Copy fp16 param to fp32 param
                        if dtype == torch.float16 or dtype == torch.bfloat16:
                            state['fp32_param'] = p.detach().clone().float()
                        else:
                            state['fp32_param'] = p
                    
                        self.init_cpu_state(p)

                    state['fp32_grad'] = grad
                    grad = None

                    if dtype == torch.float16 or dtype == torch.bfloat16:
                        p_16.append(p.data)

                    g_32.append(state['fp32_grad'].data)
                    p_32.append(state['fp32_param'].data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])

                    if len(prefetch_events) > 0:
                        prefetch_events.pop(0).wait()

                    # Parameter update
                    multi_tensor_applier(self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_32, p_32, m_32, v_32],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])

                    # Copy fp32 params to fp16 params
                    if dtype == torch.float16 or dtype == torch.bfloat16:
                        # Scaling with factor `1.0` is equivalent to copy.
                        multi_tensor_applier(self.multi_tensor_scale,
                                            self._dummy_overflow_buf,
                                            [p_32, p_16],
                                            1.0)

                    event = torch.cuda.Event()
                    event.record()
                    compute_events.append(event)

        # Offload optimizer state
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("offload")):
            if len(compute_events) > 0:
                    compute_events.pop(0).wait()
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    self.offload_state(p)
                    if len(compute_events) > 0:
                        compute_events.pop(0).wait()

            self.step_event = torch.cuda.Event()
            self.step_event.record()

        return loss

    def rollback(self):
        if not self.adam_w_mode:
            raise RuntimeError("SpiralGPUAdam only supports rollback for adam_w_mode.")

        # if found_inf is True, skip rollback
        if self.half_precision and self.found_inf:
            return

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            dtype = group['params'][0].dtype
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # create lists for multi-tensor apply
            p_16 = []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                state = self.state[p]
                assert len(state) != 0, "Rollback should be call after run optimizer.step"
                self.fetch_state(p, grad_temp=True)

                if dtype == torch.float16 or dtype == torch.bfloat16:
                    p_16.append(p.data)

                g_32.append(state['fp32_grad'].data)
                p_32.append(state['fp32_param'].data)
                m_32.append(state['exp_avg'])
                v_32.append(state['exp_avg_sq'])

            if len(g_32) > 0:
                multi_tensor_rollback_adamw(
                    g_32, p_32, m_32, v_32,
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    group['step'],
                    bias_correction,
                    group['weight_decay'])
            group['step'] -= 1

            # Copy fp32 params to fp16 params
            if dtype == torch.float16 or dtype == torch.bfloat16:
                # Scaling with factor `1.0` is equivalent to copy.
                multi_tensor_applier(self.multi_tensor_scale,
                                     self._dummy_overflow_buf,
                                     [p_32, p_16],
                                     1.0)
            
            for p in group['params']:
                self.offload_state(p)

    def sync(self, found_inf=None):
        if self.step_event is not None:
            self.step_event.synchronize()
            self.step_event = None

        if found_inf is not None:
            found_inf.fill_(1.0 if self.found_inf else 0.0)


def multi_tensor_rollback_adamw(
    g_list, p_list, m_list, v_list,
    lr,
    beta1,
    beta2,
    eps,
    step,
    bias_correction,
    weight_decay,
):
    beta1_correction, beta2_correction = 1.0, 1.0
    if bias_correction == 1:
        beta1_correction = 1 - beta1 ** step
        beta2_correction = 1 - beta2 ** step
    for i, p in enumerate(p_list):
        rollback_adamw(
            g_list[i], p_list[i], m_list[i], v_list[i],
            lr,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            weight_decay,
        )


def rollback_adamw(
    g: torch.Tensor, p: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    lr,
    beta1,
    beta2,
    beta1_correction,
    beta2_correction,
    eps,
    decay,
):
    update = (m / beta1_correction) / ((v / beta2_correction).sqrt() + eps)
    update.mul_(lr)
    p.add_(update).div_(1 - lr * decay)
    v.addcmul_(g, g, value=beta2 - 1).div_(beta2)
    m.add_(g, alpha=beta1 - 1).div_(beta1)
