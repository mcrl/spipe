import torch

from megatron.core import tensor_parallel
from megatron.optimizer.optimizer import MixedPrecisionOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.optimizer.optimizer import _zero_grad_group_helper
from megatron.spiral.optimizer.cpu_adam import SpiralCPUAdam
from megatron.spiral.utils import is_spiral_param


class SpiralFloat16Optimizer(MixedPrecisionOptimizer):

    @torch.no_grad()
    def step(self, args, timers):
        if type(self.optimizer) == SpiralCPUAdam:
            return self.optimizer.step()
        else:
            result = self.optimizer.step()
            self.found_inf.fill_(0.0 if result else 1.0)
            self.step_event = torch.cuda.Event()
            self.step_event.record()
            return result

    def sync(self, found_inf=None):
        if type(self.optimizer) == SpiralCPUAdam:
            self.optimizer.sync(found_inf)
        else:
            if self.step_event is not None:
                self.step_event.synchronize()
                self.step_event = None
            found_inf.data = self.found_inf.to(device = found_inf.device, non_blocking=False)

    @torch.no_grad()
    def rollback(self, sync=False):
        if type(self.optimizer) == SpiralCPUAdam:
            self.optimizer.rollback(sync)
        else:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if is_spiral_param(p):
                        p.fetch(non_blocking=not sync)

            self.optimizer.rollback()

            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if is_spiral_param(p):
                        p.offload(non_blocking=not sync)

    def zero_grad(self, set_to_none=True):
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)
    
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class SpiralFP32Optimizer(FP32Optimizer):
    @torch.no_grad()
    def step(self, args, timers):
        if type(self.optimizer) == SpiralCPUAdam:
            return self.optimizer.step()
        else:
            result = self.optimizer.step()
            self.step_event = torch.cuda.Event()
            self.step_event.record()
            return result

    def sync(self, found_inf=None):
        if type(self.optimizer) == SpiralCPUAdam:
            self.optimizer.sync(found_inf)
        else:
            if self.step_event is not None:
                self.step_event.synchronize()
                self.step_event = None

    def rollback(self, sync=False):
        # No need to rollback for fp32 param
        pass


class DeepSpeedFloat16Optimizer(Float16OptimizerWithFloat16Params):
    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models)

        # Re-initialize found_inf to cpu tensor
        if self.grad_scaler:
            self.found_inf = torch.FloatTensor([0.0])

        # Re-initialize _dummy_overflow_buf to cpu tensor
        if not bf16:
            self._dummy_overflow_buf = torch.IntTensor([0])

        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                # NOTE (SpiralPipe) cpu params do not need to require grad

                # float16 params:
                if param.type() in ['torch.HalfTensor',
                                    'torch.BFloat16Tensor']:
                    float16_params_this_group.append(param)
                    # Create a copy
                    main_param = param.detach().clone().float()
                    # Copy tensor model parallel attributes.
                    tensor_parallel.copy_tensor_model_parallel_attributes(main_param,
                                                                        param)
                    if hasattr(param, 'shared'):
                        main_param.shared = param.shared
                    # Replace the optimizer params with the new fp32 copy.
                    param_group['params'][i] = main_param

                    fp32_from_float16_params_this_group.append(main_param)
                    # Reset existing state dict key to the new main param.
                    if param in self.optimizer.state:
                        self.optimizer.state[main_param] \
                            = self.optimizer.state.pop(param)
                # fp32 params.
                elif param.type() == 'torch.FloatTensor':
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param

                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.FloatTensor,  '
                                    'torch.HalfTensor, or '
                                    'torch.BFloat16Tensor. '
                                    'Received {}'.format(param.type()))

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(
                fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale.cpu())

        cuda_found_inf = self.found_inf.to(device='cuda')
        torch.distributed.all_reduce(cuda_found_inf,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=self.get_model_parallel_group())

        # Check for nan.
        found_inf_flag = (cuda_found_inf.item() > 0)

        return found_inf_flag
