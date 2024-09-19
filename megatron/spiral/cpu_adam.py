import torch
from cpuinfo import get_cpu_info
from megatron import get_args

from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le

from .op_builder.cpu_adam import SpiralCPUAdamBuilder

class SpiralCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(
        self,
        model_params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adamw_mode=True,
        fp32_optimizer_states=True,
    ):
        """Fast vectorized implementation of two variations of Adam optimizer on CPU:

        * Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        * AdamW: Fixing Weight Decay Regularization in Adam (https://arxiv.org/abs/1711.05101)

        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
        In order to apply this optimizer, the model requires to have its master parameter (in FP32)
        reside on the CPU memory.

        To train on a heterogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
        the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
        with minimal impact on training throughput. DeepSpeedCPUAdam plays an important role to minimize
        the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
        (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.

        For calling step function, there are two options available: (1) update optimizer's states and (2) update
        optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
        option can bring 30% higher throughput than the doing the copy separately using option one.


        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.


        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: select between Adam and AdamW implementations (default: AdamW)
            fp32_optimizer_states: creates momentum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """

        default_args = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            amsgrad=amsgrad,
        )
        super(SpiralCPUAdam, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = (
            cpu_info["vendor_id_raw"].lower()
            if "vendor_id_raw" in cpu_info
            else "unknown"
        )
        if "amd" in self.cpu_vendor:
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group["params"]):
                    if p.dtype == torch.half:
                        logger.warning(
                            "FP16 params for CPUAdam may not work on AMD CPUs"
                        )
                        break
                else:
                    continue
                break

        self.opt_id = SpiralCPUAdam.optimizer_id
        SpiralCPUAdam.optimizer_id = SpiralCPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.nparams = sum(len(pg["params"]) for pg in self.param_groups)
        self.pool_size = get_args().spiral_stage_optimizer_pool_size
        self.ds_opt_adam = SpiralCPUAdamBuilder().load()
        self.ds_opt_adam.create_adam(
            self.opt_id,
            len(self.param_groups),
            self.nparams,
            self.pool_size,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            adamw_mode,
            should_log_le("info"),
        )

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        self.ds_opt_adam.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(SpiralCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None, **kwargs):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.
            offload_grad_ev: Gradient offload event to synchronize before using grads to update weights

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device("cpu")

        # converting the fp16 params to a group of parameter
        if type(fp16_param_groups) is list:
            if type(fp16_param_groups[0]) is not list:
                fp16_param_groups = [fp16_param_groups]
        elif fp16_param_groups is not None:
            fp16_param_groups = [[fp16_param_groups]]

        # get spiral kwargs
        ev_long = kwargs.get("spiral_offload_grad_ev_long", -1)

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group["params"]):

                if p.grad is None:
                    print(f"[Warning] Optimizer#{self.opt_id} skipped grp#{group_id} param#{param_id} step due to grad={p.grad}")
                    continue

                assert p.device == device, (
                    f"CPUAdam param is on {p.device} and must be 'cpu', make "
                    "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
                )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # print(f'group {group_id} param {param_id} = {p.numel()}')
                    state["step"] = 0

                    # use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state["exp_avg"] = torch.zeros_like(
                        p.data, dtype=state_dtype, device=device
                    )
                    # memory_format=torch.preserve_format)
                    # gradient variances
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, dtype=state_dtype, device=device
                    )
                    # memory_format=torch.preserve_format)

                state["step"] += 1
                beta1, beta2 = group["betas"]

                if fp16_param_groups is not None:
                    self.ds_opt_adam.adam_update_copy(
                        self.opt_id,
                        group_id,
                        state["step"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["weight_decay"],
                        group["bias_correction"],
                        p.data,
                        p.grad.data,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        fp16_param_groups[group_id][param_id].data,
                        ev_long,
                    )
                else:
                    self.ds_opt_adam.adam_update(
                        self.opt_id,
                        group_id,
                        state["step"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["weight_decay"],
                        group["bias_correction"],
                        p.data,
                        p.grad.data,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        ev_long,
                    )
        return loss

    def sync(self):
        self.ds_opt_adam.adam_sync(self.opt_id)
