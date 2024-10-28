import torch

from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.spiral.optimizer.cpu_adam import SpiralCPUAdam
from megatron.spiral.initialize import get_thunder_cuda_manager
from megatron.spiral.utils import is_spiral_param


def is_cpu_optimizer(optimizer):
    return isinstance(optimizer, SpiralCPUAdam)


class SpiralFloat16Optimizer(Float16OptimizerWithFloat16Params):
    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):

        for group in optimizer.param_groups:
            for p in group["params"]:
                if is_spiral_param(p):
                    p.fetch(non_blocking=False)

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models)

        for group in self.float16_groups:
            for p in group:
                if is_spiral_param(p):
                    p.free()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)

        return found_inf_flag

    @torch.no_grad()
    def step(self, args, timers):
        if is_cpu_optimizer(self.optimizer):
            return self.optimizer.step()
        else:
            super().step(args, timers)

            for group in self.float16_groups:
                for p in group:
                    if is_spiral_param(p):
                        p.offload(non_blocking=True)
                        
            self.offload_event = get_thunder_cuda_manager().Event(
                "offload", None, tag="offload"
            )
            if get_thunder_cuda_manager().record_event(self.offload_event) == -1:
                raise RuntimeError("record_event failed")

    def sync(self, found_inf=None):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.sync(found_inf)
        else:
            # TODO: check whether wait offload_event or only step_event
            if self.offload_event is not None:
                if get_thunder_cuda_manager().wait_event(self.offload_event) == -1:
                    raise RuntimeError("wait_event failed")
                self.offload_event = None

            found_inf.data = self.found_inf.to(device = found_inf.device, non_blocking=False)

    @torch.no_grad()
    def rollback(self, sync=False):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.rollback(sync)
        else:
            if self.found_inf.item() == 0:
                self.optimizer.rollback()
                self._copy_main_params_to_model_params()

                for group in self.float16_groups:
                    for p in group:
                        if is_spiral_param(p):
                            p.offload(non_blocking=not sync)


class SpiralFP32Optimizer(FP32Optimizer):
    @torch.no_grad()
    def step(self, args, timers):
        if is_cpu_optimizer(self.optimizer):
            return self.optimizer.step()
        else:
            super().step(args, timers)

            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if is_spiral_param(p):
                        p.offload(non_blocking=True)
                        
            self.offload_event = get_thunder_cuda_manager().Event(
                "offload", None, tag="offload"
            )
            if get_thunder_cuda_manager().record_event(self.offload_event) == -1:
                raise RuntimeError("record_event failed")

    def sync(self, found_inf=None):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.sync(found_inf)
        else:
            # TODO: check whether wait offload_event or only step_event
            if self.offload_event is not None:
                if get_thunder_cuda_manager().wait_event(self.offload_event) == -1:
                    raise RuntimeError("wait_event failed")
                self.offload_event = None

    def rollback(self, sync=False):
        # No need to rollback for fp32 param
        pass
