import torch
import warnings

from megatron.core import tensor_parallel
from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.spiral.optimizer.cpu_adam import SpiralCPUAdam
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

        # Update across all model parallel instances.
        # TODO (SpiralPipe) Implement all_reduce for found_inf
        warnings.warn("SpiralPipe currently does not implement all_reduce for found_inf. This is a critical bug when using TP or DP with SpiralPipe. We must implement this later on.")
        # torch.distributed.all_reduce(self.found_inf,
        #                             op=torch.distributed.ReduceOp.MAX,
        #                             group=self.get_model_parallel_group())

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)

        return found_inf_flag

    @torch.no_grad()
    def step(self, args, timers):
        if is_cpu_optimizer(self.optimizer):
            return self.optimizer.step()
        else:
            # parameter is freed in schedule
            # TODO: don't need to free in schedule
            for group in self.float16_groups:
                for p in group:
                    if is_spiral_param(p):
                        p.fetch(non_blocking=False)

            super().step(args, timers)
        
            for group in self.float16_groups:
                for p in group:
                    if is_spiral_param(p):
                        p.offload()
                        p.free()
                        
    def sync(self, found_inf=None):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.sync(found_inf)
        else:
            # TODO: make event in step and wait until event finished
            found_inf.data = self.found_inf.to(device = found_inf.device, non_blocking=False)

    def rollback(self, sync=False):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.rollback(sync)
        else:
            # TODO: need to make rollback in FusedAdam
            pass


class SpiralFP32Optimizer(FP32Optimizer):
    @torch.no_grad()
    def step(self, args, timers):
        if is_cpu_optimizer(self.optimizer):
            return self.optimizer.step()
        else:
            # parameter is freed in schedule
            # TODO: don't need to free in schedule
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if is_spiral_param(p):
                        p.fetch(non_blocking=False)

            super().step(args, timers)
            
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if is_spiral_param(p):
                        p.offload()
                        p.free()

    def sync(self, found_inf=None):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.sync(found_inf)
        else:
            # TODO: make event in step and wait until event finished
            pass

    def rollback(self, sync=False):
        if is_cpu_optimizer(self.optimizer):
            self.optimizer.rollback(sync)
        else:
            # TODO: need to make rollback in FusedAdam
            pass
