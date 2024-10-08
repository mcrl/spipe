import amp_C
import torch
import warnings
import nvtx

from megatron.core import tensor_parallel
from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.spiral.optimizer.cpu_adam import SpiralCPUAdam


class SpiralFloat16Optimizer(Float16OptimizerWithFloat16Params):
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

        if type(self.optimizer) != SpiralCPUAdam:
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

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale.cpu())

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
    def step(self, args, timers, offload_grad_ev_long=-1):
        if type(self.optimizer) != SpiralCPUAdam:
            return super().step(args, timers)
        else:
            # self.optimizer.set_grad_scaler(self.grad_scaler)
            self.optimizer.set_event_long(offload_grad_ev_long)
            self.optimizer.step()
            return 0


class SpiralFP32Optimizer(FP32Optimizer):
    @torch.no_grad()
    def step(self, args, timers, offload_grad_ev_long=-1):
        if type(self.optimizer) != SpiralCPUAdam:
            return super().step(args, timers)
        else:
            self.optimizer.set_event_long(offload_grad_ev_long)
            self.optimizer.step()
            return 0
