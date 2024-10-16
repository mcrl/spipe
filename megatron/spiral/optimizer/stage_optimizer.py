from collections import deque
import nvtx
import torch

from megatron.core import mpu
from megatron.spiral.initialize import get_thunder_cuda_manager

class SpiralStageOptimizer:

    def __init__(self, optimizers, *args, **kwargs):
        self.optimizer_list = optimizers  # do not change attr name
        self.grad_scaler = self.optimizer_list[0].grad_scaler if len(self.optimizer_list) > 0 else None
        self.inv_scale_val = self.grad_scaler.inv_scale.item() if self.grad_scaler != None else 0.0

    # Required for checkpointing
    def state_dict(self):
        state_dict = {"optimizer_list": []}
        for optimizer in self.optimizer_list:
            state_dict["optimizer_list"].append(optimizer.state_dict())
        return state_dict

    # Required for checkpointing
    def load_state_dict(self, state_dict):
        self.optimizer_list = []
        for optimizer_dict in state_dict["optimizer_list"]:
            self.optimizer.load_state_dict(optimizer_dict)
            self.optimizer_list.append(self.optimizer)

    def gather_model_params(self, args, timers):
        """
        For the case of a non-distributed-optimizer, there is nothing to
        do here.
        """
        pass

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        param_groups = []
        for optimizer in self.optimizer_list:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    param_groups = property(_get_param_groups)

    def get_loss_scale(self):
        return self.grad_scaler.scale
    
    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @nvtx.annotate("step", color="cyan")
    def step(self, idx, event_query, args, timers):
        event_long = -1
        if event_query != None:
            event_long = get_thunder_cuda_manager().get_event(event_query).cuda_event
        
        self.optimizer_list[idx].optimizer.set_inv_scale(self.inv_scale_val)
        self.optimizer_list[idx].optimizer.set_event_long(event_long)
        self.optimizer_list[idx].step(args, timers)

    @nvtx.annotate("join_step", color="red")
    def join_step(self):
        spiral_stage_optimizer_step_returns = deque()
        local_found_inf = 0
        found_inf_flag = False

        # sync all optimizers
        for optimizer in reversed(self.optimizer_list):
            found_inf = torch.FloatTensor([0])
            optimizer.optimizer.sync(found_inf)
            local_found_inf += found_inf.item()

        # update grad scaler
        if self.grad_scaler:
            found_inf = torch.cuda.FloatTensor([local_found_inf])
            torch.distributed.all_reduce(found_inf,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
            found_inf_flag = (found_inf.item() > 0)

            self.grad_scaler.update(found_inf_flag)
            self.inv_scale_val = self.grad_scaler.inv_scale.item()

        spiral_stage_optimizer_step_returns.appendleft((not found_inf_flag, None, None))

        return self._process_step_returns(spiral_stage_optimizer_step_returns)

    def _process_step_returns(self, step_rets: list):
        """Static method to reduce the return values of individual optimizer steps."""
        update_successful_values, grad_norm_values, num_zeros_in_grad_values = zip(
            *step_rets
        )

        # Calculate r_update_successful
        r_update_successful = all(update_successful_values)

        # Calculate r_grad_norm
        # TODO (SpiralPipe) This is a temporary solution by simply averaging the grad_norms
        valid_grad_norm_values = list(filter(lambda x: x is not None, grad_norm_values))
        r_grad_norm = (
            sum(valid_grad_norm_values) / len(grad_norm_values)
            if valid_grad_norm_values
            else None
        )

        # Calculate r_num_zeros_in_grad
        valid_num_zeros_in_grad_values = list(
            filter(lambda x: x is not None, num_zeros_in_grad_values)
        )
        r_num_zeros_in_grad = (
            sum(valid_num_zeros_in_grad_values)
            if valid_num_zeros_in_grad_values
            else None
        )

        return r_update_successful, r_grad_norm, r_num_zeros_in_grad

    def __getitem__(self, idx):
        return self.optimizer_list[idx]


class SpiralStageOptimizerParamScheduler:

    def __init__(self, optimizer_param_schedulers, *args, **kwargs):
        self.optimizer_param_scheduler_list = optimizer_param_schedulers

    def __getitem__(self, idx):
        return self.optimizer_param_scheduler_list[idx]
