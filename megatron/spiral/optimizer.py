from concurrent.futures import ThreadPoolExecutor
from megatron.optimizer import FP32Optimizer


class SpiralStageOptimizer(FP32Optimizer):

    def __init__(
        self,
        optimizer_list,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
        models,
    ):
        super(SpiralStageOptimizer, self).__init__(
            optimizer_list[0],
            clip_grad,
            log_num_zeros_in_grad,
            params_have_main_grad,
            use_contiguous_buffers_in_local_ddp,
            models,
        )

        self.optimizer_list = optimizer_list
        self.optimizer_thread_pool = ThreadPoolExecutor(max_workers=len(optimizer_list))

    def set_bwd_stage(self, stage_id):
        assert stage_id < len(self.optimizer_list)
        self.optimizer = self.optimizer_list[stage_id]

    def step(self, args, timers):
        return super().step(args, timers)

    def get_total_param_groups(self):
        param_groups = []
        for optimizer in self.optimizer_list:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def state_dict(self):
        state_dict = {
            'optimizer_list': []
        }
        for optimizer in self.optimizer_list:
            state_dict['optimizer_list'].append(optimizer.state_dict())
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer_list = []
        for optimizer_dict in state_dict['optimizer_list']:
            self.optimizer.load_state_dict(optimizer_dict)
            self.optimizer_list.append(self.optimizer)
