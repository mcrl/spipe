from megatron.optimizer import FP32Optimizer


class SpiralOptimizer(FP32Optimizer):

    def __init__(self, optimizer_list, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp,
                 models):

        super(SpiralOptimizer, self).__init__(
            optimizer_list[0], clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            models)

        self.optimizer_list = optimizer_list

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
        return self.optimizer_list.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer_list.load_state_dict(state_dict)
        self.set_bwd_stage(0)
