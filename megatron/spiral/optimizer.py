class SpiralStageOptimizer:

    def __init__(self, optimizers, *args, **kwargs):
        self.optimizer_list = optimizers  # do not change attr name

    def get_total_param_groups(self):
        param_groups = []
        for optimizer in self.optimizer_list:
            param_groups.extend(optimizer.param_groups)
        return param_groups

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

    def __getitem__(self, idx):
        return self.optimizer_list[idx]


class SpiralStageOptimizerParamScheduler:

    def __init__(self, optimizer_param_schedulers, *args, **kwargs):
        self.optimizer_param_scheduler_list = optimizer_param_schedulers

    def __getitem__(self, idx):
        return self.optimizer_param_scheduler_list[idx]
