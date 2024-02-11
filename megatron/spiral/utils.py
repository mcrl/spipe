import torch


def is_spiral_param(param):
    if not torch.is_tensor(param):
        return False
    return hasattr(param, "spiral_tensor")
