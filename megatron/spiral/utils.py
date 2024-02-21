import math
from functools import reduce

import torch


def is_spiral_param(param):
    if not torch.is_tensor(param):
        return False
    return hasattr(param, "spiral_id")


def num_spiral_params(module):
    num_params = 0
    for param in module.parameters(recurse=True):
        if is_spiral_param(param):
            num_params += 1
    return num_params


try:
    import math.lcm as lcm
except ImportError:
    def lcm(*args):
        assert len(args) > 0, "lcm() requires at least one argument"
        assert len(args) < 3, "For Python < 3.9, math.gcd() currently only supports upto two arguments"
        return abs(reduce(lambda acc, cur: acc * cur, args, 1) // math.gcd(*args))