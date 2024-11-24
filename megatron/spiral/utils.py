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

_GPU_LATENCY_LIST = None

class DummyMaxList:
    def __init__(self):
        self._list = []

    def append(self, item):
        from megatron.core import mpu
        max_lat = torch.tensor(item, device="cuda")
        torch.distributed.all_reduce(max_lat, group=mpu.get_pipeline_model_parallel_group(), op=torch.distributed.ReduceOp.MAX)
        max_lat = max_lat.item()
        self._list.append(max_lat)

    def clear(self):
        self._list.clear()

    def get_avg(self):
        # print(f"averaging... {self._list}")
        return sum(self._list) / len(self._list) if self._list else None


def create_gpu_latency_list():
    global _GPU_LATENCY_LIST
    if _GPU_LATENCY_LIST is None:
        _GPU_LATENCY_LIST = DummyMaxList()


def get_gpu_latency_list():
    global _GPU_LATENCY_LIST
    assert _GPU_LATENCY_LIST is not None, "GPU latency list not created"
    return _GPU_LATENCY_LIST