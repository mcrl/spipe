import torch

from megatron.core import mpu
from megatron.utils import spiral_debug

import spiral_helper

global thunder_group

class SpiralBackend:
    def __init__(self):
        
        # spiral_helper.check_mpi_initialized()
        # spiral_helper.check_sem_shm()

        torch_group = mpu.get_pipeline_model_parallel_group() # TODOMCRL: may change
        ranks = frozenset(torch.distributed.get_process_group_ranks(torch_group))
        # # thunder_group = self.group_cache.get(ranks, None)

        global thunder_group
        thunder_group = spiral_helper.Comm(sorted(ranks))