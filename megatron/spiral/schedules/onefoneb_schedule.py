import contextlib
import warnings
from typing import Callable, Iterator, List, Optional, Union, Tuple
from collections import deque
from queue import Queue
from enum import Enum
import threading

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._C._distributed_c10d import Work

from megatron import get_args, get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type, get_attr_wrapped_model
from megatron.core.pipeline_parallel import forward_step, backward_step, p2p_communication
from megatron.spiral.initialize import get_thunder_cuda_manager
from megatron.spiral.debug import spiral_print
from megatron.spiral.init_context import SpiralParamStatus, set_module_spiral_status
from megatron.spiral.generic import ContextManagers


class PHASE(Enum):
    WARMUP = 1
    STEADY = 2
    COOLDOWN = 3


# Types
Shape = Union[List[int], torch.Size]

# Constants
_PHASE: PHASE = None
_DEBUG_SCHEDULE = True


def onefoneb_schedule(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    dtype: torch.dtype,
    tensor_shape: Shape,
    decoder_seq_length: Optional[int] = None,
    grad_scaler: Callable = None,
    sequence_parallel: bool = False,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = False,
    forward_only: bool = False,
    timers: Callable = None,
    collect_non_loss_data: bool = False,
    enable_autocast: bool = False,
    deallocate_pipeline_outputs: bool = False,
    no_sync_func: Optional[Callable] = None,
    grad_sync_func: Optional[Callable] = None,
    param_sync_func: Optional[Callable] = None,
    **kwargs,
):
    """Run interleaved 1f1B schedule, with communication between pipeline stages as needed.
    Returns dictionary with losses if the last stage, empty dict otherwise."""
