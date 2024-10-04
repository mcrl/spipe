from typing import Callable, List, Union, Tuple

import torch
from torch._C._distributed_c10d import Work

from megatron.core import mpu
import megatron.spiral.p2p_communication as spiral_p2p


# Types
Shape = Union[List[int], torch.Size]


def comm_activation(
    output_tensor: torch.Tensor,
    recvs: List[Tuple[torch.Tensor, List[Work]]],
    fid: int,
    mid: int,
    nm: int,
    dtype: torch.dtype,
    tensor_shape: Shape,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = False,
    timers: Callable = None,
):
    """Communicate activation.

    Enqueue received input activation to `recvs`, if any.
    """
    skip_recv = mpu.is_pipeline_first_stage() and mid < mpu.get_pipeline_model_parallel_world_size() - 1

    if skip_recv:
        spiral_p2p.send_next(
            output_tensor,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
    else:
        recv, reqs = spiral_p2p.send_next_recv_prev(
            output_tensor,
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.append((recv, reqs))


def comm_activation_grad(
    input_tensor_grad: torch.Tensor,
    recvs: List[Tuple[torch.Tensor, List[Work]]],
    bid: int,
    mid: int,
    nm: int,
    dtype: torch.dtype,
    tensor_shape: Shape,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = False,
    timers: Callable = None,
):
    """Communicate activation gradients.

    Enqueue received output activation gradients to `recvs`, if any.
    """
    skip_send = mpu.is_pipeline_first_stage() and mid >= nm - mpu.get_pipeline_model_parallel_world_size()
    skip_recv = bid == 0 and mid == nm - 1

    if skip_send and skip_recv:
        pass
    elif skip_send:
        recv, reqs = spiral_p2p.recv_prev(
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.append((recv, reqs))
    elif skip_recv:
        spiral_p2p.send_next(
            input_tensor_grad,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
    else:
        recv, reqs = spiral_p2p.send_next_recv_prev(
            input_tensor_grad,
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.append((recv, reqs))