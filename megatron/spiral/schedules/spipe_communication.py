from typing import Callable, List, Union, Tuple

import torch
from torch._C._distributed_c10d import Work

from megatron.core import mpu
import megatron.spiral.p2p_communication as spiral_p2p


# Types
Shape = Union[List[int], torch.Size]


# Handle for nop
class NOP_Wait:
    @staticmethod
    def wait():
        pass


def comm_activation(
    output_tensor: torch.Tensor,
    recvs: List[Tuple[torch.Tensor, List[Work]]],
    fid: int,
    mid: int,
    nm: int,
    tensor_shape: Shape,
    dtype: torch.dtype,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = False,
    timers: Callable = None,
):
    """Communicate activation.

    Enqueue received input activation to `recvs`, if any.
    """
    skip_recv = (
        mpu.is_pipeline_first_stage()
        and mid < mpu.get_pipeline_model_parallel_world_size() - 1
    ) or (
        mpu.get_pipeline_model_parallel_rank() == 0
        and fid == mpu.get_spiral_forward_virtual_size() - 1
        and mid >= mpu.get_pipeline_model_parallel_world_size() - 1
    )
    skip_send = mpu.is_pipeline_last_stage()

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
    tensor_shape: Shape,
    dtype: torch.dtype,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = False,
    timers: Callable = None,
):
    """Communicate activation gradients.

    Enqueue received output activation gradients to `recvs`, if any.
    """
    skip_send = mpu.is_pipeline_first_stage()
    skip_recv = (bid == 0 and mid == nm - 1) or (
        mpu.is_pipeline_last_stage()
        and mid < mpu.get_pipeline_model_parallel_world_size() - 1
    ) or (
        mpu.get_pipeline_model_parallel_rank() == 0
        and bid == 0
        and mid >= mpu.get_pipeline_model_parallel_world_size() - 1
    )

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


def fwd_init_recvs(
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
    if mpu.is_pipeline_first_stage():
        # Must insert to head because received tensors can precede otherwise
        recvs.insert(0, (None, [NOP_Wait]))
    elif fid == 0 and mid == 0:
        recv, reqs = spiral_p2p.recv_prev(
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.insert(0, (recv, reqs))


def bwd_init_recvs(
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
    if mpu.is_pipeline_last_stage():
        # Must insert to head because received tensors can precede otherwise
        recvs.insert(0, (None, [NOP_Wait]))
