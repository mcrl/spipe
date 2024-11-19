from typing import Callable, List, Union, Tuple
from enum import Enum

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


class CaseColor(Enum):
    LightGreen = 1
    Green = 2
    Purple = 3
    Pink = 4
    Red = 5
    Else = 6


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
    omit_send_reqs: bool = False,
):
    """Communicate activation.

    Enqueue received input activation to `recvs`, if any.
    Refer to sdrv_schedule.xlsx for detailed description.
    """

    case_color: CaseColor = None
    if (
        mpu.is_pipeline_first_stage()
        and mid < mpu.get_pipeline_model_parallel_world_size() - 1
    ):
        # @color purple
        # first pipeline stage & microbatch idx < ppsize-1 -> send next
        case_color = CaseColor.Purple
    elif (
        fid == mpu.get_spiral_forward_virtual_size() - 1
        and not mpu.is_pipeline_last_stage()
    ):
        if mid == nm - 1:
            # @color lightgreen
            # (last fwd stage & !last pipeline stage) & (last microbatch) -> send next
            case_color = CaseColor.LightGreen
        elif (mid >= mpu.get_pipeline_model_parallel_world_size() - 1) and (
            mpu.get_pipeline_model_parallel_rank() == 0
        ):
            # @color pink
            # (last fwd stage & !last pipeline stage) & (microbatch idx >= ppsize-1) & !@lightgreen & first pipeline rank -> send next
            assert mid < nm - 1
            case_color = CaseColor.Pink
        else:
            # @color Else
            # -> send next, recv prev
            case_color = CaseColor.Else
    elif mpu.is_pipeline_last_stage():
        if mid == nm - 1:
            # @color green
            # last pipeline stage -> pass
            case_color = CaseColor.Green
        else:
            # @color red
            # last pipeline stage & microbatch idx < nm-1 -> recv prev
            assert mid < nm - 1
            case_color = CaseColor.Red
    else:
        # @color Else
        # -> send next, recv prev
        case_color = CaseColor.Else

    if (
        (case_color == CaseColor.Purple)
        or (case_color == CaseColor.LightGreen)
        or (case_color == CaseColor.Pink)
    ):
        spiral_p2p.send_next(
            output_tensor,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
            omit_send_reqs=omit_send_reqs,
        )
    elif case_color == CaseColor.Red:
        recv, reqs = spiral_p2p.recv_prev(
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.append((recv, reqs))
    elif case_color == CaseColor.Else:
        recv, reqs = spiral_p2p.send_next_recv_prev(
            output_tensor,
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
            omit_send_reqs=omit_send_reqs,
        )
        recvs.append((recv, reqs))
    else:
        assert case_color == CaseColor.Green, f"Invalid case_color: {case_color}"
        pass


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
    omit_send_reqs: bool = False,
):
    """Communicate activation gradients.

    Enqueue received output activation gradients to `recvs`, if any.
    """
    case_color: CaseColor = None
    if (
        mpu.is_pipeline_last_stage()
        and mid < mpu.get_pipeline_model_parallel_world_size() - 1
    ):
        # @color purple
        # last pipeline stage & microbatch idx < ppsize-1 -> send prev
        case_color = CaseColor.Purple
    elif (
        bid == 0
        and not mpu.is_pipeline_first_stage()
    ):
        if mid == nm - 1:
            # @color lightgreen
            # (first bwd stage & !first pipeline stage) & (last microbatch) -> send prev
            case_color = CaseColor.LightGreen
        elif (mid >= mpu.get_pipeline_model_parallel_world_size() - 1) and (
            mpu.get_pipeline_model_parallel_rank()
            == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            # @color pink
            # (first bwd stage & !first pipeline stage) & (microbatch idx >= ppsize-1) & !@lightgreen & last pipeline rank -> send prev
            assert mid < nm - 1
            case_color = CaseColor.Pink
        else:
            # @color Else
            # -> send prev, recv next
            case_color = CaseColor.Else
    elif mpu.is_pipeline_first_stage():
        if mid == nm - 1:
            # @color green
            # first pipeline stage -> pass
            case_color = CaseColor.Green
        else:
            # @color red
            # last pipeline stage & microbatch idx < nm-1 -> recv next
            assert mid < nm - 1
            case_color = CaseColor.Red
    else:
        # @color Else
        # -> send prev, recv next
        case_color = CaseColor.Else

    if (
        (case_color == CaseColor.Purple)
        or (case_color == CaseColor.LightGreen)
        or (case_color == CaseColor.Pink)
    ):
        spiral_p2p.send_prev(
            input_tensor_grad,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
            omit_send_reqs=omit_send_reqs,
        )
    elif case_color == CaseColor.Red:
        recv, reqs = spiral_p2p.recv_next(
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.append((recv, reqs))
    elif case_color == CaseColor.Else:
        recv, reqs = spiral_p2p.send_prev_recv_next(
            input_tensor_grad,
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
            omit_send_reqs=omit_send_reqs,
        )
        recvs.append((recv, reqs))
    else:
        assert case_color == CaseColor.Green, f"Invalid case_color: {case_color}"
        pass


def fwd_init_recvs(
    recvs: List[Tuple[torch.Tensor, List[Work]]],
    fid: int,
    mid: int,
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
    dtype: torch.dtype,
    tensor_shape: Shape,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = False,
    timers: Callable = None,
):
    if mpu.is_pipeline_last_stage():
        # Must insert to head because received tensors can precede otherwise
        recvs.insert(0, (None, [NOP_Wait]))
    elif bid == mpu.get_spiral_backward_virtual_size() - 1 and mid == 0:
        recv, reqs = spiral_p2p.recv_next(
            tensor_shape,
            dtype,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
        )
        recvs.insert(0, (recv, reqs))
