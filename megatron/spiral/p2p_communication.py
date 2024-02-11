from typing import Optional, List, Union, Callable, Tuple

import nvtx
import torch
from torch._C._distributed_c10d import Work

from megatron.core import mpu
from megatron.spiral.debug import spiral_print


# Types
Shape = Union[List[int], torch.Size]


def _batched_p2p_ops(
    *,
    tensor_sends: Optional[List[torch.Tensor]],
    tensor_recvs: Optional[List[torch.Tensor]],
    send_ranks: Optional[List[int]],
    recv_ranks: Optional[List[int]],
    group: torch.distributed.ProcessGroup,
):
    ops = []
    if tensor_sends is not None:
        assert send_ranks is not None and len(tensor_sends) == len(send_ranks)
        for tensor, rank in zip(tensor_sends, send_ranks):
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor, rank, group=group
            )
            ops.append(send_op)
    if tensor_recvs is not None:
        assert recv_ranks is not None and len(tensor_recvs) == len(recv_ranks)
        for tensor, rank in zip(tensor_recvs, recv_ranks):
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor, rank, group=group
            )
            ops.append(recv_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_sends: Optional[List[torch.Tensor]],
    tensor_recvs: Optional[List[torch.Tensor]],
    send_ranks: Optional[List[int]],
    recv_ranks: Optional[List[int]],
    group: torch.distributed.ProcessGroup,
):
    reqs = []
    if mpu.get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_sends is not None:
            assert send_ranks is not None and len(tensor_sends) == len(send_ranks)
            for tensor, rank in zip(tensor_sends, send_ranks):
                send_req = torch.distributed.isend(tensor, rank, group=group)
                reqs.append(send_req)
        if tensor_recvs is not None:
            assert recv_ranks is not None and len(tensor_recvs) == len(recv_ranks)
            for tensor, rank in zip(tensor_recvs, recv_ranks):
                recv_req = torch.distributed.irecv(tensor, src=rank, group=group)
                reqs.append(recv_req)
    else:
        if tensor_recvs is not None:
            assert recv_ranks is not None and len(tensor_recvs) == len(recv_ranks)
            for tensor, rank in zip(tensor_recvs, recv_ranks):
                recv_req = torch.distributed.irecv(tensor, src=rank, group=group)
                reqs.append(recv_req)
        if tensor_sends is not None:
            assert send_ranks is not None and len(tensor_sends) == len(send_ranks)
            for tensor, rank in zip(tensor_sends, send_ranks):
                send_req = torch.distributed.isend(tensor, rank, group=group)
                reqs.append(send_req)
    return reqs


def _communicate(
    *,
    tensor_sends: Optional[List[torch.Tensor]],
    send_ranks: Optional[List[int]],
    recv_ranks: Optional[List[int]],
    tensor_shape: Shape,
    group: Optional[torch.distributed.ProcessGroup],
    batch_p2p_comm: bool = True,
    wait_on_reqs: bool = False,
    dtype: Optional[torch.dtype],
    variable_seq_lengths: bool = False,
    use_ring_exchange_p2p: bool = False,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[Work]]]:

    # Create placeholder for receive if needed
    tensor_recvs = None

    # This will come from config in the next version, for now hard
    # code it here to match existing functionality.
    batch_p2p_sync = True

    # set group
    if group is None:
        group = mpu.get_pipeline_model_parallel_group()

    if not variable_seq_lengths:
        shape = tensor_shape
    else:
        # TODO (mcrl) implement _communicate_shape
        raise NotImplementedError(
            "SpiralPipe does not support variable sequence length is not supported yet"
        )

    if recv_ranks:
        if dtype is None:
            raise RuntimeError("dtype must be provided if recv_ranks is not None")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_ranks is not None."
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recvs = [
            torch.empty(
                shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
            for _ in recv_ranks
        ]

    if use_ring_exchange_p2p:
        # TODO (mcrl) support ring exchange
        raise NotImplementedError("SpiralPipe does not support ring exchange yet")
        # def _ring_exchange_wrapper(**kwargs):
        #     torch.distributed.ring_exchange(**kwargs)
        #     return []
        # p2p_func = _ring_exchange_wrapper
    elif batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_sends=tensor_sends,
        tensor_recvs=tensor_recvs,
        send_ranks=send_ranks,
        recv_ranks=recv_ranks,
        group=group if group is not None else mpu.get_pipeline_model_parallel_group(),
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if batch_p2p_comm and batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recvs, reqs


@nvtx.annotate("recv_input_tensor", color="red")
def recv_input_tensor(
    tensor_shape: Shape,
    dtype: torch.dtype,
    batch_p2p_comm: bool = True,
    overlap_p2p_comm: bool = False,
    timers: Callable = None,
) -> Tuple[torch.Tensor, Optional[List[Work]]]:

    if timers is not None:
        timers("recv_input_tensor", log_level=2).start()

    recv_rank = None
    if mpu.get_spiral_pipeline_parallel_forward_virtual_rank() == 0:
        recv_rank = mpu.get_pipeline_model_parallel_prev_rank()
    else:
        local_world_size = mpu.get_spiral_pipeline_parallel_intra_size()
        local_rank = mpu.get_spiral_pipeline_parallel_intra_rank()
        recv_rank = (
            mpu.get_pipeline_model_parallel_rank()
            // local_world_size
            * local_world_size
            + (local_rank - 1) % local_world_size
        )

    # TODO (mcrl) delete
    spiral_print(f"recv input_tensor({tensor_shape}) from rank {recv_rank}")

    [input_tensor], reqs = _communicate(
        tensor_sends=None,
        send_ranks=None,
        recv_ranks=[recv_rank],
        tensor_shape=tensor_shape,
        group=mpu.get_pipeline_model_parallel_group(),
        batch_p2p_comm=batch_p2p_comm,
        wait_on_reqs=not overlap_p2p_comm,
        dtype=dtype,
    )
    if timers is not None:
        timers("recv_input_tensor", log_level=2).stop()

    return input_tensor, reqs


@nvtx.annotate("send_output_tensor", color="blue")
def send_output_tensor(
    output_tensor: torch.Tensor,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Optional[List[Work]]:

    if timers is not None:
        timers("send_output_tensor", log_level=2).start()

    send_rank = None
    if (
        mpu.get_spiral_pipeline_parallel_forward_virtual_rank()
        == mpu.get_spiral_pipeline_parallel_forward_virtual_size() - 1
    ):
        send_rank = mpu.get_pipeline_model_parallel_next_rank()
    else:
        local_world_size = mpu.get_spiral_pipeline_parallel_intra_size()
        local_rank = mpu.get_spiral_pipeline_parallel_intra_rank()
        send_rank = (
            mpu.get_pipeline_model_parallel_rank()
            // local_world_size
            * local_world_size
            + (local_rank + 1) % local_world_size
        )

    # TODO (mcrl) delete
    spiral_print(f"send output_tensor({output_tensor.shape}) to rank {send_rank}")

    _, reqs = _communicate(
        tensor_sends=[output_tensor],
        send_ranks=[send_rank],
        recv_ranks=None,
        tensor_shape=None,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=None,
    )
    if timers is not None:
        timers("send_output_tensor", log_level=2).stop()

    return reqs


# @nvtx.annotate("send_output_tensor_recv_input_tensor", color="red")
# def send_output_tensor_recv_input_tensor(output_tensor: torch.Tensor,
#                            tensor_shape: Shape,
#                            dtype: torch.dtype,
#                            batch_p2p_comm: bool = True,
#                            timers: Callable = None,) -> torch.Tensor:

#     if mpu.is_pipeline_first_stage():
#         input_tensor = None
#     else:
#         if timers is not None:
#             timers("send_recv_input_tensor", log_level=2).start()

#         recv_rank = None
#         if mpu.get_spiral_pipeline_parallel_forward_virtual_rank() == 0:
#             recv_rank = mpu.get_pipeline_model_parallel_prev_rank()
#         else:
#             local_world_size = mpu.get_spiral_pipeline_parallel_intra_size()
#             local_rank = mpu.get_spiral_pipeline_parallel_intra_rank()
#             recv_rank = mpu.get_pipeline_model_parallel_rank() // local_world_size * local_world_size \
#                 + (local_rank - 1) % local_world_size

#         send_rank = None
#         if mpu.get_spiral_pipeline_parallel_forward_virtual_rank() == mpu.get_spiral_pipeline_parallel_forward_virtual_size() - 1:
#             send_rank = mpu.get_pipeline_model_parallel_next_rank()
#         else:
#             local_world_size = mpu.get_spiral_pipeline_parallel_intra_size()
#             local_rank = mpu.get_spiral_pipeline_parallel_intra_rank()
#             send_rank = mpu.get_pipeline_model_parallel_rank() // local_world_size * local_world_size \
#                 + (local_rank + 1) % local_world_size

#         [input_tensor], _ = _communicate(tensor_sends=[output_tensor],
#                                         send_ranks=[send_rank],
#                                         recv_ranks=[recv_rank],
#                                         tensor_shape=tensor_shape,
#                                         group=mpu.get_pipeline_model_parallel_group(),
#                                         batch_p2p_comm=batch_p2p_comm,
#                                         dtype=dtype)
#         if timers is not None:
#             timers("send_output_tensor_recv_input_tensor", log_level=2).stop()

#     return input_tensor


@nvtx.annotate("recv_output_tensor_grad", color="purple")
def recv_output_tensor_grad(
    tensor_shape: Shape,
    dtype: torch.dtype,
    batch_p2p_comm: bool = True,
    overlap_p2p_comm: bool = False,
    timers: Callable = None,
) -> Tuple[torch.Tensor, Optional[List[Work]]]:

    if timers is not None:
        timers("recv_output_tensor_grad", log_level=2).start()

    recv_rank = None
    if (
        mpu.get_spiral_pipeline_parallel_backward_virtual_rank()
        == mpu.get_spiral_pipeline_parallel_backward_virtual_size() - 1
    ):
        recv_rank = mpu.get_pipeline_model_parallel_prev_rank()
    else:
        local_world_size = mpu.get_spiral_pipeline_parallel_intra_size()
        local_rank = mpu.get_spiral_pipeline_parallel_intra_rank()
        recv_rank = (
            mpu.get_pipeline_model_parallel_rank()
            // local_world_size
            * local_world_size
            + (local_rank - 1) % local_world_size
        )

    # TODO (mcrl) delete
    spiral_print(f"recv output_tensor_grad({tensor_shape}) from rank {recv_rank}")

    [output_tensor_grad], reqs = _communicate(
        tensor_sends=None,
        send_ranks=None,
        recv_ranks=[recv_rank],
        tensor_shape=tensor_shape,
        group=mpu.get_pipeline_model_parallel_group(),
        batch_p2p_comm=batch_p2p_comm,
        wait_on_reqs=not overlap_p2p_comm,
        dtype=dtype,
    )
    if timers is not None:
        timers("output_tensor_grad", log_level=2).stop()

    return output_tensor_grad, reqs


@nvtx.annotate("send_input_tensor_grad", color="green")
def send_input_tensor_grad(
    input_tensor_grad: torch.Tensor,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Optional[List[Work]]:

    if timers is not None:
        timers("send_input_tensor_grad", log_level=2).start()

    send_rank = None
    if mpu.get_spiral_pipeline_parallel_backward_virtual_rank() == 0:
        send_rank = mpu.get_pipeline_model_parallel_next_rank()
    else:
        local_world_size = mpu.get_spiral_pipeline_parallel_intra_size()
        local_rank = mpu.get_spiral_pipeline_parallel_intra_rank()
        send_rank = (
            mpu.get_pipeline_model_parallel_rank()
            // local_world_size
            * local_world_size
            + (local_rank + 1) % local_world_size
        )

    # TODO (mcrl) delete
    spiral_print(
        f"send input_tensor_grad({input_tensor_grad.shape}) to rank {send_rank}"
    )

    _, reqs = _communicate(
        tensor_sends=[input_tensor_grad],
        send_ranks=[send_rank],
        recv_ranks=None,
        tensor_shape=None,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=None,
    )
    if timers is not None:
        timers("send_input_tensor_grad", log_level=2).stop()

    return reqs


# below are deprecated codes from de2a86e

# def _communicate_by_rank_id(*,
#                             send_tensor: torch.Tensor,
#                             recv_rank_id: int,
#                             send_rank_id: int,
#                             tensor_shape: Shape,
#                             batch_p2p_comm: bool = True,
#                             dtype: Optional[torch.dtype],
#                             ) -> torch.Tensor:
#     """Batched recv from recv_rank_id and send to send_rank_id in pipeline.
#     TODO(mcrl) Additional implement needed to support batch_p2p_comm
#     """
#     ops = []
#     group = mpu.get_pipeline_model_parallel_group()
#     recv_tensor = None

#     if send_tensor is not None and send_rank_id is not None:
#         send_handle = torch.distributed.P2POp(torch.distributed.isend,
#                                               send_tensor, send_rank_id, group)
#         ops.append(send_handle)

#     if recv_rank_id is not None:
#         recv_tensor = torch.empty(tensor_shape,
#                                   requires_grad=True,
#                                   device=torch.cuda.current_device(),
#                                   dtype=dtype)

#         recv_handle = torch.distributed.P2POp(torch.distributed.irecv,
#                                               recv_tensor, recv_rank_id, group)
#         ops.append(recv_handle)

#     if len(ops) > 0:
#         reqs = torch.distributed.batch_isend_irecv(ops)
#     else:
#         reqs = []

#     for req in reqs:
#         req.wait()

#     return recv_tensor


# @nvtx.annotate("send_ckpt", color="cyan")
# def send_ckpt(send_tensor: torch.Tensor,
#               send_rank_id: int,
#               batch_p2p_comm: bool = True,
#               timers: Callable = None) -> None:
#     """Send tensor to send_rank_id in pipeline (forward send).
#     """
#     if timers is not None:
#         timers('send_ckpt', log_level=2).start()
#     _communicate_by_rank_id(send_tensor=send_tensor,
#                             recv_rank_id=None,
#                             send_rank_id=send_rank_id,
#                             tensor_shape=None,
#                             batch_p2p_comm=batch_p2p_comm,
#                             dtype=None)
#     if timers is not None:
#         timers('forward-send').stop()


# @nvtx.annotate("recv_ckpt", color="cyan")
# def recv_ckpt(recv_rank_id: int,
#               tensor_shape: Shape,
#               dtype: torch.dtype,
#               batch_p2p_comm: bool = True,
#               timers: Callable = None) -> torch.Tensor:
#     """ Receive tensor from recv_rank_id in pipeline
#     """
#     if timers is not None:
#         timers('recv_ckpt', log_level=2).start()
#     input_tensor_ckpt = _communicate_by_rank_id(send_tensor=None,
#                                                 recv_rank_id=recv_rank_id,
#                                                 send_rank_id=None,
#                                                 tensor_shape=tensor_shape,
#                                                 batch_p2p_comm=batch_p2p_comm,
#                                                 dtype=dtype)
#     if timers is not None:
#         timers('recv_ckpt').stop()
#     return input_tensor_ckpt


# @nvtx.annotate("send_ckpt_recv_ckpt", color="cyan")
# def send_ckpt_recv_ckpt(send_tensor: torch.Tensor,
#                         recv_rank_id: int,
#                         send_rank_id: int,
#                         tensor_shape: Shape,
#                         dtype: torch.dtype,
#                         batch_p2p_comm: bool = True,
#                         timers: Callable = None) -> torch.Tensor:
#     """Batched recv from recv_rank_id and send to send_rank_id in pipeline.
#     """
#     if timers is not None:
#         timers('send_ckpt_recv_ckpt', log_level=2).start()
#     input_tensor_ckpt = _communicate_by_rank_id(send_tensor=send_tensor,
#                                                 recv_rank_id=recv_rank_id,
#                                                 send_rank_id=send_rank_id,
#                                                 tensor_shape=tensor_shape,
#                                                 batch_p2p_comm=batch_p2p_comm,
#                                                 dtype=dtype)
#     if timers is not None:
#         timers('send_ckpt_recv_ckpt').stop()
#     return input_tensor_ckpt
