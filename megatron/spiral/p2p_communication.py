from typing import Optional, List, Union, Callable, Tuple
from collections import deque

import nvtx
import torch
from torch._C._distributed_c10d import Work

from megatron.core import mpu

# Types
Shape = Union[List[int], torch.Size]

# Constants
MINIMIZE_INTERNODE_COMM = False # TODO (SpiralPipe) further optimizer internode comm.
_DEBUG_COMM = False

if _DEBUG_COMM:
    from megatron.spiral.debug import spiral_print


# Tensor stores to use for self send-recv
# Only used when pipeline world size is 1
# TODO (SpiralPipe) This is temporary solution to avoid breaking the code
class NOP_Wait:
    @staticmethod
    def wait():
        pass

_self_send_recv_input_tensor_store = deque()
_self_send_recv_output_tensor_grad_store = deque()

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
    # batch_p2p_sync = True

    # set group
    if group is None:
        group = mpu.get_pipeline_model_parallel_group()

    if not variable_seq_lengths:
        shape = tensor_shape
    else:
        # TODO (SpiralPipe) implement _communicate_shape
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
        # TODO (SpiralPipe) support ring exchange
        raise NotImplementedError("SpiralPipe does not support ring exchange yet")
        # def _ring_exchange_wrapper(**kwargs):
        #     torch.distributed.ring_exchange(**kwargs)
        #     return []
        # p2p_func = _ring_exchange_wrapper
    elif batch_p2p_comm:
        # assert wait_on_reqs
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

    # if batch_p2p_comm and batch_p2p_sync:
    #     # To protect against race condition when using batch_isend_irecv().
    #     # User should assert that we have a modern enough PyTorch to not need this
    #     torch.cuda.synchronize()

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
    if MINIMIZE_INTERNODE_COMM:
        if mpu.get_spiral_forward_virtual_rank() == 0:
            recv_rank = mpu.get_pipeline_model_parallel_prev_rank()
        else:
            local_world_size = mpu.get_spiral_intra_size()
            local_rank = mpu.get_spiral_intra_rank()
            recv_rank = (
                mpu.get_pipeline_model_parallel_rank()
                // local_world_size
                * local_world_size
                + (local_rank - 1) % local_world_size
            )
    else:
        recv_rank = mpu.get_pipeline_model_parallel_prev_rank()

    if recv_rank != mpu.get_pipeline_model_parallel_rank():
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
    else:
        # If we are receiving from ourselves, just pop the tensor from the store
        input_tensor = _self_send_recv_input_tensor_store.popleft()
        reqs = [NOP_Wait]

    if timers is not None:
        timers("recv_input_tensor").stop()

    if _DEBUG_COMM:
        if overlap_p2p_comm:
            spiral_print(f"recv input_tensor(src={recv_rank})")
        else:
            spiral_print(f"recv input_tensor(src={recv_rank}, mean={torch.mean(input_tensor)})")

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
    if MINIMIZE_INTERNODE_COMM:
        if (
            mpu.get_spiral_forward_virtual_rank()
            == mpu.get_spiral_forward_virtual_size() - 1
        ):
            send_rank = mpu.get_pipeline_model_parallel_next_rank()
        else:
            local_world_size = mpu.get_spiral_intra_size()
            local_rank = mpu.get_spiral_intra_rank()
            send_rank = (
                mpu.get_pipeline_model_parallel_rank()
                // local_world_size
                * local_world_size
                + (local_rank + 1) % local_world_size
            )
    else:
        send_rank = mpu.get_pipeline_model_parallel_next_rank()

    if send_rank != mpu.get_pipeline_model_parallel_rank():
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
    else:
        # If we are sending to ourselves, just store the tensor for later
        _self_send_recv_input_tensor_store.append(output_tensor.detach())
        reqs = [NOP_Wait]

    if timers is not None:
        timers("send_output_tensor").stop()

    if _DEBUG_COMM:
        spiral_print(f"send output_tensor(dst={send_rank}, mean={torch.mean(output_tensor)})")

    return reqs


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
    if MINIMIZE_INTERNODE_COMM:
        if (
            mpu.get_spiral_backward_virtual_rank()
            == mpu.get_spiral_backward_virtual_size() - 1
        ):
            if mpu.is_spiral_remap():
                recv_rank = mpu.get_pipeline_model_parallel_prev_rank()
            elif mpu.is_spiral():
                recv_rank = mpu.get_pipeline_model_parallel_next_rank()
            else:
                raise ValueError("Unknown pipeline parallelism type for spiral_p2p")
        else:
            local_world_size = mpu.get_spiral_intra_size()
            local_rank = mpu.get_spiral_intra_rank()
            if mpu.is_spiral_remap():
                recv_rank = (
                    mpu.get_pipeline_model_parallel_rank()
                    // local_world_size
                    * local_world_size
                    + (local_rank - 1) % local_world_size
                )
            elif mpu.is_spiral():
                recv_rank = (
                    mpu.get_pipeline_model_parallel_rank()
                    // local_world_size
                    * local_world_size
                    + (local_rank + 1) % local_world_size
                )
            else:
                raise ValueError("Unknown pipeline parallelism type for spiral_p2p")
    else:
        if mpu.is_spiral_remap():
            recv_rank = mpu.get_pipeline_model_parallel_prev_rank()
        elif mpu.is_spiral():
            recv_rank = mpu.get_pipeline_model_parallel_next_rank()
        else:
            raise ValueError("Unknown pipeline parallelism type for spiral_p2p")

    if recv_rank != mpu.get_pipeline_model_parallel_rank():
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
    else:
        # If we are receiving from ourselves, just pop the tensor from the store
        output_tensor_grad = _self_send_recv_output_tensor_grad_store.popleft()
        reqs = [NOP_Wait]

    if timers is not None:
        timers("output_tensor_grad").stop()

    if _DEBUG_COMM:
        if overlap_p2p_comm:
            spiral_print(f"recv output_tensor_grad(src={recv_rank})")
        else:
            spiral_print(f"recv output_tensor_grad(src={recv_rank}, mean={torch.mean(output_tensor_grad)})")

    return output_tensor_grad, reqs


@nvtx.annotate("send_input_tensor_grad", color="rapids")
def send_input_tensor_grad(
    input_tensor_grad: torch.Tensor,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Optional[List[Work]]:

    if timers is not None:
        timers("send_input_tensor_grad", log_level=2).start()

    send_rank = None
    if MINIMIZE_INTERNODE_COMM:
        if mpu.get_spiral_backward_virtual_rank() == 0:
            if mpu.is_spiral_remap():
                send_rank = mpu.get_pipeline_model_parallel_next_rank()
            elif mpu.is_spiral():
                send_rank = mpu.get_pipeline_model_parallel_prev_rank()
            else:
                raise ValueError("Unknown pipeline parallelism type for spiral_p2p")
        else:
            local_world_size = mpu.get_spiral_intra_size()
            local_rank = mpu.get_spiral_intra_rank()
            if mpu.is_spiral_remap():
                send_rank = (
                    mpu.get_pipeline_model_parallel_rank()
                    // local_world_size
                    * local_world_size
                    + (local_rank + 1) % local_world_size
                )
            elif mpu.is_spiral():
                send_rank = (
                    mpu.get_pipeline_model_parallel_rank()
                    // local_world_size
                    * local_world_size
                    + (local_rank - 1) % local_world_size
                )
            else:
                raise ValueError("Unknown pipeline parallelism type for spiral_p2p")
    else:
        if mpu.is_spiral_remap():
            send_rank = mpu.get_pipeline_model_parallel_next_rank()
        elif mpu.is_spiral():
            send_rank = mpu.get_pipeline_model_parallel_prev_rank()
        else:
            raise ValueError("Unknown pipeline parallelism type for spiral_p2p")

    if send_rank != mpu.get_pipeline_model_parallel_rank():
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
    else:
        # If we are sending to ourselves, just store the tensor for later
        _self_send_recv_output_tensor_grad_store.append(input_tensor_grad.detach())
        reqs = [NOP_Wait]

    if timers is not None:
        timers("send_input_tensor_grad").stop()

    if _DEBUG_COMM:
        spiral_print(f"send input_tensor_grad(dst={send_rank}, mean={torch.mean(input_tensor_grad)})")

    return reqs


@nvtx.annotate("snrp", color="cyan")
def send_next_recv_prev(
    tensor_send: torch.Tensor,
    tensor_shape: Shape,
    dtype: torch.dtype,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Tuple[torch.Tensor, Optional[List[Work]]]:
    if timers is not None:
        timers("send_next_recv_prev", log_level=2).start()
    [recv], reqs = _communicate(
        tensor_sends=[tensor_send],
        send_ranks=[mpu.get_pipeline_model_parallel_next_rank()],
        recv_ranks=[mpu.get_pipeline_model_parallel_prev_rank()],
        tensor_shape=tensor_shape,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=dtype,
    )
    if timers is not None:
        timers("send_next_recv_prev").stop()
    return recv, reqs


@nvtx.annotate("sprn", color="cyan")
def send_prev_recv_next(
    tensor_send: torch.Tensor,
    tensor_shape: Shape,
    dtype: torch.dtype,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Tuple[torch.Tensor, Optional[List[Work]]]:
    if timers is not None:
        timers("send_next_recv_prev", log_level=2).start()
    [recv], reqs = _communicate(
        tensor_sends=[tensor_send],
        send_ranks=[mpu.get_pipeline_model_parallel_prev_rank()],
        recv_ranks=[mpu.get_pipeline_model_parallel_next_rank()],
        tensor_shape=tensor_shape,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=dtype,
    )
    if timers is not None:
        timers("send_next_recv_prev").stop()
    return recv, reqs


@nvtx.annotate("sn", color="cyan")
def send_next(
    tensor_send: torch.Tensor,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Optional[Work]:
    if timers is not None:
        timers("send_next", log_level=2).start()
    _, reqs = _communicate(
        tensor_sends=[tensor_send],
        send_ranks=[mpu.get_pipeline_model_parallel_next_rank()],
        recv_ranks=None,
        tensor_shape=None,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=None,
    )
    if timers is not None:
        timers("send_next").stop()
    return reqs


@nvtx.annotate("rp", color="purple")
def recv_prev(
    tensor_shape: Shape,
    dtype: torch.dtype,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Tuple[torch.Tensor, Optional[Work]]:
    if timers is not None:
        timers("recv_prev", log_level=2).start()
    [recv], reqs = _communicate(
        tensor_sends=None,
        send_ranks=None,
        recv_ranks=[mpu.get_pipeline_model_parallel_prev_rank()],
        tensor_shape=tensor_shape,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=dtype,
    )
    if timers is not None:
        timers("recv_prev").stop()
    return recv, reqs


@nvtx.annotate("sp", color="cyan")
def send_prev(
    tensor_send: torch.Tensor,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Optional[Work]:
    if timers is not None:
        timers("send_next", log_level=2).start()
    _, reqs = _communicate(
        tensor_sends=[tensor_send],
        send_ranks=[mpu.get_pipeline_model_parallel_prev_rank()],
        recv_ranks=None,
        tensor_shape=None,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=None,
    )
    if timers is not None:
        timers("send_next").stop()
    return reqs


@nvtx.annotate("rn", color="purple")
def recv_next(
    tensor_shape: Shape,
    dtype: torch.dtype,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: Callable = None,
) -> Tuple[torch.Tensor, Optional[Work]]:
    if timers is not None:
        timers("recv_prev", log_level=2).start()
    [recv], reqs = _communicate(
        tensor_sends=None,
        send_ranks=None,
        recv_ranks=[mpu.get_pipeline_model_parallel_next_rank()],
        tensor_shape=tensor_shape,
        group=mpu.get_pipeline_model_parallel_group(),
        wait_on_reqs=not overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        dtype=dtype,
    )
    if timers is not None:
        timers("recv_prev").stop()
    return recv, reqs