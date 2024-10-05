import contextlib
import warnings
import nvtx
import sys
from typing import Callable, Iterator, List, Optional, Union, Tuple
from collections import deque
from queue import Queue
import threading

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._C._distributed_c10d import Work

from megatron import get_args, get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type, get_attr_wrapped_model
from megatron.core.pipeline_parallel import forward_step, backward_step
from megatron.spiral.initialize import get_thunder_cuda_manager
from megatron.spiral.debug import spiral_print
import megatron.spiral.p2p_communication as spiral_p2p
from megatron.spiral.init_context import SpiralParamStatus, set_module_spiral_status
from megatron.spiral.utils import is_spiral_param
from megatron.spiral.generic import ContextManagers
import megatron.spiral.build_state as sbs

from .spipe_ckpt_communication import comm_ckpt
from .spipe_ckpt_schedule import CkptSendRecvType, CkptSendRecvSchedule, CkptSendRecvOp
from .spipe_communication import comm_activation, comm_activation_grad

# Types
Shape = Union[List[int], torch.Size]

# Constants
_DEBUG_SCHEDULE = True


def spipe_schedule(
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
    """Run sprial schedule, with communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    assert isinstance(
        model, list
    ), "SpiralPipe expected model chunking by stage"
    assert isinstance(
        data_iterator, list
    ), "SpiralPipe expected each model chunk to have a data iterator"

    # TODO (SpiralPipe) disable async grad reductions?
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    if mpu.get_spiral_forward_virtual_rank() is not None:
        warnings.warn(
            "SpiralPipe forward virtual rank is not None on scheule entry. There may be a bug."
        )
    if mpu.get_spiral_backward_virtual_rank() is not None:
        warnings.warn(
            "SpiralPipe backward virtual rank is not None on scheule entry. There may be a bug."
        )

    if sequence_parallel:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // mpu.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Spiral is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError(
            "Spiral is not supported with a different decoder sequence length."
        )

    offload_grad_after_bwd_stage = get_args().spiral_overlap_offload_grad
    optimize_after_bwd_stage = False
    if offload_grad_after_bwd_stage and get_args().spiral_stage_optimizer:
        assert "spiral_stage_optimizer" in kwargs
        assert "spiral_grad_scaler" in kwargs
        assert "spiral_stage_optimizer_step_returns" in kwargs and isinstance(kwargs["spiral_stage_optimizer_step_returns"], deque)
        optimize_after_bwd_stage = True
        optimizer = kwargs["spiral_stage_optimizer"]
        grad_scaler = kwargs["spiral_grad_scaler"]
    if optimize_after_bwd_stage:
        optimizer_threads = [] # (thread, queue, ev)

    def _cleanup():
        # cleanup checkpointed input tensors
        for module in model:
            empty_input_tensors: Callable = get_attr_wrapped_model(
                module, "empty_input_tensors"
            )
            empty_input_tensors()

    # Init input ckpt send recv schedule
    ckpt_send_recv_schedule = None  # placeholder
    if not forward_only:
        ckpt_send_recv_schedule = CkptSendRecvSchedule(num_microbatches)

    # Data structures for training
    forward_data_store = []
    recvs: List[Tuple[torch.Tensor, List[Work]]] = []
    ckpt_recvs: List[List[Tuple[torch.Tensor, Work]]] = [[] for _ in range(mpu.get_spiral_backward_virtual_size())]

    # Event dictionaries. Key is the event tag, thus an event necessarily requires it.
    prefetch_event_queries = {}
    compute_event_queries = {}
    offload_event_queries = {}
    free_event_queries = {}

    """ Start training """

    # nop pre-pipeline non-compute timesteps
    __num_pre_pipeline_non_compute_ts = mpu.get_pipeline_model_parallel_rank()
    for _ in range(__num_pre_pipeline_non_compute_ts):
        if not forward_only:
            comm_ckpt(
                next(ckpt_send_recv_schedule),
                model,
                ckpt_recvs,
                tensor_shape,
                dtype,
            )

    # TODO: prefetch 1st fwd stage

    # fwd
    for fwd_stage_id in range(mpu.get_spiral_forward_virtual_size()):
        if _DEBUG_SCHEDULE:
            spiral_print(f"Start fwd stage {fwd_stage_id}")
        mpu.set_spiral_forward_virtual_rank(fwd_stage_id)

        assert (
            hasattr(model[fwd_stage_id], "spiral_forward_stage_id")
            and getattr(model[fwd_stage_id], "spiral_forward_stage_id") == fwd_stage_id
        ), "Forward stage ID mismatch between virtual rank and model."

        # TODO: prefetch next stage

        # TODO: compute stream wait for prefetch current stage

        # fwd microbatches
        for m_i in range(num_microbatches):
            if _DEBUG_SCHEDULE:
                spiral_print(f" microbatch {m_i}")
            torch.cuda.nvtx.range_push(f"f[{fwd_stage_id}]m[{m_i}]")

            # set input tensor
            if mpu.is_pipeline_first_stage():
                input_tensor = None
                ### TEMP CODE FOR JUST CHECK
                input_tensor = torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
                ###
            elif fwd_stage_id == 0 and m_i == 0:
                spiral_print("recv_prev")
                input_tensor, reqs = spiral_p2p.recv_prev(
                    tensor_shape,
                    dtype,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait()
            else:
                input_tensor, reqs = recvs.pop(0)
                for req in reqs:
                    req.wait() # wait for recv complete

            spiral_print(f"tin={torch.mean(input_tensor)}")

            # TODO: Get output tensor
            output_tensor = None # TODO

            ### TEMP CODE FOR JUST CHECK
            output_tensor = torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
            spiral_print(f"tout={torch.mean(output_tensor)}")
            ###

            # sdrv ckpt
            if not forward_only:
                comm_ckpt(next(ckpt_send_recv_schedule),
                    model,
                    ckpt_recvs,
                    tensor_shape,
                    dtype,
                )

            # sdrv activation
            comm_activation(
                output_tensor,
                recvs,
                fwd_stage_id,
                m_i,
                num_microbatches,
                tensor_shape,
                dtype,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
            )

            torch.cuda.nvtx.range_pop()
        # end fwd microbatches

        mpu.set_spiral_forward_virtual_rank(None)

    if forward_only:
        _cleanup()
        return forward_data_store
    # end fwd

    # bwd
    assert not forward_only, "Forward only mode should have returned already"
    for bwd_stage_id in range(mpu.get_spiral_backward_virtual_size() - 1, -1, -1):
        if _DEBUG_SCHEDULE:
            spiral_print(f"Start bwd stage {bwd_stage_id}")
        mpu.set_spiral_backward_virtual_rank(bwd_stage_id)

        assert (
            hasattr(model[-bwd_stage_id - 1], "spiral_backward_stage_id")
            and getattr(model[-bwd_stage_id - 1], "spiral_backward_stage_id")
            == bwd_stage_id
        ), "Backward stage ID mismatch between virtual rank and model."

        # TODO: prefetch next stage

        # TODO: compute stream wait for prefetch current stage
        # NOTE: prefetch for last bwd stage is called in the fwd loop

        # bwd microbatches
        for m_i in range(num_microbatches):
            if _DEBUG_SCHEDULE:
                spiral_print(f" microbatch {m_i}")
            torch.cuda.nvtx.range_push(f"b[{bwd_stage_id}]m[{m_i}]")

            # set input tensor ckpt
            # NOTE: Must be done in compute stream to avoid error
            assert len(ckpt_recvs[bwd_stage_id]) > 0, "Missing input tensor ckpt"
            input_tensor_ckpt, ckpt_reqs = ckpt_recvs[bwd_stage_id].pop(0)
            for ckpt_req in ckpt_reqs:
                ckpt_req.wait()

            # TODO: Recomputation

            # TODO: Backward

            # set output tensor grad
            output_tensor_grad, reqs = recvs.pop(0)
            for req in reqs:
                req.wait()
            ###
            spiral_print(f"tin={torch.mean(output_tensor_grad)}")
            ###
            # TODO: Get input tensor grad
            ### TEMP CODE FOR JUST CHECK
            input_tensor_grad = torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
            spiral_print(f"tout={torch.mean(input_tensor_grad)}")
            ###
            # input_tensor_grad = None

            # sdrv ckpt
            comm_ckpt(next(ckpt_send_recv_schedule),
                model,
                ckpt_recvs,
                tensor_shape,
                dtype,
            )

            # sdrv activation grad
            comm_activation_grad(
                input_tensor_grad,
                recvs,
                bwd_stage_id,
                m_i,
                num_microbatches,
                tensor_shape,
                dtype,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
            )

            torch.cuda.nvtx.range_pop()
        # end bwd microbatches

        mpu.set_spiral_backward_virtual_rank(None)
    # end bwd

    ### TEMP CODE FOR JUST CHECK
    spiral_print("SpiralPipe schedule finished")
    exit(0)
    ###

    _cleanup()
    return forward_data_store
