import contextlib
import warnings
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
from megatron.spiral.generic import ContextManagers


# Types
Shape = Union[List[int], torch.Size]

# Constants
_DEBUG_SCHEDULE = True


def mobius_schedule(
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
    """Run SpiralPipe schedule w/o remapping, with communication between pipeline stages as needed.
    Resembles Mobius pipeline schedule. Implements both (non)re-computation versions.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # TODO (SpiralPipe) Recomputation implementation currently suffers redundant saving of input/output tensors.
    recompute = get_args().spiral_recompute_activations

    assert isinstance(model, list), "SpiralPipe expected model chunking by stage"
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
        # cleanup checkpointed input tensors and output tensors
        for module in model:
            empty_input_tensors: Callable = get_attr_wrapped_model(
                module, "empty_input_tensors"
            )
            empty_output_tensors: Callable = get_attr_wrapped_model(
                module, "empty_output_tensors"
            )
            empty_input_tensors()
            empty_output_tensors()

    # Data structures for training
    forward_data_store = []
    recvs: List[Tuple[torch.Tensor, List[Work]]] = []
    output_tensors = []

    # NOTE (SpiralPipe) forward_step() in megatron/core/pipeline_parallel/schedules.py has some additional logic to compute loss using output tensor of the last pipeline stage, which is not captured by spiral output tensors. This is a temporary workaround to capture the loss tensor, and a better solution may exist.
    if (
        not forward_only
        and not recompute
        and mpu.get_pipeline_model_parallel_rank()
        == mpu.get_pipeline_model_parallel_world_size() - 1
    ):
        losses = []

    # NOTE (SpiralPipe) microbatch data is yielded by the data iterator into _recompute_data_iterator, in order to guarantee the same data for re-computation. This issue is present only when (1) a stage is used both as fwd and bwd stage, and (2) re-computation is performed. Using a data iterator for forward_step() in original fwd and re-compute fwd incurs some microbatches only being fed to re-compute fwd.
    if not forward_only and recompute:
        _recompute_data_list = [[] for _ in range(len(data_iterator))]

    # Event dictionaries. Key is the event tag, thus an event necessarily requires it.
    prefetch_event_queries = {}
    compute_event_queries = {}
    offload_event_queries = {}
    free_event_queries = {}

    # Placeholders
    recv_handles = None

    """ Start training """
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
                for i, req in enumerate(reqs):
                    spiral_print(f"wait for req[{i}/{len(reqs)}]")
                    req.wait() # wait for recv complete
                    spiral_print("done")

            spiral_print(f"tin={torch.mean(input_tensor)}")

            # TODO: Get output tensor
            output_tensor = None # TODO

            ### TEMP CODE FOR JUST CHECK
            output_tensor = torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
            spiral_print(f"tout={torch.mean(output_tensor)}")
            ###

            # save output tensor
            if mpu.is_pipeline_last_stage():
                output_tensors.append(output_tensor)

            # send output tensor
            if (fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1 and not mpu.is_pipeline_last_stage()) and m_i == num_microbatches - 1:
                ## non-pipeline-last-stage 이지만 마지막 fwd stage의 경우 마지막 micro-batch 때 send output tensor (연두)
                # sd tensors
                spiral_print("send_next")
                reqs = spiral_p2p.send_next(
                    output_tensor,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait() # wait for send enqueue
            elif mpu.is_pipeline_last_stage() and m_i == num_microbatches - 1:
                ## pipeline last stage의 경우 마지막 micro-batch는 sdrv 없음 (진초록)
                spiral_print("pass")
                pass
            elif mpu.is_pipeline_first_stage() and m_i < mpu.get_pipeline_model_parallel_world_size() - 1:
                ## first pipeline stage의 경우, microbatch ppsize - 1 개만큼은 recv 없음 (보라))
                # sd tensor, skip recv
                spiral_print("send_next")
                reqs = spiral_p2p.send_next(
                    output_tensor,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait() # wait for send enqueue
            elif (fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1 and not mpu.is_pipeline_last_stage()) and m_i >= mpu.get_pipeline_model_parallel_world_size() - 1:
                ## non-pipeline-last-stage 이지만 마지막 fwd stage의 경우 micro-batch idx가 ppsize-1보다 크거나 같을 경우 recv 없음 (핑크) -- 물론 연두는 제외됨
                # sd tensor, skip recv
                spiral_print("send_next")
                reqs = spiral_p2p.send_next(
                    output_tensor,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait() # wait for send enqueue
            elif mpu.is_pipeline_last_stage() and m_i < num_microbatches - 1:
                ## last pipeline stage의 경우, 무조건 send 없음 (m_i=num_microbatches-1을 제외한건 진녹에서 이미 처리하고 있기 때문) (빨강)
                # rv tensor, skip send
                spiral_print("recv_prev")
                recv_tensor, reqs = spiral_p2p.recv_prev(
                    tensor_shape,
                    dtype,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                recvs.append((recv_tensor, reqs))
            else:
                ## 나머진 무조건 sr
                # sdrv tensors
                spiral_print("send_next_recv_prev")
                recv_tensor, reqs = spiral_p2p.send_next_recv_prev(
                    output_tensor,
                    tensor_shape,
                    dtype,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                recvs.append((recv_tensor, reqs))

            torch.cuda.nvtx.range_pop()
        # end fwd microbatches

        mpu.set_spiral_forward_virtual_rank(None)

    if forward_only:
        _cleanup()
        return forward_data_store
    # end fwd

    # bwd
    assert not forward_only, "Forward only mode should have returned already"
    for bwd_stage_id in range(
        mpu.get_spiral_backward_virtual_size() - 1, -1, -1
    ):
        if _DEBUG_SCHEDULE:
            spiral_print(f"Start bwd stage {bwd_stage_id}")
        mpu.set_spiral_backward_virtual_rank(bwd_stage_id)

        assert (
            hasattr(model[bwd_stage_id], "spiral_backward_stage_id")
            and getattr(model[bwd_stage_id], "spiral_backward_stage_id") == bwd_stage_id
        ), "Backward stage ID mismatch between virtual rank and model."

        # bwd microbatches
        for m_i in range(num_microbatches):
            if _DEBUG_SCHEDULE:
                spiral_print(f" microbatch {m_i}")
            torch.cuda.nvtx.range_push(f"b[{bwd_stage_id}]m[{m_i}]")

            # set output tensor grad
            if mpu.is_pipeline_last_stage():
                output_tensor_grad = None
                ### TEMP CODE FOR JUST CHECK
                output_tensor_grad = output_tensors.pop(0)
                ###
            elif bwd_stage_id == mpu.get_spiral_backward_virtual_size() - 1 and m_i == 0:
                spiral_print("recv_next")
                output_tensor_grad, reqs = spiral_p2p.recv_next(
                    tensor_shape,
                    dtype,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait()
            else:
                output_tensor_grad, reqs = recvs.pop(0)
                for i, req in enumerate(reqs):
                    spiral_print(f"wait for req[{i}]")
                    req.wait() # wait for recv complete
                    spiral_print("done")

            ###
            spiral_print(f"tin={torch.mean(output_tensor_grad)}")
            ###

            # TODO: Get input tensor grad
            ### TEMP CODE FOR JUST CHECK
            input_tensor_grad = torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
            spiral_print(f"tout={torch.mean(input_tensor_grad)}")
            ###

            # send input tensor grad
            if mpu.is_pipeline_first_stage() and m_i == num_microbatches - 1:
                ## pipeline first stage의 경우 마지막 micro-batch는 sdrv 없음 (진초록)
                spiral_print("pass")
                pass
            elif mpu.is_pipeline_first_stage() and m_i < num_microbatches - 1:
                ## first pipeline stage의 경우, 무조건 send 없음 (m_i=num_microbatches-1을 제외한건 진녹에서 이미 처리하고 있기 때문) (빨강)
                spiral_print("recv_next")
                # rv tensor, skip send
                recv_tensor, reqs = spiral_p2p.recv_next(
                    tensor_shape,
                    dtype,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                recvs.append((recv_tensor, reqs))
            elif mpu.is_pipeline_last_stage() and m_i < mpu.get_pipeline_model_parallel_world_size() - 1:
                ## last pipeline stage의 경우, microbatch ppsize - 1 개만큼은 recv 없음 (보라))
                spiral_print("send_prev")
                # sd tensor, skip recv
                reqs = spiral_p2p.send_prev(
                    input_tensor_grad,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait()
            elif (bwd_stage_id == 0 and not mpu.is_pipeline_first_stage()) and m_i == num_microbatches - 1:
                ## non-pipeline-first-stage 이지만 첫번째 bwd stage의 경우 마지막 micro batch는 무조건 send임 (연두)
                spiral_print("send_prev")
                # sd tensor, skip recv
                reqs = spiral_p2p.send_prev(
                    input_tensor_grad,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait()
            elif (bwd_stage_id == 0 and not mpu.is_pipeline_first_stage()) and m_i >= mpu.get_pipeline_model_parallel_world_size() - 1:
                ## non-pipeline-first-stage 이지만 첫번째 bwd stage의 경우 micro-batch idx가 ppsize-1보다 크거나 같을 경우 recv 없음 (핑크) -- 물론 연두는 제외됨
                # sd tensor, skip recv
                spiral_print("send_prev")
                reqs = spiral_p2p.send_prev(
                    input_tensor_grad,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                for req in reqs:
                    req.wait()
            else:
                ## 나머진 무조건 sr
                spiral_print("send_prev_recv_next")
                # sdrv tensors
                recv_tensor, reqs = spiral_p2p.send_prev_recv_next(
                    input_tensor_grad,
                    tensor_shape,
                    dtype,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
                recvs.append((recv_tensor, reqs))

            torch.cuda.nvtx.range_pop()
        # end bwd microbatches

        mpu.set_spiral_backward_virtual_rank(None)
    # end bwd

    ### TEMP CODE FOR JUST CHECK
    spiral_print(f"SpiralPipe schedule finished {len(recvs)=}")
    assert len(recvs) == 0, "recv queue is not empty"
    torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())
    spiral_print("All recv completed. Exit program")
    exit(0)
    ###

    _cleanup()
    return forward_data_store