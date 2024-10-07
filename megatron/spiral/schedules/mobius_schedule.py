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
from megatron.spiral.init_context import SpiralParamStatus, set_module_spiral_status
from megatron.spiral.generic import ContextManagers

from .mobius_communication import (
    comm_activation,
    comm_activation_grad,
    fwd_init_recvs,
    bwd_init_recvs,
)


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
    recvs: List[Tuple[torch.Tensor, List[Work]]] = [] # for input actv/output actv grad recv from other ranks

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
    recv_reqs: List[Work] = []

    """ Start training """

    # prefetch 1st fwd stage
    with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
        model[0].spiral_fetch(non_blocking=True)
        prefetch_f0 = get_thunder_cuda_manager().Event(
            "prefetch",
            "compute",
            tag="prefetch:f0",
            post_wait_fn=lambda: set_module_spiral_status(
                model[0], SpiralParamStatus.GPU
            ),
        )
        if get_thunder_cuda_manager().record_event(prefetch_f0) == -1:
            raise RuntimeError("record_event failed")
        prefetch_event_queries[prefetch_f0.tag] = prefetch_f0

    # fwd
    for fwd_stage_id in range(mpu.get_spiral_forward_virtual_size()):
        if _DEBUG_SCHEDULE:
            spiral_print(f"Start fwd stage {fwd_stage_id}")
        mpu.set_spiral_forward_virtual_rank(fwd_stage_id)

        assert (
            hasattr(model[fwd_stage_id], "spiral_forward_stage_id")
            and getattr(model[fwd_stage_id], "spiral_forward_stage_id") == fwd_stage_id
        ), "Forward stage ID mismatch between virtual rank and model."

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            if not fwd_stage_id == 0:
                if (
                    get_thunder_cuda_manager().wait_event(free_event_queries.pop(f"free:f{fwd_stage_id - 1}"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

            # skip prefetch if processing last fwd stage, as it will be reused as last bwd stage
            if not fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1:
                model[fwd_stage_id + 1].spiral_fetch(non_blocking=True)
                tag = "prefetch:" + f"f{fwd_stage_id + 1}"
                prefetch_next = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag=tag,
                    post_wait_fn=lambda fwd_stage_id=fwd_stage_id: set_module_spiral_status(
                        model[fwd_stage_id + 1], SpiralParamStatus.GPU
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_next) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_event_queries[prefetch_next.tag] = prefetch_next

        # compute stream wait for prefetch current stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            if (
                get_thunder_cuda_manager().wait_event(prefetch_event_queries.pop(f"prefetch:f{fwd_stage_id}"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

        # fwd microbatches
        for m_i in range(num_microbatches):
            if _DEBUG_SCHEDULE:
                spiral_print(f" microbatch {m_i}")
            torch.cuda.nvtx.range_push(f"f[{fwd_stage_id}]m[{m_i}]")

            # set input tensor
            fwd_init_recvs(
                recvs,
                fwd_stage_id,
                m_i,
                num_microbatches,
                dtype,
                    tensor_shape,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
            input_tensor, recv_reqs = recvs.pop(0)

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # NOTE (SpiralPipe) Creating a new iterator with _data results in deep copy of _data. So, calling detach_variable() as in backward() of CheckpointFunction (random.py) is not necessary. However, we must note that a redundant data storage is created here, while a detached tensor shares the underlying storage with the original. https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                # TODO (SpiralPipe) Seek better solution to use detached tensor, using detach_variable().
                _data = next(data_iterator[fwd_stage_id])
                _data_iterator = iter([_data]) # wrap
                if not forward_only and recompute:
                    _recompute_data_list[fwd_stage_id].append(_data)

                # wait for recv input tensor
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                if recv_reqs is not None:
                    for req in recv_reqs:
                        req.wait()

                _ctx = []
                if forward_only or recompute:
                    _ctx.append(torch.no_grad())
                with ContextManagers(_ctx):
                    output_tensor = forward_step(
                        forward_step_func,
                        _data_iterator,
                        model[fwd_stage_id],
                        num_microbatches,
                        input_tensor,
                        forward_data_store,
                        timers,
                        collect_non_loss_data,
                        dtype,
                        enable_autocast,
                    )

                if (
                    not forward_only
                    and not recompute
                    and mpu.is_pipeline_last_stage()
                ):
                    # save loss for pipeline last stage bwd
                    assert isinstance(output_tensor, torch.Tensor) and output_tensor.numel() == 1
                    losses.append(output_tensor)

                compute_microbatch = get_thunder_cuda_manager().Event(
                    "compute",
                    None,
                    tag=f"compute:f{fwd_stage_id}m{m_i}",
                )
                if get_thunder_cuda_manager().record_event(compute_microbatch) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatch.tag] = compute_microbatch

                if m_i == num_microbatches - 1:
                    # notify free stream to act
                    compute_microbatches_end = get_thunder_cuda_manager().Event(
                        "compute", "free", tag=f"compute:f{fwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end
            # end compute stream

            # sd/rv output/input tensor
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:f{fwd_stage_id}m{m_i}"))
                == -1
            ):
                raise RuntimeError("wait_event failed")
            # sdrv activation
            comm_activation(
                output_tensor,
                recvs,
                fwd_stage_id,
                m_i,
                num_microbatches,
                dtype,
                tensor_shape,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
            )

            torch.cuda.nvtx.range_pop()
        # end fwd microbatches

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("free")):
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:f{fwd_stage_id}end"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

            # skip free if processing last fwd stage, as it will be reused as last bwd stage
            if not fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1:
                model[fwd_stage_id].spiral_free()
                free_curr = get_thunder_cuda_manager().Event(
                    "free",
                    "prefetch",
                    tag=f"free:f{fwd_stage_id}",
                )
                if get_thunder_cuda_manager().record_event(free_curr) == -1:
                    raise RuntimeError("record_event failed")
                free_event_queries[free_curr.tag] = free_curr
        # end free fwd stage

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

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            # skip free wait if processing last bwd stage, as last fwd stage doesn't free
            if not bwd_stage_id == mpu.get_spiral_backward_virtual_size() - 1:
                if (
                    get_thunder_cuda_manager().wait_event(free_event_queries.pop(f"free:b{bwd_stage_id + 1}"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

            if not bwd_stage_id == 0:
                model[bwd_stage_id - 1].spiral_fetch(non_blocking=True)
                prefetch_next = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag="prefetch:" + f"b{bwd_stage_id - 1}",
                    post_wait_fn=lambda bwd_stage_id=bwd_stage_id: set_module_spiral_status(
                        model[bwd_stage_id - 1], SpiralParamStatus.GPU
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_next) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_event_queries[prefetch_next.tag] = prefetch_next

        # compute stream wait for prefetch current stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            # skip prefetch wait if processing last bwd stage, as last bwd stage reuse last fwd stage
            if not bwd_stage_id == mpu.get_spiral_backward_virtual_size() - 1:
                if (
                    get_thunder_cuda_manager().wait_event(prefetch_event_queries.pop(f"prefetch:b{bwd_stage_id}"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

        if recompute:
            _recompute_data_iterator = iter(_recompute_data_list[bwd_stage_id])

        # bwd microbatches
        for m_i in range(num_microbatches):
            if _DEBUG_SCHEDULE:
                spiral_print(f" microbatch {m_i}")
            torch.cuda.nvtx.range_push(f"b[{bwd_stage_id}]m[{m_i}]")

            # set output tensor grad
            bwd_init_recvs(
                recvs,
                bwd_stage_id,
                m_i,
                num_microbatches,
                dtype,
                    tensor_shape,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )
            output_tensor_grad, recv_reqs = recvs.pop(0)

            # set input tensor ckpt
            input_tensor_ckpt = (
                model[bwd_stage_id].module[0].spiral_input_tensors.popleft()
            )

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                if recompute:
                    with torch.enable_grad():
                        output_tensor = forward_step(
                            forward_step_func,
                            _recompute_data_iterator,
                            model[bwd_stage_id],
                            num_microbatches,
                            input_tensor_ckpt,
                            [],
                            timers,
                            collect_non_loss_data,
                            dtype,
                            enable_autocast,
                        )
                else:
                    output_tensor = (
                        model[bwd_stage_id].module[-1].spiral_output_tensors.popleft()
                    )
                    # NOTE (SpiralPipe) Although we can make conditional code with above code line that handles output tensor, we do it this way to pop the output tensor from the last pipeline stage, just to be sure that those tensors should not be accesssed from somewhere else.
                    if mpu.is_pipeline_last_stage():
                        output_tensor = losses.pop(0)
                        assert isinstance(output_tensor, torch.Tensor) and output_tensor.numel() == 1
                    assert output_tensor.requires_grad

                if optimize_after_bwd_stage:
                    # grad_scaler is aligned (ascending) w.r.t bwd_stage_id
                    # (same as optimizer_list in SpiralStageOptimizer)
                    _grad_scaler = grad_scaler[bwd_stage_id]
                else:
                    _grad_scaler = grad_scaler

                # wait for recv output tensor grad
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                if recv_reqs is not None:
                    for req in recv_reqs:
                        req.wait()

                input_tensor_grad = backward_step(
                    _grad_scaler,
                    input_tensor_ckpt,
                    output_tensor,
                    output_tensor_grad,
                    model_type,
                    timers,
                    deallocate_pipeline_outputs,
                )

                compute_microbatch = get_thunder_cuda_manager().Event(
                    "compute",
                    None,
                    tag=f"compute:b{bwd_stage_id}m{m_i}",
                )
                if get_thunder_cuda_manager().record_event(compute_microbatch) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatch.tag] = compute_microbatch

                if m_i == num_microbatches - 1:
                    # notify free stream to act
                    compute_microbatches_end = get_thunder_cuda_manager().Event(
                        "compute", None, tag=f"compute:b{bwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end
            # end compute stream

            # sd/rv input/output tensor grad
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:b{bwd_stage_id}m{m_i}"))
                == -1
            ):
                raise RuntimeError("wait_event failed")
            # sdrv activation grad
            comm_activation_grad(
                input_tensor_grad,
                recvs,
                bwd_stage_id,
                m_i,
                num_microbatches,
                dtype,
                tensor_shape,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
            )

            torch.cuda.nvtx.range_pop()
        # end bwd microbatches

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("free")):
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.get(f"compute:b{bwd_stage_id}end"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

            # free bwd stage
            model[bwd_stage_id].spiral_free()
            free_curr = get_thunder_cuda_manager().Event(
                "free",
                None if bwd_stage_id == 0 else "prefetch",
                tag=f"free:b{bwd_stage_id}",
            )
            if get_thunder_cuda_manager().record_event(free_curr) == -1:
                raise RuntimeError("record_event failed")
            free_event_queries[free_curr.tag] = free_curr
        # end free bwd stage

        # if grad offload is not overlapped, then it should be reduced and offloaded after `forward_backward_func` at train_step()
        if offload_grad_after_bwd_stage:
            with torch.cuda.stream(get_thunder_cuda_manager().Stream("offload")):
                if (
                    get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:b{bwd_stage_id}end"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")
                # if not spiral stage optimizer, then gradient should be manually reduced
                if optimize_after_bwd_stage:
                    optimizer[bwd_stage_id].reduce_model_grads(get_args(), get_timers())
                else:
                    model[bwd_stage_id].allreduce_gradients()
                # offload grads
                model[bwd_stage_id].spiral_offload_grad(non_blocking=True)
                offload_grad_curr = get_thunder_cuda_manager().Event(
                    "offload",
                    "free",
                    tag=f"offload_grad:b{bwd_stage_id}"
                )
                offload_grad_ev_long = get_thunder_cuda_manager().record_event(offload_grad_curr)
                if offload_grad_ev_long == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries[offload_grad_curr.tag] = offload_grad_curr

                # if not spiral stage optimizer, then optimizer will be executed after iteration
                if optimize_after_bwd_stage:
                    _offload_grad_ev_cpu = torch.cuda.Event() # event to synchronize at cpu
                    _optimizer_thread_queue = Queue()
                    _offload_grad_ev_cpu.record()

                    inner_step_kwargs = {}
                    inner_step_kwargs["spiral_offload_grad_ev"] = _offload_grad_ev_cpu
                    inner_step_kwargs["spiral_optimizer_thread_queue"] = _optimizer_thread_queue
                    inner_step_kwargs["spiral_offload_grad_ev_long"] = _offload_grad_ev_cpu.cuda_event

                    # TODO (SpiralPipe) timers is None. Fix it
                    op = threading.Thread(
                        target=optimizer[bwd_stage_id].step,
                        args=(get_args(), get_timers()),
                        kwargs=inner_step_kwargs,
                    )
                    op.start()
                    optimizer_threads.append((op, _optimizer_thread_queue))

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("free")):
                if (
                    get_thunder_cuda_manager().wait_event(offload_event_queries.pop(f"offload_grad:b{bwd_stage_id}"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

                # free bwd stage grads
                model[bwd_stage_id].spiral_free_grad()
                free_grad_curr = get_thunder_cuda_manager().Event(
                    "free",
                    None,
                    tag=f"free_grad:b{bwd_stage_id}",
                )
                if get_thunder_cuda_manager().record_event(free_grad_curr) == -1:
                    raise RuntimeError("record_event failed")
                free_event_queries[free_grad_curr.tag] = free_grad_curr
        # end offload & free grad

        mpu.set_spiral_backward_virtual_rank(None)
    # end bwd

    # cleanup schedule events
    if (
        get_thunder_cuda_manager().wait_event(free_event_queries.pop(f"free:b0"))
        == -1
    ):
        raise RuntimeError("wait_event failed")

    if offload_grad_after_bwd_stage:
        for bwd_stage_id in range(mpu.get_spiral_backward_virtual_size() - 1, -1, -1):
            # flush free grad event queries
            if (
                get_thunder_cuda_manager().wait_event(
                    free_event_queries.pop(f"free_grad:b{bwd_stage_id}")
                )
                == -1
            ):
                raise RuntimeError("wait_event failed")

    # join optimizer
    if optimize_after_bwd_stage:
        for op, q in optimizer_threads:
            op.join()
            kwargs["spiral_stage_optimizer_step_returns"].appendleft(q.get())

    _cleanup()
    return forward_data_store
