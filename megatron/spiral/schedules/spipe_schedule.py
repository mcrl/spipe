import contextlib
import warnings
from typing import Callable, Iterator, List, Optional, Union, Tuple

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

from .spipe_ckpt_communication import comm_ckpt
from .spipe_ckpt_schedule import CkptSendRecvSchedule
from .spipe_communication import (
    comm_activation,
    comm_activation_grad,
    fwd_pre_pipeline_init_recvs,
    fwd_init_recvs,
    bwd_init_recvs,
)


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
    """Run spipe schedule, with communication between pipeline stages as needed.

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

    use_batch_p2p_comm = not get_args().spiral_actv_p2p

    offload_grad_after_bwd_stage = get_args().spiral_overlap_offload_grad
    optimize_after_bwd_stage = offload_grad_after_bwd_stage and get_args().spiral_stage_optimizer
    if get_args().spiral_stage_optimizer:
        assert "spiral_stage_optimizer" in kwargs
        optimizer = kwargs["spiral_stage_optimizer"]

    def _is_cpu_optimizer(bwd_stage_id):
        if get_args().spiral_stage_optimizer:
            return optimizer.is_cpu_optimizer(bwd_stage_id)
        else:
            # If the stage optimizer is not enabled, the optimizer should always be the CPU optimizer.
            return True

    def _cleanup():
        # cleanup checkpointed input tensors
        for module in model:
            empty_input_tensors: Callable = get_attr_wrapped_model(
                module, "empty_input_tensors"
            )
            empty_input_tensors()

    def _wait_reqs(reqs: List[Work]):
        if reqs is not None:
            for req in reqs:
                req.wait()

    # Init input ckpt send recv schedule
    ckpt_send_recv_schedule = None  # placeholder
    sync_ckpt_comm = get_args().spiral_sync_ckpt_communication
    if not forward_only:
        ckpt_send_recv_schedule = CkptSendRecvSchedule(num_microbatches=num_microbatches, use_sync=sync_ckpt_comm)

    # Data structures for training
    forward_data_store = []
    recvs: List[Tuple[torch.Tensor, List[Work]]] = []
    ckpt_recvs: List[List[Tuple[torch.Tensor, Work]]] = [[] for _ in range(mpu.get_spiral_backward_virtual_size())]

    # Event dictionaries. Key is the event tag, thus an event necessarily requires it.
    prefetch_event_queries = {}
    compute_event_queries = {}
    offload_event_queries = {}
    free_event_queries = {}

    # Placeholders
    recv_reqs: List[Work] = []
    ckpt_recv_reqs: List[Work] = []

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

    # pre-pipeline non-compute timesteps
    __num_pre_pipeline_non_compute_ts = mpu.get_pipeline_model_parallel_rank()
    for ppnct in range(__num_pre_pipeline_non_compute_ts):
        if ppnct == __num_pre_pipeline_non_compute_ts - 1:
            # NOTE In the last non_compute timestep, fwd_pre_pipeline_init_recvs **MUST PRECEDE** pre-pipeline
            # comm_ckpt to avoid deadlock.
            #   e.g., rank 0: a) comm_actv (isend) -> b) comm_ckpt (isend)
            #         rank 1: b) pre-pipeline comm_ckpt (irecv) -> a) comm_actv (irecv)
            # Above pattern deadlocks as a)s and b)s requires synchronization, respectively.
            # Another solution can be to move fwd_pre_pipeline_init_recvs logic into fwd_init_recvs and reorder to
            # comm_ckpt >> comm_actv/_actv_grad pattern. However, this stall activation sdrvs which are the
            # critical path for the pipeline. So, we choose to keep comm_actv/_actv_grad >> comm_ckpt pattern and
            # use current form.
            # NOTE Moving this out of current for loop, or merging with fwd_init_recvs in the fwd for loop will
            # definitely lead to deadlock.
            fwd_pre_pipeline_init_recvs(
                recvs,
                dtype,
                tensor_shape,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=use_batch_p2p_comm,
                timers=timers,
            )
        # endif
        if not forward_only:
            comm_ckpt(
                next(ckpt_send_recv_schedule),
                model,
                ckpt_recvs,
                tensor_shape,
                dtype,
            )

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

            # skip prefetch if forward only and processing last fwd stage
            if not (
                forward_only
                and fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1
            ):
                model[fwd_stage_id + 1].spiral_fetch(non_blocking=True)
                tag = "prefetch:" + (
                    f"f{fwd_stage_id + 1}"
                    if fwd_stage_id + 1 < mpu.get_spiral_forward_virtual_size()
                    else f"b{mpu.get_spiral_backward_virtual_size() - 1}"
                )
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
            fwd_init_recvs(recvs) # set input tensor of first pipeline stage
            input_tensor, recv_reqs = recvs.pop(0)

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # wait for recv input tensor
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                _wait_reqs(recv_reqs)

                # NOTE (SpiralPipe) Original FWD should not create computation graph for two reasons
                # 1. It will be recomputed in BWD
                # 2. It creates computation graph to the input ckpt tensor, which leads to computation graph duplication when BWD is performed in the same rank. Symptoms include size error due to ops with size 0 tensor.
                with torch.no_grad():
                    output_tensor = forward_step(
                        forward_step_func,
                        data_iterator[fwd_stage_id],
                        model[fwd_stage_id],
                        num_microbatches,
                        input_tensor,
                        forward_data_store,
                        timers,
                        collect_non_loss_data,
                        dtype,
                        enable_autocast,
                    )

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
                tensor_shape,
                dtype,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=use_batch_p2p_comm,
                timers=timers,
                omit_send_reqs=not use_batch_p2p_comm,
            )
            # sdrv ckpt
            if not forward_only:
                comm_ckpt(
                    next(ckpt_send_recv_schedule),
                    model,
                    ckpt_recvs,
                    tensor_shape,
                    dtype,
                )

            torch.cuda.nvtx.range_pop()
        # end fwd microbatches

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("free")):
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:f{fwd_stage_id}end"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

            # free fwd stage
            model[fwd_stage_id].spiral_free()
            if not (
                forward_only
                and fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1
            ):
                free_curr = get_thunder_cuda_manager().Event(
                    "free",
                    "prefetch",
                    tag=f"free:f{fwd_stage_id}",
                )
                if get_thunder_cuda_manager().record_event(free_curr) == -1:
                    raise RuntimeError("record_event failed")
                free_event_queries[free_curr.tag] = free_curr

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
            hasattr(model[-bwd_stage_id - 1], "spiral_backward_stage_id")
            and getattr(model[-bwd_stage_id - 1], "spiral_backward_stage_id")
            == bwd_stage_id
        ), "Backward stage ID mismatch between virtual rank and model."

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            if (
                get_thunder_cuda_manager().wait_event(
                    free_event_queries.pop(
                        f"free:f{fwd_stage_id}"
                        if bwd_stage_id == mpu.get_spiral_backward_virtual_size() - 1
                        else f"free:b{bwd_stage_id + 1}"
                    )
                )
                == -1
            ):
                raise RuntimeError("wait_event failed")

            if not bwd_stage_id == 0:
                model[-bwd_stage_id].spiral_fetch(non_blocking=True)
                prefetch_next = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag="prefetch:" + f"b{bwd_stage_id - 1}",
                    post_wait_fn=lambda bwd_stage_id=bwd_stage_id: set_module_spiral_status(
                        model[-bwd_stage_id], SpiralParamStatus.GPU
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_next) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_event_queries[prefetch_next.tag] = prefetch_next

        # compute stream wait for prefetch current stage
        # NOTE: prefetch for last bwd stage is called in the fwd loop
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            if (
                get_thunder_cuda_manager().wait_event(prefetch_event_queries.pop(f"prefetch:b{bwd_stage_id}"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

        # bwd microbatches
        for m_i in range(num_microbatches):
            if _DEBUG_SCHEDULE:
                spiral_print(f" microbatch {m_i}")
            torch.cuda.nvtx.range_push(f"b[{bwd_stage_id}]m[{m_i}]")

            # set output tensor grad
            bwd_init_recvs(recvs)
            output_tensor_grad, recv_reqs = recvs.pop(0)

            # set input tensor ckpt
            assert len(ckpt_recvs[bwd_stage_id]) > 0, "Missing input tensor ckpt"
            input_tensor_ckpt, ckpt_recv_reqs = ckpt_recvs[bwd_stage_id].pop(0)

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # wait for recv input tensor ckpt
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                _wait_reqs(ckpt_recv_reqs)

                output_tensor = forward_step(
                    forward_step_func,
                    data_iterator[
                        bwd_stage_id + mpu.get_spiral_forward_virtual_size()
                    ],
                    model[-bwd_stage_id - 1],
                    num_microbatches,
                    input_tensor_ckpt,
                    [],
                    timers,
                    collect_non_loss_data,
                    dtype,
                    enable_autocast,
                )

                # wait for recv output tensor grad
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                _wait_reqs(recv_reqs)

                input_tensor_grad = backward_step(
                    grad_scaler,
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
                tensor_shape,
                dtype,
                overlap_p2p_comm=overlap_p2p_comm,
                batch_p2p_comm=use_batch_p2p_comm,
                timers=timers,
                omit_send_reqs=not use_batch_p2p_comm,
            )
            # sdrv ckpt
            comm_ckpt(
                next(ckpt_send_recv_schedule),
                model,
                ckpt_recvs,
                tensor_shape,
                dtype,
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
            if _is_cpu_optimizer(bwd_stage_id):
                model[-bwd_stage_id - 1].spiral_free()
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
                    model[-bwd_stage_id - 1].allreduce_gradients()

                # offload grads
                if _is_cpu_optimizer(bwd_stage_id):
                    model[-bwd_stage_id - 1].spiral_offload_grad(non_blocking=True)
                offload_grad_curr = get_thunder_cuda_manager().Event(
                    "offload",
                    None,
                    tag=f"offload_grad:b{bwd_stage_id}"
                )
                if get_thunder_cuda_manager().record_event(offload_grad_curr) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries[offload_grad_curr.tag] = offload_grad_curr

                # if not spiral stage optimizer, then optimizer will be executed after iteration
                if optimize_after_bwd_stage:
                    optimizer.step(bwd_stage_id, offload_grad_curr, get_args(), get_timers())

                    # for gpu optimizer, need to offload/free parameter
                    if not optimizer.is_cpu_optimizer(bwd_stage_id):
                        model[-bwd_stage_id - 1].spiral_offload(non_blocking=True)
                        model[-bwd_stage_id - 1].spiral_free()
                        offload_param_curr = get_thunder_cuda_manager().Event(
                            "offload",
                            None,
                            tag=f"offload_param:b{bwd_stage_id}"
                        )
                        if get_thunder_cuda_manager().record_event(offload_param_curr) == -1:
                            raise RuntimeError("record_event failed")
                        offload_event_queries[offload_param_curr.tag] = offload_param_curr

                # free bwd stage grads (spiral_free_grad is cpu job with tensor.record_stream)
                model[-bwd_stage_id - 1].spiral_free_grad()
        # end offload & free grad

        mpu.set_spiral_backward_virtual_rank(None)
    # end bwd

    # post-pipeline non-compute timesteps
    if sync_ckpt_comm:
        assert not forward_only, "Forward only mode should have returned already"
        __num_post_pipeline_non_compute_ts = (
            mpu.get_pipeline_model_parallel_world_size()
            - mpu.get_pipeline_model_parallel_rank()
            - 1
        )
        for _ in range(__num_post_pipeline_non_compute_ts):
            if not forward_only:
                comm_ckpt(
                    next(ckpt_send_recv_schedule),
                    model,
                    ckpt_recvs,
                    tensor_shape,
                    dtype,
                )

    # cleanup schedule events
    if (
        get_thunder_cuda_manager().wait_event(free_event_queries.pop(f"free:b0"),
                                              sync=True)
        == -1
    ):
        raise RuntimeError("wait_event failed")

    if offload_grad_after_bwd_stage:
        for bwd_stage_id in range(mpu.get_spiral_backward_virtual_size() - 1, -1, -1):
            # flush offload grad event queries
            if (
                get_thunder_cuda_manager().wait_event(
                    offload_event_queries.pop(f"offload_grad:b{bwd_stage_id}"),
                    sync=True
                )
                == -1
            ):
                raise RuntimeError("wait_event failed")

        if optimize_after_bwd_stage:
            for bwd_stage_id in range(mpu.get_spiral_backward_virtual_size() - 1, -1, -1):
                if not optimizer.is_cpu_optimizer(bwd_stage_id):
                    # flush offload param event queries
                    if (
                        get_thunder_cuda_manager().wait_event(
                            offload_event_queries.pop(f"offload_param:b{bwd_stage_id}"),
                            sync=True
                        )
                        == -1
                    ):
                        raise RuntimeError("wait_event failed")

    _cleanup()
    return forward_data_store