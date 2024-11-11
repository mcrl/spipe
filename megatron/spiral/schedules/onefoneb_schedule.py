from typing import Callable, Iterator, List, Optional, Union, Tuple
from enum import Enum

import torch
from torch._C._distributed_c10d import Work

from megatron import get_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type, get_attr_wrapped_model
from megatron.core.pipeline_parallel import forward_step, backward_step, p2p_communication
from megatron.spiral.initialize import get_thunder_cuda_manager
from megatron.spiral.debug import spiral_print
from megatron.spiral.init_context import SpiralParamStatus, set_module_spiral_status


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

    """Initialize variables."""

    # Activation tensors
    input_tensors: List[Tuple[torch.Tensor, List[Work]]] = [[] for _ in range(len(model))]
    output_tensors: List[Tuple[torch.Tensor, List[Work]]] = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads: List[Tuple[torch.Tensor, List[Work]]] = [[] for _ in range(len(model))]

    # Event dictionaries. Key is the event tag, thus an event necessarily requires it.
    prefetch_events = []
    compute_event_queries = {}
    offload_events = []
    free_events = []

    # Misc.
    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError("Interleaving is not supported with a different decoder sequence length.")

    if sequence_parallel:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // mpu.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    # Placeholders
    output_tensor = None
    fwd_wait_handles = None
    bwd_wait_handles = None

    """Offload grads and optimizer related"""
    offload_grad_after_bwd_stage = get_args().spiral_overlap_offload_grad
    optimize_after_bwd_stage = (
        offload_grad_after_bwd_stage and get_args().spiral_stage_optimizer
    )
    assert not optimize_after_bwd_stage, (
        "Currently, spipe 1f1b does not support optimize_after_bwd_stage,"
        "but there is no obstacle to implement it."
    )

    """Compute number of warmup and remaining microbatches."""
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
            # TODO: Implement all_warmup_microbatches
            raise RuntimeError(
                "Current implementation does not support all_warmup_microbatches"
                "(num_microbatches == pipeline_parallel_size)"
            )
        else:
            num_warmup_microbatches = (
                pipeline_parallel_size - pipeline_parallel_rank - 1
            ) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(
                num_warmup_microbatches, total_num_microbatches
            )
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    """Helper functions."""

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (
            pipeline_parallel_size * num_model_chunks
        )
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return (
                microbatch_id_in_group % pipeline_parallel_size
                == pipeline_parallel_size - 1
            )
        else:
            return False

    def prefetch_target_stage_id(microbatch_id: int) -> int:
        """Check if this microbatch should trigger prefetch. Return stage_id to prefetch else -1."""
        if _PHASE == PHASE.WARMUP:
            # if next fwd microbatch id (microbatch_id+1) translates to a different fwd stage id, prefetch
            next_forward_model_chunk_id = get_model_chunk_id(
                microbatch_id + 1, forward=True
            )
            if mpu.get_spiral_forward_virtual_rank() != next_forward_model_chunk_id:
                return next_forward_model_chunk_id
        elif _PHASE == PHASE.COOLDOWN:
            # skip prefetch if at the last wave of ppsize microbatches for the backward virtual rank 0
            if (
                mpu.get_spiral_backward_virtual_rank() == 0
                and microbatch_id >= total_num_microbatches - pipeline_parallel_size
            ):
                return -1
            # if next bwd microbatch id (microbatch_id+1) translates to a different bwd stage id, prefetch
            next_backward_model_chunk_id = get_model_chunk_id(
                microbatch_id + 1, forward=False
            )
            if mpu.get_spiral_backward_virtual_rank() != next_backward_model_chunk_id:
                return next_backward_model_chunk_id
        else:
            assert _PHASE == PHASE.STEADY
            if (
                mpu.get_spiral_backward_virtual_rank() != None
                and microbatch_id == num_microbatches_remaining - 1
            ):
                # if bwd microbatch id (microbatch_id) is the last steady state microbatch,
                # then if next bwd microbatch id (microbatch_id+1) translates to a different
                # bwd stage id, prefetch
                next_backward_model_chunk_id = get_model_chunk_id(
                    microbatch_id + 1, forward=False
                )
                if (
                    mpu.get_spiral_backward_virtual_rank()
                    != next_backward_model_chunk_id
                ):
                    return next_backward_model_chunk_id
            else:
                # else (i.e., not last bwd microbatch in the steady state),
                if mpu.get_spiral_forward_virtual_rank() != None:
                    next_backward_model_chunk_id = get_model_chunk_id(
                        microbatch_id - num_warmup_microbatches, forward=False
                    )
                    if (
                        mpu.get_spiral_forward_virtual_rank()
                        != next_backward_model_chunk_id
                    ):
                        return next_backward_model_chunk_id
                else:
                    assert mpu.get_spiral_backward_virtual_rank() != None
                    next_forward_model_chunk_id = get_model_chunk_id(
                        microbatch_id + 1 + num_warmup_microbatches, forward=True
                    )
                    if (
                        mpu.get_spiral_backward_virtual_rank()
                        != next_forward_model_chunk_id
                    ):
                        return next_forward_model_chunk_id
        return -1

    def should_free_stage(microbatch_id: int) -> bool:
        """Check if this microbatch should free the stage."""
        # NOTE: Currently implements simplified logic since we assume
        # "one microbatch ahead" prefetch at all times.
        # This is not optimal for warmup and cooldown phases.
        return prefetch_target_stage_id(microbatch_id) != -1

    def should_offload_grad(microbatch_id: int) -> bool:
        """Check if this microbatch should offload grad of the stage"""
        if (not offload_grad_after_bwd_stage) or (
            mpu.get_spiral_backward_virtual_rank() == None
        ):
            return False
        return is_last_microbatch_for_model_chunk(microbatch_id)

    def warmup_comm(k, output_tensor):
        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        input_tensors[next_forward_model_chunk_id].append(
            p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                dtype=dtype,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
                overlap_p2p_comm=True,
            )
        )

        if (
            k == (num_warmup_microbatches - 1)
            and not forward_only
            and not all_warmup_microbatches
        ):
            input_tensor_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    batch_p2p_comm=batch_p2p_comm,
                    dtype=dtype,
                    timers=timers,
                    overlap_p2p_comm=True,
                )
            )

    def steady_forward_comm(k, forward_k, output_tensor):
        # Last virtual stage no activation tensor to send
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                            forward=True)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
            output_tensor,
            recv_prev=recv_prev,
            tensor_shape=tensor_shape,
            dtype=dtype,
            batch_p2p_comm=batch_p2p_comm,
            timers=timers,
            overlap_p2p_comm=True,
        )

        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(
                (input_tensor, fwd_wait_handles)
            )

    def steady_backward_comm(k, backward_k, input_tensor_grad):
        # First virtual stage no activation gradient tensor to send
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None

        # Determine if the current virtual stage has an activation gradient tensor to receive
        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False
            )
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k + 1, forward=False
            )

        output_tensor_grad, bwd_wait_handles = (
            p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                dtype=dtype,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
                overlap_p2p_comm=True,
            )
        )

        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(
                (output_tensor_grad, bwd_wait_handles)
            )

    def cooldown_comm(k, input_tensor_grad):
        next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if next_backward_model_chunk_id == (num_model_chunks - 1):
                recv_next = False
        if k == (total_num_microbatches - 1):
            recv_next = False
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            # comm under compute stream since blocking recv comm
            output_tensor_grad = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers)
            output_tensor_grads[next_backward_model_chunk_id].append(
                (output_tensor_grad, None)
            )  # recv_handle is None since blocking comm.

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

    def _wait_reqs(reqs: List[Work]):
        if reqs is not None:
            for req in reqs:
                req.wait()

    ####################################################################################################
    """ Critical helper functions """
    ####################################################################################################

    def forward_step_helper(microbatch_id):
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        torch.cuda.nvtx.range_push(f"f[{model_chunk_id}]m[{microbatch_id}]")
        if _DEBUG_SCHEDULE:
            spiral_print(f"fwd stage {model_chunk_id} microbatch {microbatch_id}")

        # TODO: launch param synchronization for next model chunk

        # compute stream wait for prefetch current stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            if len(prefetch_events) > 0:
                if get_thunder_cuda_manager().wait_event(prefetch_events.pop(0)) == -1:
                    raise RuntimeError("wait_event failed")
        # end compute stream

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            if len(free_events) > 0:
                if get_thunder_cuda_manager().wait_event(free_events.pop(0)) == -1:
                    raise RuntimeError("wait_event failed")
            _prefetch_stage_id = prefetch_target_stage_id(microbatch_id)
            if _prefetch_stage_id != -1:
                if _DEBUG_SCHEDULE:
                    spiral_print(f" prefetch {_prefetch_stage_id}")
                model[_prefetch_stage_id].spiral_fetch(non_blocking=True)
                tag = "prefetch:" + f"{_prefetch_stage_id}"
                prefetch_next = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag=tag,
                    post_wait_fn=lambda: set_module_spiral_status(
                        model[_prefetch_stage_id], SpiralParamStatus.GPU
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_next) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_events.append(prefetch_next)
        # end prefetch stream

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            # set input tensor
            if mpu.is_pipeline_first_stage():
                input_tensors[0].append(
                    (None, None)
                ) # recv_handle is None since receiving None
            input_tensor, fwd_wait_handles = input_tensors[
                mpu.get_spiral_forward_virtual_rank()
            ][-1] # do not pop for reuse in backward step

            # wait for recv input tensor
            # NOTE (SpiralPipe) Must be done in compute stream to avoid error
            _wait_reqs(fwd_wait_handles)

            # forward step
            if _DEBUG_SCHEDULE:
                spiral_print(f" call forward_step: {model_chunk_id}")
            output_tensor = forward_step(
                forward_step_func,
                data_iterator[model_chunk_id],
                model[model_chunk_id],
                num_microbatches,
                input_tensor,
                forward_data_store,
                timers,
                collect_non_loss_data,
                dtype,
                enable_autocast,
            )
            output_tensors[model_chunk_id].append(output_tensor)

            if should_free_stage(microbatch_id):
                # notify free stream to act
                compute_microbatches_end = get_thunder_cuda_manager().Event(
                    "compute", "free", tag=f"compute:{model_chunk_id}end"
                )
                if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end
        # end compute stream

        # TODO: check forwad_only input/output tensor handling

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("free")):
            if should_free_stage(microbatch_id):
                if _DEBUG_SCHEDULE:
                    spiral_print(f" free {model_chunk_id}")

                # synchronize compute stream before free
                if (
                    get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:{model_chunk_id}end"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

                # free fwd stage
                model[model_chunk_id].spiral_free()

                free_curr = get_thunder_cuda_manager().Event(
                    "free",
                    "prefetch",
                    tag=f"free:{model_chunk_id}",
                )
                if get_thunder_cuda_manager().record_event(free_curr) == -1:
                    raise RuntimeError("record_event failed")
                free_events.append(free_curr)
        # end free stream

        torch.cuda.nvtx.range_pop()
        return output_tensor
    # end forward_step_helper

    def backward_step_helper(microbatch_id):
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        torch.cuda.nvtx.range_push(f"b[{model_chunk_id}]m[{microbatch_id}]")
        if _DEBUG_SCHEDULE:
            spiral_print(f"bwd stage {model_chunk_id} microbatch {microbatch_id}")

        # TODO: launch grad synchronization (default)

        # compute stream wait for prefetch current stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            if len(prefetch_events) > 0:
                if get_thunder_cuda_manager().wait_event(prefetch_events.pop(0)) == -1:
                    raise RuntimeError("wait_event failed")
        # end compute stream

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            if len(free_events) > 0:
                if get_thunder_cuda_manager().wait_event(free_events.pop(0)) == -1:
                    raise RuntimeError("wait_event failed")
            _prefetch_stage_id = prefetch_target_stage_id(microbatch_id)
            if _prefetch_stage_id != -1:
                if _DEBUG_SCHEDULE:
                    spiral_print(f" prefetch {_prefetch_stage_id}")
                model[_prefetch_stage_id].spiral_fetch(non_blocking=True)
                tag = "prefetch:" + f"{_prefetch_stage_id}"
                prefetch_next = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag=tag,
                    post_wait_fn=lambda: set_module_spiral_status(
                        model[_prefetch_stage_id], SpiralParamStatus.GPU
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_next) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_events.append(prefetch_next)
        # end prefetch stream

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            # set output tensor grad
            if mpu.is_pipeline_last_stage():
                if len(output_tensor_grads[model_chunk_id]) == 0:
                    output_tensor_grads[model_chunk_id].append(
                        (None, None)
                    ) # recv_handle is None since receiving None
            input_tensor, _ = input_tensors[model_chunk_id].pop(0)  # fwd_wait_handles is already
                                                                    # waited in the forward step
            output_tensor = output_tensors[model_chunk_id].pop(0)
            output_tensor_grad, bwd_wait_handles = output_tensor_grads[model_chunk_id].pop(0)

            # wait for recv output tensor grad
            # NOTE (SpiralPipe) Must be done in compute stream to avoid error
            _wait_reqs(bwd_wait_handles)

            # backward step
            if _DEBUG_SCHEDULE:
                spiral_print(f" call backward_step: {model_chunk_id}")
            input_tensor_grad = backward_step(
                grad_scaler,
                input_tensor,
                output_tensor,
                output_tensor_grad,
                model_type,
                timers,
                deallocate_pipeline_outputs,
            )

            if should_free_stage(microbatch_id):
                # notify free stream to act
                compute_microbatches_end = get_thunder_cuda_manager().Event(
                    "compute", "free", tag=f"compute:{model_chunk_id}end:free"
                )
                if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end

            if offload_grad_after_bwd_stage and should_offload_grad(microbatch_id):
                # notify offload stream to act
                compute_microbatches_end = get_thunder_cuda_manager().Event(
                    "compute", "offload", tag=f"compute:{model_chunk_id}end:offload"
                )
                if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end
        # end compute stream

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("free")):
            if should_free_stage(microbatch_id):
                if _DEBUG_SCHEDULE:
                    spiral_print(f" free {model_chunk_id}")

                # synchronize compute stream before free
                if (
                    get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:{model_chunk_id}end:free"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

                # free bwd stage
                model[model_chunk_id].spiral_free()

                free_curr = get_thunder_cuda_manager().Event(
                    "free",
                    "prefetch",
                    tag=f"free:{model_chunk_id}",
                )
                if get_thunder_cuda_manager().record_event(free_curr) == -1:
                    raise RuntimeError("record_event failed")
                free_events.append(free_curr)
        # end free stream

        # TODO: launch grad synchronization (custom grad sync)

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("offload")):
            if offload_grad_after_bwd_stage and should_offload_grad(microbatch_id):
                if _DEBUG_SCHEDULE:
                    spiral_print(f" offload_grad {model_chunk_id}")

                # synchronize compute stream before offload
                if (
                    get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:{model_chunk_id}end:offload"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

                # if not spiral stage optimizer, then gradient should be manually reduced
                if optimize_after_bwd_stage:
                    # TODO: implement
                    pass
                else:
                    model[model_chunk_id].allreduce_gradients()

                # offload grads
                # NOTE: this currently does not care about cpu-gpu hybrid optimizer
                model[model_chunk_id].spiral_offload_grad(non_blocking=True)
                offload_grad_curr = get_thunder_cuda_manager().Event(
                    "offload",
                    None,
                    tag=f"offload_grad:b{model_chunk_id}"
                )
                if get_thunder_cuda_manager().record_event(offload_grad_curr) == -1:
                    raise RuntimeError("record_event failed")
                offload_events.append(offload_grad_curr)

                # if not spiral stage optimizer, then optimizer will be executed after iteration
                if optimize_after_bwd_stage:
                    # TODO: Implement this
                    pass

                # free bwd stage grads (spiral_free_grad is cpu job with tensor.record_stream)
                model[model_chunk_id].spiral_free_grad()
        # end offload & free grad

        torch.cuda.nvtx.range_pop()
        return input_tensor_grad
    # end backward_step_helper

    ####################################################################################################
    """ Start training """
    ####################################################################################################

    # prefetch 1st fwd stage
    with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
        if _DEBUG_SCHEDULE:
            spiral_print(f" prefetch {0}")
        model[0].spiral_fetch(non_blocking=True)
        prefetch_f0 = get_thunder_cuda_manager().Event(
            "prefetch",
            "compute",
            tag="prefetch:0",
            post_wait_fn=lambda: set_module_spiral_status(
                model[0], SpiralParamStatus.GPU
            ),
        )
        if get_thunder_cuda_manager().record_event(prefetch_f0) == -1:
            raise RuntimeError("record_event failed")
        prefetch_events.append(prefetch_f0)

    ##### warmup
    _PHASE = PHASE.WARMUP
    if _DEBUG_SCHEDULE:
        spiral_print(f"warmup = {num_warmup_microbatches}")

    # Receive the first input tensor
    mpu.set_spiral_forward_virtual_rank(0)
    with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
        # comm under compute stream since blocking recv comm
        input_tensor = p2p_communication.recv_forward(
            tensor_shape, dtype=dtype, batch_p2p_comm=batch_p2p_comm, timers=timers
        )
        input_tensors[0].append((input_tensor, None)) # recv_handle is None since blocking comm.
    mpu.set_spiral_forward_virtual_rank(None)

    # warmup microbatches
    for k in range(num_warmup_microbatches):
        # set fwd virtual rank
        mpu.set_spiral_forward_virtual_rank(get_model_chunk_id(k, forward=True))

        # run forward step
        output_tensor = forward_step_helper(k)

        # warmup communication
        warmup_comm(k, output_tensor)

        # TODO: Check currently removed `deallocate_output_tensor` call

        # unset fwd virtual rank
        mpu.set_spiral_forward_virtual_rank(None)
    # end warmup microbatches

    ##### steady
    _PHASE = PHASE.STEADY
    if _DEBUG_SCHEDULE:
        spiral_print(f"steady state = {num_microbatches_remaining}")

    # steady state microbatches
    for k in range(num_microbatches_remaining):
        if overlap_p2p_comm:
            # Forward pass.
            forward_k = k + num_warmup_microbatches
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)

            # set fwd virtual rank
            mpu.set_spiral_forward_virtual_rank(forward_model_chunk_id)

            # TODO: Check currently removed `deallocate_output_tensor` call

            # run forward step
            output_tensor = forward_step_helper(forward_k)

            # steady forward communication
            steady_forward_comm(k, forward_k, output_tensor)

            # unset fwd virtual rank
            mpu.set_spiral_forward_virtual_rank(None)

            # Backward pass.
            backward_k = k
            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)

            # set bwd virtual rank
            mpu.set_spiral_backward_virtual_rank(backward_model_chunk_id)

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # wait for recv input tensor
                if bwd_wait_handles is not None:
                    # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                    for req in bwd_wait_handles:
                        req.wait()
            # end compute stream

            # run backward step
            input_tensor_grad = backward_step_helper(backward_k)

            # steady backward communication
            steady_backward_comm(k, backward_k, input_tensor_grad)

            # unset bwd virtual rank
            mpu.set_spiral_backward_virtual_rank(None)

        # TODO: Handle overlap_p2p_comm=False case

    # TODO: Check currently removed `deallocate_output_tensor` call

    ##### cooldown
    _PHASE = PHASE.COOLDOWN
    if _DEBUG_SCHEDULE:
        spiral_print(f"cooldown = {total_num_microbatches - num_microbatches_remaining}")

    if not forward_only:
        # TODO: Handle all warmup microbatches case

        for k in range(num_microbatches_remaining, total_num_microbatches):
            # set bwd virtual rank
            mpu.set_spiral_backward_virtual_rank(get_model_chunk_id(k, forward=False))

            # run backward step
            input_tensor_grad = backward_step_helper(k)

            # cooldown communication
            cooldown_comm(k, input_tensor_grad)

            # unset bwd virtual rank
            mpu.set_spiral_backward_virtual_rank(None)

    # TODO: Launch any remaining grad reductions

    # wait for all offload events
    while len(offload_events) > 0:
        if get_thunder_cuda_manager().wait_event(offload_events.pop(0), sync=True) == -1:
            raise RuntimeError("wait_event failed")

    _cleanup()
    return forward_data_store
