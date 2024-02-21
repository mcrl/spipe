import contextlib
import warnings
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type
from megatron.core.pipeline_parallel import forward_step, backward_step
from megatron.spiral import get_thunder_cuda_manager
from megatron.spiral.debug import spiral_print
import megatron.spiral.p2p_communication as spiral_p2p
from megatron.spiral.init_context import SpiralParamStatus
from megatron.spiral.utils import is_spiral_param


# Types
Shape = Union[List[int], torch.Size]


def forward_backward_pipelining_with_spiral(
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
    overlap_p2p_comm: bool = False,  # TODO (mcrl) check
    batch_p2p_comm: bool = False,  # TODO (mcrl) check
    forward_only: bool = False,
    timers: Callable = None,
    collect_non_loss_data: bool = False,
    enable_autocast: bool = False,
    deallocate_pipeline_outputs: bool = False,
    no_sync_func: Optional[Callable] = None,
    grad_sync_func: Optional[Callable] = None,
    param_sync_func: Optional[Callable] = None,
):
    """Run sprial schedule, with communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    assert isinstance(
        model, list
    ), "Spiral pipeline parallelism expected model chunking by stage"
    assert isinstance(
        data_iterator, list
    ), "Spiral pipeline parallelism expected each model chunk to have a data iterator"

    # TODO (mcrl) disable async grad reductions?
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

    if mpu.get_spiral_pipeline_parallel_forward_virtual_rank() is not None:
        warnings.warn(
            "Spiral pipeline parallel forward virtual rank is not None on scheule entry. There may be a bug."
        )
    if mpu.get_spiral_pipeline_parallel_backward_virtual_rank() is not None:
        warnings.warn(
            "Spiral pipeline parallel backward virtual rank is not None on scheule entry. There may be a bug."
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
    # Start training
    forward_data_store = []

    # input ckpts
    if not forward_only:
        input_tensor_ckpts = [
            [] for _ in range(mpu.get_spiral_pipeline_parallel_backward_virtual_size())
        ]

    recv_handles = None

    prefetch_event_queries = []
    compute_event_queries = []
    offload_event_queries = []

    prefetch_query = None
    compute_query = None
    offload_query = None

    # TODO (mcrl) below line is temporarily added to sync between minibatches. Remove after optimizer sync is implemented
    # torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())

    # prefetch 1st fwd stage
    with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
        model[0].spiral_fetch(non_blocking=True)
        prefetch_query = get_thunder_cuda_manager().Event(
            "prefetch",
            "compute",
            tag="prefetch:f0",
            post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                model[0], SpiralParamStatus.ACTIVE
            ),
        )
        if get_thunder_cuda_manager().record_event(prefetch_query) == -1:
            raise RuntimeError("record_event failed")
        prefetch_event_queries.append(prefetch_query)

    # fwd
    for fwd_stage_id in range(mpu.get_spiral_pipeline_parallel_forward_virtual_size()):
        spiral_print(f"Start fwd stage {fwd_stage_id}")
        mpu.set_spiral_pipeline_parallel_forward_virtual_rank(fwd_stage_id)
        # input_tensor_ckpt_dst = mpu.get_pipeline_model_parallel_world_size() - mpu.get_pipeline_model_parallel_rank() - 1 # TODO (mcrl) can be moved out of fwd for loop

        assert (
            hasattr(model[fwd_stage_id], "spiral_forward_stage_id")
            and getattr(model[fwd_stage_id], "spiral_forward_stage_id") == fwd_stage_id
        ), "Forward stage ID mismatch between virtual rank and model."

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            if not fwd_stage_id == 0:
                if (
                    get_thunder_cuda_manager().wait_event(offload_event_queries.pop(0))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

            if not (
                forward_only
                and fwd_stage_id
                == mpu.get_spiral_pipeline_parallel_forward_virtual_size() - 1
            ):
                model[fwd_stage_id + 1].spiral_fetch(non_blocking=True)
                tag = (
                    "prefetch:" + f"f{fwd_stage_id + 1}"
                    if fwd_stage_id
                    < mpu.get_spiral_pipeline_parallel_forward_virtual_size()
                    else f"b{mpu.get_spiral_pipeline_parallel_backward_virtual_size() - 1}"
                )
                prefetch_query = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag=tag,
                    post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                        model[fwd_stage_id + 1], SpiralParamStatus.ACTIVE
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_query) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_event_queries.append(prefetch_query)

        # compute stream wait for prefetch current stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            if (
                get_thunder_cuda_manager().wait_event(prefetch_event_queries.pop(0))
                == -1
            ):
                raise RuntimeError("wait_event failed")

        # fwd microbatches
        for i in range(num_microbatches):
            spiral_print(f" microbatch {i}")
            torch.cuda.nvtx.range_push(f"f[{fwd_stage_id}]m[{i}]")

            # set input tensor
            if mpu.is_pipeline_first_stage():
                input_tensor = None
                recv_handles = None
            else:
                input_tensor, recv_handles = spiral_p2p.recv_input_tensor(
                    tensor_shape,
                    dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    overlap_p2p_comm=overlap_p2p_comm,
                    timers=timers,
                )

            # wait for recv input tensor
            if recv_handles is not None:
                for req in recv_handles:
                    req.wait()

            # input ckpt
            # p2p_communication.send_ckpt(input_tensor, input_tensor_ckpt_dst, timers=timers)

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
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

                compute_query = get_thunder_cuda_manager().Event(
                    "compute",
                    None,
                    tag=f"compute:f{fwd_stage_id}m{i}",
                )
                if get_thunder_cuda_manager().record_event(compute_query) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries.append(compute_query)

                if i == num_microbatches - 1:
                    # notify offload stream to act
                    compute_query = get_thunder_cuda_manager().Event(
                        "compute", "offload", tag=f"compute:f{fwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_query) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries.append(compute_query)

            # send output tensor
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(0))
                == -1
            ):
                # wait event with tag=f"compute:f{fwd_stage_id}m{i}"
                raise RuntimeError("wait_event failed")
            if not mpu.is_pipeline_last_stage():
                _ = spiral_p2p.send_output_tensor(
                    output_tensor,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )

            torch.cuda.nvtx.range_pop()
        # end fwd microbatches

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("offload")):
            while len(compute_event_queries) > 0:
                if (
                    get_thunder_cuda_manager().wait_event(compute_event_queries.pop(0))
                    == -1
                ):
                    # wait event with tag=f"compute:f{fwd_stage_id}end"
                    raise RuntimeError("wait_event failed")
            model[fwd_stage_id].spiral_free()

            if not (
                forward_only
                and fwd_stage_id
                == mpu.get_spiral_pipeline_parallel_forward_virtual_size() - 1
            ):
                offload_query = get_thunder_cuda_manager().Event(
                    "offload",
                    "prefetch",
                    tag=f"offload:f{fwd_stage_id}",
                    post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                        model[fwd_stage_id], SpiralParamStatus.REMOTE
                    ),
                )
                if get_thunder_cuda_manager().record_event(offload_query) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries.append(offload_query)
        mpu.set_spiral_pipeline_parallel_forward_virtual_rank(None)

    if forward_only:
        return forward_data_store
    # end fwd

    # bwd
    for bwd_stage_id in range(
        mpu.get_spiral_pipeline_parallel_backward_virtual_size() - 1, -1, -1
    ):
        spiral_print(f"Start bwd stage {bwd_stage_id}")
        mpu.set_spiral_pipeline_parallel_backward_virtual_rank(bwd_stage_id)

        assert (
            hasattr(model[-bwd_stage_id - 1], "spiral_backward_stage_id")
            and getattr(model[-bwd_stage_id - 1], "spiral_backward_stage_id")
            == bwd_stage_id
        ), "Backward stage ID mismatch between virtual rank and model."

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            if (
                get_thunder_cuda_manager().wait_event(offload_event_queries.pop(0))
                == -1
            ):
                raise RuntimeError("wait_event failed")

            if not bwd_stage_id == 0:
                model[-bwd_stage_id].spiral_fetch(non_blocking=True)
                prefetch_query = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag="prefetch:" + f"b{bwd_stage_id - 1}",
                    post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                        model[-bwd_stage_id], SpiralParamStatus.ACTIVE
                    ),
                )
                if get_thunder_cuda_manager().record_event(prefetch_query) == -1:
                    raise RuntimeError("record_event failed")
                prefetch_event_queries.append(prefetch_query)

        # compute stream wait for prefetch current stage
        # NOTE: prefetch for last bwd stage is called in the fwd loop
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
            if (
                get_thunder_cuda_manager().wait_event(prefetch_event_queries.pop(0))
                == -1
            ):
                raise RuntimeError("wait_event failed")

        # TODO (mcrl) temporary code; remove after receive ckpts
        input_tensor_ckpt = (
            torch.ones(
                tensor_shape,
                dtype=torch.float,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )
            if not mpu.is_pipeline_first_stage()
            else None
        )

        # bwd microbatches
        for i in range(num_microbatches):
            spiral_print(f" microbatch {i}")
            torch.cuda.nvtx.range_push(f"b[{bwd_stage_id}]m[{i}]")

            # set output tensor grad
            if mpu.is_pipeline_last_stage():
                output_tensor_grad = None
                recv_handles = None
            else:
                output_tensor_grad, recv_handles = spiral_p2p.recv_output_tensor_grad(
                    tensor_shape,
                    dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    overlap_p2p_comm=overlap_p2p_comm,
                    timers=timers,
                )

            # wait for recv output tensor grad
            if recv_handles is not None:
                for req in recv_handles:
                    req.wait()

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                output_tensor = forward_step(
                    forward_step_func,
                    data_iterator[bwd_stage_id + mpu.get_spiral_pipeline_parallel_forward_virtual_size()],
                    model[-bwd_stage_id - 1],
                    num_microbatches,
                    input_tensor_ckpt,
                    [],
                    timers,
                    collect_non_loss_data,
                    dtype,
                    enable_autocast,
                )

                input_tensor_grad = backward_step(
                    grad_scaler,
                    input_tensor_ckpt,
                    output_tensor,
                    output_tensor_grad,
                    model_type,
                    timers,
                    deallocate_pipeline_outputs,
                )

                compute_query = get_thunder_cuda_manager().Event(
                    "compute",
                    None,
                    tag=f"compute:b{bwd_stage_id}m{i}",
                )
                if get_thunder_cuda_manager().record_event(compute_query) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries.append(compute_query)

                if i == num_microbatches - 1:
                    # notify offload stream to act
                    compute_query = get_thunder_cuda_manager().Event(
                        "compute", "offload", tag=f"compute:b{bwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_query) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries.append(compute_query)

            # send input tensor grad
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(0))
                == -1
            ):
                # wait event with tag=f"compute:b{bwd_stage_id}m{i}"
                raise RuntimeError("wait_event failed")
            if not mpu.is_pipeline_first_stage():
                _ = spiral_p2p.send_input_tensor_grad(
                    input_tensor_grad,
                    overlap_p2p_comm=overlap_p2p_comm,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                )

            torch.cuda.nvtx.range_pop()
        # end bwd microbatches

        with torch.cuda.stream(get_thunder_cuda_manager().Stream("offload")):
            while len(compute_event_queries) > 0:
                if (
                    get_thunder_cuda_manager().wait_event(compute_event_queries.pop(0))
                    == -1
                ):
                    # wait event with tag=f"compute:b{bwd_stage_id}end"
                    raise RuntimeError("wait_event failed")
            model[-bwd_stage_id - 1].spiral_offload_grad(non_blocking=True)
            model[-bwd_stage_id - 1].spiral_free()

            if not bwd_stage_id == 0:
                offload_query = get_thunder_cuda_manager().Event(
                    "offload",
                    "prefetch",
                    tag=f"offload:b{bwd_stage_id}",
                    post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                        model[-bwd_stage_id - 1], SpiralParamStatus.REMOTE
                    ),
                )
                if get_thunder_cuda_manager().record_event(offload_query) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries.append(offload_query)
        mpu.set_spiral_pipeline_parallel_backward_virtual_rank(None)
    # end bwd

    return forward_data_store


def _post_wait_set_spiral_param_status(module, status: SpiralParamStatus):
    for param in module.parameters(recurse=True):
        if is_spiral_param(param):
            param.spiral_param_status = status
