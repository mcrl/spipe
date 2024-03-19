import contextlib
import warnings
import nvtx
import sys
from typing import Callable, Iterator, List, Optional, Union, Tuple
from enum import Enum
from concurrent.futures import wait, as_completed

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_num_microbatches, get_args, get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type, get_attr_wrapped_model
from megatron.core.pipeline_parallel import forward_step, backward_step
from megatron.spiral.initialize import get_thunder_cuda_manager
from megatron.spiral.debug import spiral_print
import megatron.spiral.p2p_communication as spiral_p2p
from megatron.spiral.init_context import SpiralParamStatus
from megatron.spiral.utils import is_spiral_param
from megatron.spiral.generic import ContextManagers
import megatron.spiral.build_state as sbs


# Due to Tuple() ctor support from python >= 3.9
_PYTHON_VERSION = sys.version_info

# Types
Shape = Union[List[int], torch.Size]


class CkptSendRecvType(Enum):
    SEND = "send"
    RECV = "recv"


CkptSendRecvOp = Tuple[CkptSendRecvType, int, int]  # (comm_type, phase_id, rank)


class CkptSendRecvSchedule:
    """A schedule for sending and receiving input tensor checkpoints.

    Construct global schedule for all timestep of all rank.
    A timestep is a time unit for a micro-batch fwd/bwd computation.
    All timestep means from beginning of the minibatch on PP rank 0 to the end of the minibatch on PP rank N-1.
    Thus, a rank's schedule includes non_compute timesteps where no fwd/bwd computation is performed due to pipeline fill/drain.
    non_compute timesteps are blank timesteps in below example

    Example:
    _SPIRAL_FORWARD_VIRTUAL_SIZE = 3
    _SPIRAL_BACKWARD_VIRTUAL_SIZE = 3
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 4

        | ts0   | ts1   | ts1   | ts1   | ts1   | ts1   | ts1   | ts1   | ts1   | ts1   | ts1   |
    r0  | fstart| comp  | comp  | comp  | bstart| comp  | comp  | comp  |       |       |       |
    r1  |       | comp  | comp  | comp  | comp  | comp  | comp  | comp  | comp  |       |       |
    r2  |       |       | comp  | comp  | comp  | comp  | comp  | comp  | comp  | comp  |       |
    r3  |       |       |       | comp  | comp  | comp  | fend  | comp  | comp  | comp  | bend  |

    """

    def __init__(self, num_microbatches: Optional[int]):
        if num_microbatches is None:
            num_microbatches = get_num_microbatches()
        self.num_microbatches = num_microbatches

        num_compute_ts = (
            mpu.get_spiral_forward_virtual_size()
            + mpu.get_spiral_backward_virtual_size()
        ) * self.num_microbatches
        num_non_compute_ts = mpu.get_pipeline_model_parallel_world_size() - 1
        self.total_ts = num_compute_ts + num_non_compute_ts
        self.global_schedule: List[List[List[CkptSendRecvOp]]] = [
            [[] for _ in range(self.total_ts)]
            for _ in range(mpu.get_pipeline_model_parallel_world_size())
        ]

        self._set_recv_schedule()
        self._set_send_schedule()
        self._schedule_generator = self._generator()

    def _set_recv_schedule(self):
        for pp_rank in range(mpu.get_pipeline_model_parallel_world_size()):
            num_pre_pipeline_non_compute_ts = pp_rank
            recv_start_ts = (
                num_pre_pipeline_non_compute_ts
                + mpu.get_spiral_forward_virtual_size() * self.num_microbatches
            )
            curr_ts = recv_start_ts

            for bwd_stage_id in range(
                mpu.get_spiral_backward_virtual_size() - 1, -1, -1
            ):
                bwd_phases_start = (
                    sbs.get_spiral_backward_stage_build_phase_size()
                    * mpu.get_pipeline_model_parallel_world_size()
                    * bwd_stage_id
                    + sbs.get_spiral_backward_stage_build_phase_size()
                    * (mpu.get_pipeline_model_parallel_world_size() - pp_rank - 1)
                )
                recv_phase = bwd_phases_start  # modify when needed
                for _ in range(self.num_microbatches):
                    if _PYTHON_VERSION >= (3, 9):
                        _op = CkptSendRecvOp(
                            CkptSendRecvType.RECV,
                            recv_phase,
                            sbs.get_pp_rank_for_fwd_phase(recv_phase),
                        )
                    else:
                        _op = (
                            CkptSendRecvType.RECV,
                            recv_phase,
                            sbs.get_pp_rank_for_fwd_phase(recv_phase),
                        )
                    self.global_schedule[pp_rank][curr_ts].append(_op)
                    curr_ts += 1

    def _set_send_schedule(self):
        # TODO (SpiralPipe) sends of only self is required
        for j in range(self.total_ts):
            for pp_rank in range(mpu.get_pipeline_model_parallel_world_size()):
                for recv_comm in filter(
                    lambda x: x[0] == CkptSendRecvType.RECV,
                    self.global_schedule[pp_rank][j],
                ):
                    # add send to dst for the sender rank queue
                    _src_rank = recv_comm[2]
                    if _PYTHON_VERSION >= (3, 9):
                        _op = CkptSendRecvOp(
                            CkptSendRecvType.SEND, recv_comm[1], pp_rank
                        )
                    else:
                        _op = (CkptSendRecvType.SEND, recv_comm[1], pp_rank)
                    self.global_schedule[_src_rank][j].append(_op)

    def __str__(self) -> str:
        _str = ""
        for pp_rank in range(len(self.global_schedule)):
            _str += f"rank {pp_rank}\n"
            for ts, comms in enumerate(self.global_schedule[pp_rank]):
                _str += f"\tts {ts}: {comms}\n"
        return _str

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._schedule_generator)

    def _generator(self):
        for comms in self.global_schedule[mpu.get_pipeline_model_parallel_rank()]:
            yield comms


def _post_wait_set_spiral_param_status(module, status: SpiralParamStatus):
    for param in module.parameters(recurse=True):
        if is_spiral_param(param):
            assert hasattr(param, "spiral_status"), "spiral_status not found in param"
            setattr(param, "spiral_status", status)


def _weight_update(optimizer, bwd_stage_id, event_query, args, timers):
    torch.cuda.nvtx.range_push(f"opt b[{bwd_stage_id}]")
    if (
        get_thunder_cuda_manager().wait_event(event_query, sync=True)
        == -1
    ):
        raise RuntimeError("sync_event failed")
    optimizer.set_bwd_stage(bwd_stage_id)
    if timers is not None:
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    # TODO (SpiralPipe) timers is None. Fix it
    # optimizer.step(args, timers)
    optimizer.step(args, get_timers())
    if timers is not None:
        timers('optimizer').stop()
    torch.cuda.nvtx.range_pop()


def forward_backward_pipelining_with_spiral_remap(
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

    optimize_after_bwd_stage = False
    if get_args().spiral_stage_optimizer:
        assert "spiral_stage_optimizer" in kwargs, "spiral_stage_optimizer not found in kwargs"
        optimize_after_bwd_stage = True
        optimizer = kwargs["spiral_stage_optimizer"]
        grad_scaler = kwargs["spiral_grad_scaler"]
        optimizer_threads_status = []

    # TODO (SpiralPipe) Move to spiral_p2p
    @nvtx.annotate("comm_input_ckpt", color="red")
    def comm_input_ckpt(schedule):
        spiral_print(f"comm: {schedule}")

        tensor_sends = []
        send_ranks = []
        recv_ranks = []

        insert_idx_to_recvs = []
        insert_value_to_recvs = []

        recv_idx = 0
        send_idx = 0
        for idx, (comm_type, phase_id, rank) in enumerate(schedule):

            phase_fwd_rank = sbs.get_pp_rank_for_fwd_phase(phase_id)
            local_stage_id, local_phase_id = sbs.fwd_phase2local_stage_phase(phase_id)

            _prefix = str(schedule[idx])

            if comm_type == CkptSendRecvType.RECV:
                assert (
                    phase_fwd_rank == rank
                ), f"[RECV] phase_fwd_rank = {phase_fwd_rank}, rank = {rank} mismatch"

                if phase_id == 0:
                    spiral_print(_prefix + " Phase 0 => insert None to recvs")
                    insert_idx_to_recvs.append(recv_idx)
                    insert_value_to_recvs.append(None)
                elif rank == mpu.get_pipeline_model_parallel_rank():
                    spiral_print(_prefix + " Self recv => pop ckpt and insert to recvs")
                    insert_idx_to_recvs.append(recv_idx)
                    # NOTE (SpiralPipe) Using the popped input tensor from original fwd stage can lead to trouble, as it contains
                    # the computation graph constructed already. We are prone to this error when #fwd != #bwd and hence the same rank
                    # can perform fwd and bwd of the same phase. Re-computation using this tensor will lead to duplicated computation
                    # graph being constructed. So, we currently perform the original FWD in torch.no_grad() mode, and then recompute
                    # in BWD without torch.no_grad(). Another solution may exist.
                    input_ckpt_ = (
                        model[local_stage_id]
                        .module[local_phase_id]
                        .spiral_input_tensors.popleft()
                    )
                    assert (
                        input_ckpt_.requires_grad
                    ), "Input ckpt must require grad before feeding to BWD"
                    insert_value_to_recvs.append(input_ckpt_)
                else:
                    spiral_print(
                        _prefix + " Recv from other rank => append to recv_ranks"
                    )
                    recv_ranks.append(rank)
                recv_idx += 1

            elif comm_type == CkptSendRecvType.SEND:
                assert (
                    phase_fwd_rank == mpu.get_pipeline_model_parallel_rank()
                ), f"[SEND] phase_fwd_rank = {phase_fwd_rank}, self = {mpu.get_pipeline_model_parallel_rank()} mismatch"

                if phase_id == 0:
                    spiral_print(_prefix + " Phase 0 => skip")
                elif rank == mpu.get_pipeline_model_parallel_rank():
                    spiral_print(_prefix + " Self send => skip")
                else:
                    spiral_print(
                        _prefix
                        + " Send to other rank => pop ckpt & append to tensor sends and append to send_ranks"
                    )
                    tensor_sends.append(
                        model[local_stage_id]
                        .module[local_phase_id]
                        .spiral_input_tensors.popleft()
                    )
                    send_ranks.append(rank)
                send_idx += 1
            else:
                raise RuntimeError(f"Invalid comm type {comm_type}")

        recvs, reqs = None, []  # placeholder
        if len(send_ranks) > 0 or len(recv_ranks) > 0:
            recvs, reqs = spiral_p2p._communicate(
                tensor_sends=tensor_sends if len(tensor_sends) > 0 else None,
                send_ranks=send_ranks if len(send_ranks) > 0 else None,
                recv_ranks=recv_ranks if len(recv_ranks) > 0 else None,
                tensor_shape=tensor_shape,
                group=mpu.get_spiral_input_tensor_ckpt_group(),
                batch_p2p_comm=True,
                wait_on_reqs=False,
                dtype=dtype,
            )

        for recv_idx, recv_val in zip(insert_idx_to_recvs, insert_value_to_recvs):
            if recvs is None:
                recvs = []
            recvs.insert(recv_idx, recv_val)

        return recvs, reqs
    # end comm_input_ckpt()

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

    # Event dictionaries. Key is the event tag, thus an event necessarily requires it.
    prefetch_event_queries = {}
    compute_event_queries = {}
    offload_event_queries = {}

    # Placeholders
    recv_handles = None

    """ Start training """

    # nop pre-pipeline non-compute timesteps
    __num_pre_pipeline_non_compute_ts = mpu.get_pipeline_model_parallel_rank()
    for _ in range(__num_pre_pipeline_non_compute_ts):
        if not forward_only:
            next(ckpt_send_recv_schedule)

    # prefetch 1st fwd stage
    with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
        model[0].spiral_fetch(non_blocking=True)
        prefetch_f0 = get_thunder_cuda_manager().Event(
            "prefetch",
            "compute",
            tag="prefetch:f0",
            post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                model[0], SpiralParamStatus.ACTIVE
            ),
        )
        if get_thunder_cuda_manager().record_event(prefetch_f0) == -1:
            raise RuntimeError("record_event failed")
        prefetch_event_queries[prefetch_f0.tag] = prefetch_f0

    # fwd
    for fwd_stage_id in range(mpu.get_spiral_forward_virtual_size()):
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
                    get_thunder_cuda_manager().wait_event(offload_event_queries.pop(f"free:f{fwd_stage_id - 1}"))
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
                    post_wait_fn=lambda fwd_stage_id=fwd_stage_id: _post_wait_set_spiral_param_status(
                        model[fwd_stage_id + 1], SpiralParamStatus.ACTIVE
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
        for i in range(num_microbatches):
            spiral_print(f" microbatch {i}")
            torch.cuda.nvtx.range_push(f"f[{fwd_stage_id}]m[{i}]")

            # send input tensor ckpt
            if not forward_only:
                comm_input_ckpt(next(ckpt_send_recv_schedule))

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

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # wait for recv input tensor
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                if recv_handles is not None:
                    for req in recv_handles:
                        req.wait()

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
                    tag=f"compute:f{fwd_stage_id}m{i}",
                )
                if get_thunder_cuda_manager().record_event(compute_microbatch) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatch.tag] = compute_microbatch

                if i == num_microbatches - 1:
                    # notify offload stream to act
                    compute_microbatches_end = get_thunder_cuda_manager().Event(
                        "compute", "offload", tag=f"compute:f{fwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end

            # send output tensor
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:f{fwd_stage_id}m{i}"))
                == -1
            ):
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
                    "offload",
                    "prefetch",
                    tag=f"free:f{fwd_stage_id}",
                    post_wait_fn=lambda fwd_stage_id=fwd_stage_id: _post_wait_set_spiral_param_status(
                        model[fwd_stage_id], SpiralParamStatus.REMOTE
                    ),
                )
                if get_thunder_cuda_manager().record_event(free_curr) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries[free_curr.tag] = free_curr

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
                    offload_event_queries.pop(
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
                    post_wait_fn=lambda bwd_stage_id=bwd_stage_id: _post_wait_set_spiral_param_status(
                        model[-bwd_stage_id], SpiralParamStatus.ACTIVE
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
        for i in range(num_microbatches):
            spiral_print(f" microbatch {i}")
            torch.cuda.nvtx.range_push(f"b[{bwd_stage_id}]m[{i}]")

            # recv input tensor ckpt
            recv_input_ckpts, recv_input_ckpt_handle = comm_input_ckpt(
                next(ckpt_send_recv_schedule)
            )

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

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # wait for recv input tensor ckpt
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                if recv_input_ckpt_handle is not None:
                    for req in recv_input_ckpt_handle:
                        req.wait()
                if recv_input_ckpts is not None:
                    assert (
                        len(recv_input_ckpts) == 1
                    ), f"Only 1 input tensor ckpt expected. Got {len(recv_input_ckpts)}"
                    input_tensor_ckpt = recv_input_ckpts.pop(0)

                # wait for recv output tensor grad
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                if recv_handles is not None:
                    for req in recv_handles:
                        req.wait()

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

                if optimize_after_bwd_stage:
                    # grad_scaler is aligned (ascending) w.r.t bwd_stage_id
                    # (same as optimizer_list in SpiralStageOptimizer)
                    _grad_scaler = grad_scaler[bwd_stage_id]
                else:
                    _grad_scaler = grad_scaler
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
                    tag=f"compute:b{bwd_stage_id}m{i}",
                )
                if get_thunder_cuda_manager().record_event(compute_microbatch) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatch.tag] = compute_microbatch

                if i == num_microbatches - 1:
                    # notify offload stream to act
                    compute_microbatches_end = get_thunder_cuda_manager().Event(
                        "compute", "offload", tag=f"compute:b{bwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end

            # send input tensor grad
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:b{bwd_stage_id}m{i}"))
                == -1
            ):
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
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:b{bwd_stage_id}end"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

            # free bwd stage
            model[-bwd_stage_id - 1].spiral_free()
            free_curr = get_thunder_cuda_manager().Event(
                "offload",
                None if bwd_stage_id == 0 else "prefetch",
                tag=f"free:b{bwd_stage_id}",
                post_wait_fn=lambda bwd_stage_id=bwd_stage_id: _post_wait_set_spiral_param_status(
                    model[-bwd_stage_id - 1], SpiralParamStatus.REMOTE
                ),
            )
            if get_thunder_cuda_manager().record_event(free_curr) == -1:
                raise RuntimeError("record_event failed")
            offload_event_queries[free_curr.tag] = free_curr

            # offload gradient
            model[-bwd_stage_id - 1].spiral_offload_grad(non_blocking=True)
            if optimize_after_bwd_stage:
                offload_grad_curr = get_thunder_cuda_manager().Event(
                    "offload",
                    None,
                    tag=f"optimizer:b{bwd_stage_id}"
                )
                if get_thunder_cuda_manager().record_event(offload_grad_curr) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries[offload_grad_curr.tag] = offload_grad_curr

        # spiral stage optimizer
        if optimize_after_bwd_stage:
            # NOTE (SpiralPipe) multi-processing (either python multiprocessing or torch multiprocessing) does not work since cuda ctx is not shared between processes. So, we use threading.
            _optimizer_thread_status = getattr(optimizer, "optimizer_thread_pool").submit(
                _weight_update,
                *(
                    optimizer,
                    bwd_stage_id,
                    offload_event_queries.pop(f"optimizer:b{bwd_stage_id}"),
                    get_args(),
                    timers,
                ),
            )
            optimizer_threads_status.append(_optimizer_thread_status)

        mpu.set_spiral_backward_virtual_rank(None)
    # end bwd

    # do post-pipeline non-compute timesteps
    assert not forward_only, "Forward only mode should have returned already"
    __num_post_pipeline_non_compute_ts = (
        mpu.get_pipeline_model_parallel_world_size()
        - mpu.get_pipeline_model_parallel_rank()
        - 1
    )
    for _ in range(__num_post_pipeline_non_compute_ts):
        if not forward_only:
            comm_input_ckpt(next(ckpt_send_recv_schedule))

    # join spiral stage optimizer
    if optimize_after_bwd_stage:
        for _optimizer_thread_status in as_completed(optimizer_threads_status):
            try:
                _optimizer_thread_status.result()
            except Exception as e:
                raise Exception(f"Optimizer thread raised an exception: {e}")

    if (
        get_thunder_cuda_manager().wait_event(offload_event_queries.pop(f"free:b0"))
        == -1
    ):
        raise RuntimeError("wait_event failed")

    _cleanup()
    return forward_data_store


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

    optimize_after_bwd_stage = False
    if get_args().spiral_stage_optimizer:
        assert "spiral_stage_optimizer" in kwargs, "spiral_stage_optimizer not found in kwargs"
        optimize_after_bwd_stage = True
        optimizer = kwargs["spiral_stage_optimizer"]
        grad_scaler = kwargs["spiral_grad_scaler"]
        optimizer_threads_status = []

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
        _recompute_data_iterator = [[] for _ in range(len(data_iterator))]

    # Event dictionaries. Key is the event tag, thus an event necessarily requires it.
    prefetch_event_queries = {}
    compute_event_queries = {}
    offload_event_queries = {}

    # Placeholders
    recv_handles = None

    """ Start training """

    # prefetch 1st fwd stage
    with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
        model[0].spiral_fetch(non_blocking=True)
        prefetch_f0 = get_thunder_cuda_manager().Event(
            "prefetch",
            "compute",
            tag="prefetch:f0",
            post_wait_fn=lambda: _post_wait_set_spiral_param_status(
                model[0], SpiralParamStatus.ACTIVE
            ),
        )
        if get_thunder_cuda_manager().record_event(prefetch_f0) == -1:
            raise RuntimeError("record_event failed")
        prefetch_event_queries[prefetch_f0.tag] = prefetch_f0

    # fwd
    for fwd_stage_id in range(mpu.get_spiral_forward_virtual_size()):
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
                    get_thunder_cuda_manager().wait_event(offload_event_queries.pop(f"free:f{fwd_stage_id - 1}"))
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
                    post_wait_fn=lambda fwd_stage_id=fwd_stage_id: _post_wait_set_spiral_param_status(
                        model[fwd_stage_id + 1], SpiralParamStatus.ACTIVE
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

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # NOTE (SpiralPipe) Creating a new iterator with _data results in deep copy of _data. So, calling detach_variable() as in backward() of CheckpointFunction (random.py) is not necessary. However, we must note that a redundant data storage is created here, while a detached tensor shares the underlying storage with the original. https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                # TODO (SpiralPipe) Seek better solution to use detached tensor, using detach_variable().
                _data = next(data_iterator[fwd_stage_id])
                _data_iterator = iter([_data]) # wrap
                if not forward_only and recompute:
                    _recompute_data_iterator[fwd_stage_id].append(_data)

                # wait for recv input tensor
                # NOTE (SpiralPipe) Must be done in compute stream to avoid error
                if recv_handles is not None:
                    for req in recv_handles:
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
                    tag=f"compute:f{fwd_stage_id}m{i}",
                )
                if get_thunder_cuda_manager().record_event(compute_microbatch) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatch.tag] = compute_microbatch

                if i == num_microbatches - 1:
                    # notify offload stream to act
                    compute_microbatches_end = get_thunder_cuda_manager().Event(
                        "compute", "offload", tag=f"compute:f{fwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end

            # send output tensor
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:f{fwd_stage_id}m{i}"))
                == -1
            ):
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
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:f{fwd_stage_id}end"))
                == -1
            ):
                raise RuntimeError("wait_event failed")
            # skip free if processing last fwd stage, as it will be reused as last bwd stage
            if not fwd_stage_id == mpu.get_spiral_forward_virtual_size() - 1:
                model[fwd_stage_id].spiral_free()
                free_curr = get_thunder_cuda_manager().Event(
                    "offload",
                    "prefetch",
                    tag=f"free:f{fwd_stage_id}",
                    post_wait_fn=lambda fwd_stage_id=fwd_stage_id: _post_wait_set_spiral_param_status(
                        model[fwd_stage_id], SpiralParamStatus.REMOTE
                    ),
                )
                if get_thunder_cuda_manager().record_event(free_curr) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries[free_curr.tag] = free_curr

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
        spiral_print(f"Start bwd stage {bwd_stage_id}")
        mpu.set_spiral_backward_virtual_rank(bwd_stage_id)

        assert (
            hasattr(model[bwd_stage_id], "spiral_backward_stage_id")
            and getattr(model[bwd_stage_id], "spiral_backward_stage_id") == bwd_stage_id
        ), "Backward stage ID mismatch between virtual rank and model."

        # prefetch next stage
        with torch.cuda.stream(get_thunder_cuda_manager().Stream("prefetch")):
            # skip offload wait if processing last bwd stage, as last fwd stage doesn't free
            if not bwd_stage_id == mpu.get_spiral_backward_virtual_size() - 1:
                if (
                    get_thunder_cuda_manager().wait_event(offload_event_queries.pop(f"free:b{bwd_stage_id + 1}"))
                    == -1
                ):
                    raise RuntimeError("wait_event failed")

            if not bwd_stage_id == 0:
                model[bwd_stage_id - 1].spiral_fetch(non_blocking=True)
                prefetch_next = get_thunder_cuda_manager().Event(
                    "prefetch",
                    "compute",
                    tag="prefetch:" + f"b{bwd_stage_id - 1}",
                    post_wait_fn=lambda bwd_stage_id=bwd_stage_id: _post_wait_set_spiral_param_status(
                        model[bwd_stage_id - 1], SpiralParamStatus.ACTIVE
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

            # set input tensor ckpt
            input_tensor_ckpt = (
                model[bwd_stage_id].module[0].spiral_input_tensors.popleft()
            )

            # wait for recv output tensor grad
            if recv_handles is not None:
                for req in recv_handles:
                    req.wait()

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                if recompute:
                    with torch.enable_grad():
                        output_tensor = forward_step(
                            forward_step_func,
                            iter(_recompute_data_iterator[bwd_stage_id]),
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
                    tag=f"compute:b{bwd_stage_id}m{i}",
                )
                if get_thunder_cuda_manager().record_event(compute_microbatch) == -1:
                    raise RuntimeError("record_event failed")
                compute_event_queries[compute_microbatch.tag] = compute_microbatch

                if i == num_microbatches - 1:
                    # notify offload stream to act
                    compute_microbatches_end = get_thunder_cuda_manager().Event(
                        "compute", "offload", tag=f"compute:b{bwd_stage_id}end"
                    )
                    if get_thunder_cuda_manager().record_event(compute_microbatches_end) == -1:
                        raise RuntimeError("record_event failed")
                    compute_event_queries[compute_microbatches_end.tag] = compute_microbatches_end

            # send input tensor grad
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:b{bwd_stage_id}m{i}"))
                == -1
            ):
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
            if (
                get_thunder_cuda_manager().wait_event(compute_event_queries.pop(f"compute:b{bwd_stage_id}end"))
                == -1
            ):
                raise RuntimeError("wait_event failed")

            # free bwd stage
            model[bwd_stage_id].spiral_free()
            free_curr = get_thunder_cuda_manager().Event(
                "offload",
                None if bwd_stage_id == 0 else "prefetch",
                tag=f"free:b{bwd_stage_id}",
                post_wait_fn=lambda bwd_stage_id=bwd_stage_id: _post_wait_set_spiral_param_status(
                    model[bwd_stage_id], SpiralParamStatus.REMOTE
                ),
            )
            if get_thunder_cuda_manager().record_event(free_curr) == -1:
                raise RuntimeError("record_event failed")
            offload_event_queries[free_curr.tag] = free_curr

            # offload gradient
            model[bwd_stage_id].spiral_offload_grad(non_blocking=True)
            if optimize_after_bwd_stage:
                offload_grad_curr = get_thunder_cuda_manager().Event(
                    "offload",
                    None,
                    tag=f"optimizer:b{bwd_stage_id}"
                )
                if get_thunder_cuda_manager().record_event(offload_grad_curr) == -1:
                    raise RuntimeError("record_event failed")
                offload_event_queries[offload_grad_curr.tag] = offload_grad_curr

        # spiral stage optimizer
        if optimize_after_bwd_stage:
            # NOTE (SpiralPipe) multi-processing (either python multiprocessing or torch multiprocessing) does not work since cuda ctx is not shared between processes. So, we use threading.
            _optimizer_thread_status = getattr(optimizer, "optimizer_thread_pool").submit(
                _weight_update,
                *(
                    optimizer,
                    bwd_stage_id,
                    offload_event_queries.pop(f"optimizer:b{bwd_stage_id}"),
                    get_args(),
                    timers,
                ),
            )
            optimizer_threads_status.append(_optimizer_thread_status)

        mpu.set_spiral_backward_virtual_rank(None)
    # end bwd

    # join spiral stage optimizer
    if optimize_after_bwd_stage:
        for _optimizer_thread_status in as_completed(optimizer_threads_status):
            try:
                _optimizer_thread_status.result()
            except Exception as e:
                raise Exception(f"Optimizer thread raised an exception: {e}")

    if (
        get_thunder_cuda_manager().wait_event(offload_event_queries.pop(f"free:b0"))
        == -1
    ):
        raise RuntimeError("wait_event failed")

    _cleanup()
    return forward_data_store
