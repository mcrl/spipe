import contextlib
import warnings
from typing import Callable, Iterator, List, Optional, Union, Tuple
from enum import Enum
import nvtx
import sys

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_num_microbatches
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type, get_attr_wrapped_model
from megatron.core.pipeline_parallel import forward_step, backward_step
from megatron.spiral import get_thunder_cuda_manager
from megatron.spiral.debug import spiral_print
import megatron.spiral.p2p_communication as spiral_p2p
from megatron.spiral.init_context import SpiralParamStatus
from megatron.spiral.utils import is_spiral_param
import megatron.spiral.build_state as sbs


# Due to Tuple() ctor support from python >= 3.9
_PYTHON_VERSION = sys.version_info

# Types
Shape = Union[List[int], torch.Size]

class CkptSendRecvType(Enum):
    SEND = "send"
    RECV = "recv"
CkptSendRecvOp = Tuple[CkptSendRecvType, int, int] # (comm_type, phase_id, rank)


class CkptSendRecvSchedule:
    """A schedule for sending and receiving input tensor checkpoints.

    Construct global schedule for all timestep of all rank.
    A timestep is a time unit for a micro-batch fwd/bwd computation.
    All timestep means from beginning of the minibatch on PP rank 0 to the end of the minibatch on PP rank N-1.
    Thus, a rank's schedule includes non_compute timesteps where no fwd/bwd computation is performed due to pipeline fill/drain.
    non_compute timesteps are blank timesteps in below example

    Example:
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE = 3
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE = 3
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
            mpu.get_spiral_pipeline_parallel_forward_virtual_size()
            + mpu.get_spiral_pipeline_parallel_backward_virtual_size()
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
                + mpu.get_spiral_pipeline_parallel_forward_virtual_size()
                * self.num_microbatches
            )
            curr_ts = recv_start_ts

            for bwd_stage_id in range(
                mpu.get_spiral_pipeline_parallel_backward_virtual_size() - 1, -1, -1
            ):
                bwd_phases_start = (
                    sbs.get_spiral_pipeline_parallel_backward_stage_build_phase_size()
                    * mpu.get_pipeline_model_parallel_world_size()
                    * bwd_stage_id
                    + sbs.get_spiral_pipeline_parallel_backward_stage_build_phase_size()
                    * (mpu.get_pipeline_model_parallel_world_size() - pp_rank - 1)
                )
                recv_phase = bwd_phases_start  # modify when needed
                for _ in range(self.num_microbatches):
                    if _PYTHON_VERSION >= (3, 9):
                        _op = CkptSendRecvOp(CkptSendRecvType.RECV, recv_phase, sbs.get_pp_rank_for_fwd_phase(recv_phase))
                    else:
                        _op = (CkptSendRecvType.RECV, recv_phase, sbs.get_pp_rank_for_fwd_phase(recv_phase))
                    self.global_schedule[pp_rank][curr_ts].append(_op)
                    curr_ts += 1

    def _set_send_schedule(self):
        # TODO (mcrl) sends of only self is required
        for j in range(self.total_ts):
            for pp_rank in range(mpu.get_pipeline_model_parallel_world_size()):
                for recv_comm in filter(
                    lambda x: x[0] == CkptSendRecvType.RECV, self.global_schedule[pp_rank][j]
                ):
                    # add send to dst for the sender rank queue
                    _src_rank = recv_comm[2]
                    if _PYTHON_VERSION >= (3, 9):
                        _op = CkptSendRecvOp(CkptSendRecvType.SEND, recv_comm[1], pp_rank)
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

    # TODO (mcrl): Move to spiral_p2p
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
                    # NOTE (mcrl) Using the popped input tensor from original fwd stage can lead to trouble, as it contains
                    # the computation graph constructed already. We are prone to this error when #fwd != #bwd and hence the same rank
                    # can perform fwd and bwd of the same phase. Re-computation using this tensor will lead to duplicated computation
                    # graph being constructed. So, we currently perform the original FWD in torch.no_grad() mode, and then recompute
                    # in BWD without torch.no_grad(). Another solution may exist.
                    input_ckpt_ = (
                        model[local_stage_id]
                        .module[local_phase_id]
                        .spiral_input_tensor_ckpts.popleft()
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
                        .spiral_input_tensor_ckpts.popleft()
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

    def cleanup():
        # cleanup input tensor ckpts
        for module in model:
            empty_input_tensor_ckpts: Callable = get_attr_wrapped_model(
                module, "empty_input_tensor_ckpts"
            )
            empty_input_tensor_ckpts()

    # Init input ckpt send recv schedule
    ckpt_send_recv_schedule = None  # placeholder
    if not forward_only:
        ckpt_send_recv_schedule = CkptSendRecvSchedule(num_microbatches)

    # Data structures for training
    forward_data_store = []

    recv_handles = None

    prefetch_event_queries = []
    compute_event_queries = []
    offload_event_queries = []

    prefetch_query = None
    compute_query = None
    offload_query = None

    """ Start training """

    # TODO (mcrl) below line is temporarily added to sync between minibatches. Remove after optimizer sync is implemented
    # torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())

    # nop pre-pipeline non-compute timesteps
    __num_pre_pipeline_non_compute_ts = mpu.get_pipeline_model_parallel_rank()
    for _ in range(__num_pre_pipeline_non_compute_ts):
        next(ckpt_send_recv_schedule)

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

            # wait for recv input tensor
            if recv_handles is not None:
                for req in recv_handles:
                    req.wait()

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                # NOTE (mcrl) Original FWD should not create computation graph for two reasons
                # 1. It will be recomputed in BWD
                # 2. It creates computation graph to the input ckpt tensor, which leads to computation graph duplication
                #   when BWD is performed in the same rank. Symptoms include size error due to ops with size 0 tensor.
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
        cleanup()
        return forward_data_store
    # end fwd

    # bwd
    assert not forward_only, "Forward only mode should have returned already"
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

            # wait for recv input tensor ckpt
            if recv_input_ckpt_handle is not None:
                for req in recv_input_ckpt_handle:
                    req.wait()
            if recv_input_ckpts is not None:
                assert (
                    len(recv_input_ckpts) == 1
                ), f"Only 1 input tensor ckpt expected. Got {len(recv_input_ckpts)}"
                input_tensor_ckpt = recv_input_ckpts.pop(0)

            # wait for recv output tensor grad
            if recv_handles is not None:
                for req in recv_handles:
                    req.wait()

            with torch.cuda.stream(get_thunder_cuda_manager().Stream("compute")):
                output_tensor = forward_step(
                    forward_step_func,
                    data_iterator[
                        bwd_stage_id
                        + mpu.get_spiral_pipeline_parallel_forward_virtual_size()
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

    # do post-pipeline non-compute timesteps
    assert not forward_only, "Forward only mode should have returned already"
    __num_post_pipeline_non_compute_ts = (
        mpu.get_pipeline_model_parallel_world_size()
        - mpu.get_pipeline_model_parallel_rank()
        - 1
    )
    for _ in range(__num_post_pipeline_non_compute_ts):
        comm_input_ckpt(next(ckpt_send_recv_schedule))

    cleanup()
    return forward_data_store


def _post_wait_set_spiral_param_status(module, status: SpiralParamStatus):
    for param in module.parameters(recurse=True):
        if is_spiral_param(param):
            param.spiral_param_status = status
