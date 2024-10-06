import sys
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass

from megatron import get_num_microbatches
from megatron.core import mpu
import megatron.spiral.build_state as sbs


# Constants
_USE_ASYNC_CKPT_SDRV = True


class CkptSendRecvType(Enum):
    SEND = "send"
    RECV = "recv"


@dataclass
class CkptSendRecvOp:
    comm_type: CkptSendRecvType
    phase_id: int
    rank: int


class CkptSendRecvScheduleMeta(type):
    """CkptSendRecvSchedule cache with num_microbatches as key."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        num_microbatches = kwargs.get("num_microbatches", get_num_microbatches())
        if num_microbatches not in cls._instances:
            cls._instances[num_microbatches] = super().__call__(*args, **kwargs)
        _ins = cls._instances[num_microbatches]
        setattr(_ins, "_schedule_generator", _ins._generator())
        return _ins


class CkptSendRecvSchedule(metaclass=CkptSendRecvScheduleMeta):
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

    def __init__(self, num_microbatches = None):
        assert (
            num_microbatches >= mpu.get_pipeline_model_parallel_world_size()
        ), "CkptSendRecvSchedule requires num_microbatches >= pipeline model parallel world size"
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

        if _USE_ASYNC_CKPT_SDRV:
            # recv early but wait at the exact timestep to overlap transmission
            self._set_send_schedule_async()
            self._set_recv_schedule_async()
            self._optimize_schedule()
            self._resolve_p2p_hang()
        else:
            # recv and wait at the exact timestep for simplicity
            self._set_recv_schedule()
            self._set_send_schedule()

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
                    _op = CkptSendRecvOp(
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
                    lambda x: x.comm_type == CkptSendRecvType.RECV,
                    self.global_schedule[pp_rank][j],
                ):
                    # add send to dst for the sender rank queue
                    _src_rank = recv_comm.rank
                    _op = CkptSendRecvOp(
                        CkptSendRecvType.SEND, recv_comm.phase_id, pp_rank
                    )
                    self.global_schedule[_src_rank][j].append(_op)

    def _set_recv_schedule_async(self):
        # TODO (SpiralPipe) recv of only self is required
        for j in range(self.total_ts):
            for pp_rank in range(mpu.get_pipeline_model_parallel_world_size()):
                for send_comm in filter(
                    lambda x: x.comm_type == CkptSendRecvType.SEND,
                    self.global_schedule[pp_rank][j],
                ):
                    # add recv from src for the receiver rank queue
                    _dst_rank = send_comm.rank
                    _op = CkptSendRecvOp(
                        CkptSendRecvType.RECV, send_comm.phase_id, pp_rank
                    )
                    self.global_schedule[_dst_rank][j].append(_op)

    def _set_send_schedule_async(self):
        for pp_rank in range(mpu.get_pipeline_model_parallel_world_size()):
            num_pre_pipeline_non_compute_ts = pp_rank
            send_start_ts = num_pre_pipeline_non_compute_ts
            curr_ts = send_start_ts

            for fwd_stage_id in range(mpu.get_spiral_forward_virtual_size()):
                fwd_phases_start = (
                    sbs.get_spiral_forward_stage_build_phase_size()
                    * mpu.get_pipeline_model_parallel_world_size()
                    * fwd_stage_id
                    + sbs.get_spiral_forward_stage_build_phase_size() * pp_rank
                )
                for _ in range(self.num_microbatches):
                    for send_phase in range(
                        fwd_phases_start,
                        fwd_phases_start
                        + sbs.get_spiral_forward_stage_build_phase_size(),
                    ):
                        _op = CkptSendRecvOp(
                            CkptSendRecvType.SEND,
                            send_phase,
                            sbs.get_pp_rank_for_bwd_phase(send_phase),
                        )
                        self.global_schedule[pp_rank][curr_ts].append(_op)
                    # end for fwd phase
                    curr_ts += 1
                # end for microbatch

    def _optimize_schedule(self):
        """Optimize the schedule by dropping unnecessary send/recv ops.

        e.g., if bwd stage i = [bp2, bp1, bp0], only send/recv ops for bp2 are required.
        """
        self.global_schedule = [
            [
                [op for op in ts_schedule if (op.phase_id + 1) % sbs.get_spiral_backward_stage_build_phase_size() == 0]
                for ts_schedule in pprank_schedule
            ]
            for pprank_schedule in self.global_schedule
        ]

    def _resolve_p2p_hang(self):
        """Resolve P2P hang (when using p2p_ops) by jointly sorting timestep schedule of ranks by isend-irecv pairs.

        e.g., timestep 5 of gpu=3, #f=2, #b=3, #micro=3
        Before) Hangs since irecvs of corresponding isends are not called
          rank 0                      rank 1                      rank 2
            isend->rank 1(phase 9)      isend->rank 2(phase 13)     isend->rank 1(phase 15)
            irecv<-rank 2(phase 17)     irecv<-rank 0(phase 9)      isend->rank 0(phase 17)
                                        irecv<-rank 2(phase 15)     irecv<-rank 1(phase 13)
        After) Hangs resolved
          rank 0                      rank 1                      rank 2
            isend->rank 1(phase 9)      irecv<-rank 0(phase 9)
                                        isend->rank 2(phase 13)     irecv<-rank 1(phase 13)
                                        irecv<-rank 2(phase 15)     isend->rank 1(phase 15)
            irecv<-rank 2(phase 17)                                 isend->rank 0(phase 17)
        """
        _unoptimized_schedule = self.global_schedule
        self.global_schedule = [
            [[] for _ in range(self.total_ts)]
            for _ in range(mpu.get_pipeline_model_parallel_world_size())
        ]

        for ts, ts_schedules in enumerate(zip(*_unoptimized_schedule)):
            _merged = [
                (op, pp_rank)
                for pp_rank, ts_schedule in enumerate(ts_schedules)
                for op in ts_schedule
            ]
            _merged.sort(key=lambda tup: tup[0].phase_id)
            for op, pp_rank in _merged:
                self.global_schedule[pp_rank][ts].append(op)

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
