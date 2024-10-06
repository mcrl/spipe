import nvtx
from typing import List, Union

import torch

import megatron.spiral.build_state as sbs
import megatron.spiral.p2p_communication as spiral_p2p
from megatron.core import mpu
from megatron.spiral.build_state import bwd_phase2local_stage_phase
from megatron.spiral.debug import spiral_print
from .spipe_ckpt_schedule import CkptSendRecvType


# Types
Shape = Union[List[int], torch.Size]

# Constants
_DEBUG_CKPT_COMMUNICATION = True


# Handle for self send/recv
class NOP_Wait:
    @staticmethod
    def wait():
        pass


def _get_empty_tensor(tensor_shape: Shape, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(
        tensor_shape,
        requires_grad=True,
        device=torch.cuda.current_device(),
        dtype=dtype,
    )


@nvtx.annotate("comm_ckpt", color="darkgreen")
def comm_ckpt(schedule, model, ckpt_recvs, tensor_shape: Shape, dtype: torch.dtype):
    if _DEBUG_CKPT_COMMUNICATION:
        spiral_print(f"comm: {schedule}")

    recvs, reqs = [], []

    for idx, op in enumerate(schedule):

        phase_fwd_rank = sbs.get_pp_rank_for_fwd_phase(op.phase_id)
        local_stage_id, local_phase_id = sbs.fwd_phase2local_stage_phase(op.phase_id)

        _prefix = str(schedule[idx])

        if op.comm_type == CkptSendRecvType.RECV:
            assert (
                phase_fwd_rank == op.rank
            ), f"[RECV] phase_fwd_rank = {phase_fwd_rank}, rank = {op.rank} mismatch"

            if op.phase_id == 0:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(_prefix + " Phase 0 => insert None to recvs")
                recvs.append(None)
                reqs.append(NOP_Wait)
            elif op.rank == mpu.get_pipeline_model_parallel_rank():
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(_prefix + " Self recv => pop ckpt and insert to recvs")
                # NOTE (SpiralPipe) Using the popped input tensor from original fwd stage can lead to trouble, as it contains
                # the computation graph constructed already. We are prone to this error when #fwd != #bwd and hence the same rank
                # can perform fwd and bwd of the same phase. Re-computation using this tensor will lead to duplicated computation
                # graph being constructed. So, we currently perform the original FWD in torch.no_grad() mode, and then recompute
                # in BWD without torch.no_grad(). Another solution may exist.
                # input_ckpt_ = (
                #     model[local_stage_id]
                #     .module[local_phase_id]
                #     .spiral_input_tensors.popleft()
                #     .detach()
                #     .requires_grad_()
                # )

                ### TEMP
                input_ckpt_ = torch.randn(
                    tensor_shape,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=True,
                )
                ###
                assert (
                    input_ckpt_.requires_grad
                ), "Input ckpt must require grad before feeding to BWD"
                recvs.append(input_ckpt_)
                reqs.append(NOP_Wait)

                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(f"  ckptout = {torch.mean(input_ckpt_)}")
            else:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(
                        _prefix + " Recv from other rank => append to recvs"
                    )
                et = _get_empty_tensor(tensor_shape, dtype)
                src = (
                    mpu.translate_pp_rank_to_cm_rank(op.rank)
                    if mpu.is_spiral_cross_mapping()
                    else op.rank
                )
                reqs.append(
                    torch.distributed.irecv(
                        et, src=src, group=mpu.get_spiral_input_tensor_ckpt_group()
                    )
                )
                recvs.append(et)

        elif op.comm_type == CkptSendRecvType.SEND:
            assert (
                phase_fwd_rank == mpu.get_pipeline_model_parallel_rank()
            ), f"[SEND] phase_fwd_rank = {phase_fwd_rank}, self = {mpu.get_pipeline_model_parallel_rank()} mismatch"

            if op.phase_id == 0:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(_prefix + " Phase 0 => skip")
                reqs.append(NOP_Wait)
            elif op.rank == mpu.get_pipeline_model_parallel_rank():
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(_prefix + " Self send => skip")
                reqs.append(NOP_Wait)
            else:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(
                        _prefix
                        + " Send to other rank => pop ckpt"
                    )
                # tensor_sends.append(
                #     model[local_stage_id]
                #     .module[local_phase_id]
                #     .spiral_input_tensors.popleft()
                # )

                ### TEMP
                t = torch.randn(
                    tensor_shape, dtype=dtype, device=torch.cuda.current_device()
                )
                dst = (
                    mpu.translate_pp_rank_to_cm_rank(op.rank)
                    if mpu.is_spiral_cross_mapping()
                    else op.rank
                )
                reqs.append(
                    torch.distributed.isend(
                        t, dst, group=mpu.get_spiral_input_tensor_ckpt_group()
                    )
                )
                ###
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(f"  ckptout = {torch.mean(t)}")
        else:
            raise RuntimeError(f"Invalid comm type {op.comm_type}")

    if recvs and len(recvs) > 0:
        for recv, (req, op) in zip(
            recvs,
            filter(
                lambda x: x[1].comm_type == CkptSendRecvType.RECV,
                zip(reqs, schedule),
            ),
        ):
            bid, _ = bwd_phase2local_stage_phase(op.phase_id)
            ckpt_recvs[bid].append((recv, [req]))
