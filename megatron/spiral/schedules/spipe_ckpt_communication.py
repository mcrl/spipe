import nvtx
from typing import List, Union

import torch
import torch.distributed

import megatron.spiral.build_state as sbs
from megatron.core import mpu
from megatron.spiral.build_state import bwd_phase2local_stage_phase
from megatron.spiral.debug import spiral_print
from .spipe_ckpt_schedule import CkptSendRecvType
from megatron.spiral.initialize import get_thunder_cuda_manager


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

        if not op.comm_type == CkptSendRecvType.SDRV:
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
                input_ckpt_ = (
                    model[local_stage_id]
                    .module[local_phase_id]
                    .spiral_input_tensors.popleft()
                    .detach()
                    .requires_grad_()
                )

                assert (
                    input_ckpt_.requires_grad
                ), "Input ckpt must require grad before feeding to BWD"
                recvs.append(input_ckpt_)
                reqs.append(NOP_Wait)

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
                    # torch.distributed.irecv(
                    #     et, src=src, group=mpu.get_spiral_input_tensor_ckpt_group()
                    # )

                    # TODO: Junyeol temp code
                    torch.distributed.irecv(
                        et, src=src, group=mpu.get_spiral_input_tensor_ckpt_groups(src)
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
                    spiral_print(_prefix + " Send to other rank => pop ckpt")
                input_ckpt_ = (
                    model[local_stage_id]
                    .module[local_phase_id]
                    .spiral_input_tensors.popleft()
                )
                dst = (
                    mpu.translate_pp_rank_to_cm_rank(op.rank)
                    if mpu.is_spiral_cross_mapping()
                    else op.rank
                )
                reqs.append(
                    # torch.distributed.isend(
                    #     input_ckpt_, dst, group=mpu.get_spiral_input_tensor_ckpt_group()
                    # )

                    # TODO: Junyeol temp code
                    torch.distributed.isend(
                        input_ckpt_, dst, group=mpu.get_spiral_input_tensor_ckpt_groups(dst)
                    )
                )

        elif op.comm_type == CkptSendRecvType.SDRV:
            sd_phase_fwd_rank = sbs.get_pp_rank_for_fwd_phase(op.phase_ids[0])
            sd_local_stage_id, sd_local_phase_id = sbs.fwd_phase2local_stage_phase(op.phase_ids[0])

            rv_phase_fwd_rank = sbs.get_pp_rank_for_fwd_phase(op.phase_ids[1])
            rv_local_stage_id, rv_local_phase_id = sbs.fwd_phase2local_stage_phase(op.phase_ids[1])

            input_ckpt_ = (
                model[sd_local_stage_id]
                .module[sd_local_phase_id]
                .spiral_input_tensors.popleft()
            )
            dst = (
                mpu.translate_pp_rank_to_cm_rank(op.rank)
                if mpu.is_spiral_cross_mapping()
                else op.rank
            )
            et = _get_empty_tensor(tensor_shape, dtype)
            src = dst
            sd = torch.distributed.P2POp(torch.distributed.isend, input_ckpt_, dst, group=mpu.get_spiral_input_tensor_ckpt_groups(dst))
            rv = torch.distributed.P2POp(torch.distributed.irecv, et, src, group=mpu.get_spiral_input_tensor_ckpt_groups(src))
            _req = torch.distributed.batch_isend_irecv([sd, rv])
            reqs.append(_req)
            recvs.append(et)

        else:
            raise RuntimeError(f"Invalid comm type {op.comm_type}")

    if recvs and len(recvs) > 0:
        # zip only recv with corresponding req from same op => append to ckpt_recvs
        for recv, (req, op) in zip(
            recvs,
            filter(
                lambda x: x[1].comm_type == CkptSendRecvType.RECV or x[1].comm_type == CkptSendRecvType.SDRV,
                zip(reqs, schedule),
            ),
        ):
            if op.comm_type == CkptSendRecvType.RECV:
                bid, _ = bwd_phase2local_stage_phase(op.phase_id)
            elif op.comm_type == CkptSendRecvType.SDRV:
                bid, _ = bwd_phase2local_stage_phase(op.phase_ids[1])
            ckpt_recvs[bid].append((recv, [req]))
