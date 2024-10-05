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


# TODO (SpiralPipe) Move to spiral_p2p
@nvtx.annotate("comm_ckpt", color="darkgreen")
def comm_ckpt(schedule, model, ckpt_recvs, tensor_shape: Shape, dtype: torch.dtype):
    if _DEBUG_CKPT_COMMUNICATION:
        spiral_print(f"comm: {schedule}")

    tensor_sends = []
    send_ranks = []
    recv_ranks = []

    insert_idx_to_recvs = []
    insert_value_to_recvs = []

    recv_idx = 0
    send_idx = 0
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
                insert_idx_to_recvs.append(recv_idx)
                insert_value_to_recvs.append(None)
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

                ### TEMP
                # input_ckpt_ = torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)
                ###
                assert (
                    input_ckpt_.requires_grad
                ), "Input ckpt must require grad before feeding to BWD"
                insert_idx_to_recvs.append(recv_idx)
                insert_value_to_recvs.append(input_ckpt_)
            else:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(
                        _prefix + " Recv from other rank => append to recv_ranks"
                    )
                recv_ranks.append(op.rank)
            recv_idx += 1

        elif op.comm_type == CkptSendRecvType.SEND:
            assert (
                phase_fwd_rank == mpu.get_pipeline_model_parallel_rank()
            ), f"[SEND] phase_fwd_rank = {phase_fwd_rank}, self = {mpu.get_pipeline_model_parallel_rank()} mismatch"

            if op.phase_id == 0:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(_prefix + " Phase 0 => skip")
            elif op.rank == mpu.get_pipeline_model_parallel_rank():
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(_prefix + " Self send => skip")
            else:
                if _DEBUG_CKPT_COMMUNICATION:
                    spiral_print(
                        _prefix
                        + " Send to other rank => pop ckpt & append to tensor sends and append to send_ranks"
                    )
                tensor_sends.append(
                    model[local_stage_id]
                    .module[local_phase_id]
                    .spiral_input_tensors.popleft()
                )

                ### TEMP
                # tensor_sends.append(torch.randn(tensor_shape, dtype=dtype, device=torch.cuda.current_device()))
                ###

                send_ranks.append(op.rank)
            send_idx += 1
        else:
            raise RuntimeError(f"Invalid comm type {op.comm_type}")

    recvs, reqs = None, [] # placeholder
    if len(send_ranks) > 0 or len(recv_ranks) > 0:
        if mpu.is_spiral_cross_mapping():
            # translate cm_rank back to pp_rank before communication
            send_ranks = [mpu.translate_cm_rank_to_pp_rank(rank) for rank in send_ranks]
            recv_ranks = [mpu.translate_cm_rank_to_pp_rank(rank) for rank in recv_ranks]

        recvs, reqs = spiral_p2p._communicate(
            tensor_sends=tensor_sends if len(tensor_sends) > 0 else None,
            send_ranks=send_ranks if len(send_ranks) > 0 else None,
            recv_ranks=recv_ranks if len(recv_ranks) > 0 else None,
            tensor_shape=tensor_shape,
            group=mpu.get_spiral_input_tensor_ckpt_group(),
            batch_p2p_comm=False,
            wait_on_reqs=False,
            dtype=dtype,
        )
        reqs = reqs[len(send_ranks):] if mpu.get_pipeline_model_parallel_rank() % 2 == 0 else reqs[:len(recv_ranks)]

    for recv_idx, recv_val in zip(insert_idx_to_recvs, insert_value_to_recvs):
        if recvs is None:
            recvs = []
        recvs.insert(recv_idx, recv_val)
        reqs.insert(recv_idx, NOP_Wait)

    if recvs and len(recvs) > 0:
        for _recv, _req, _op in zip(recvs, reqs, filter(lambda op: op.comm_type == CkptSendRecvType.RECV, schedule)):
            bid, _ = bwd_phase2local_stage_phase(_op.phase_id)
            ckpt_recvs[bid].append((_recv, [_req]))

    if _DEBUG_CKPT_COMMUNICATION:
        spiral_print(f"comm: {schedule} => done")