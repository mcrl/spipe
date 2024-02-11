import contextlib
import warnings
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_type
from megatron.core.pipeline_parallel import forward_step, backward_step
from megatron.spiral.debug import spiral_print
import megatron.spiral.p2p_communication as spiral_p2p


# Types
Shape = Union[List[int], torch.Size]


def forward_backward_pipelining_with_spiral(*,
                                            forward_step_func,
                                            data_iterator: Union[Iterator, List[Iterator]],
                                            model: Union[torch.nn.Module, List[torch.nn.Module]],
                                            num_microbatches: int,
                                            dtype: torch.dtype,
                                            tensor_shape: Shape,
                                            decoder_seq_length: Optional[int] = None,
                                            grad_scaler: Callable = None,
                                            sequence_parallel: bool = False,
                                            overlap_p2p_comm: bool = False, # TODO (mcrl) check
                                            batch_p2p_comm: bool = False, # TODO (mcrl) check
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
    
    assert isinstance(model, list), \
        "Spiral pipeline parallelism expected model chunking by stage"
    assert isinstance(data_iterator, list), \
        "Spiral pipeline parallelism expected each model chunk to have a data iterator"

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
        warnings.warn("Spiral pipeline parallel forward virtual rank is not None on scheule entry. There may be a bug.")
    if mpu.get_spiral_pipeline_parallel_backward_virtual_rank() is not None:
        warnings.warn("Spiral pipeline parallel backward virtual rank is not None on scheule entry. There may be a bug.")

    if sequence_parallel:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // mpu.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError(
            "Spiral is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError(
            "Spiral is not supported with a different decoder sequence length.")\
    
    # Start training
    forward_data_store = []

    # input ckpts
    if not forward_only:
        input_tensor_ckpts = [[] for _ in range(mpu.get_spiral_pipeline_parallel_backward_virtual_size())]

    fwd_wait_handles = None
    bwd_wait_handles = None

    # fwd
    for fwd_stage_id in range(mpu.get_spiral_pipeline_parallel_forward_virtual_size()):
        spiral_print(f"Start fwd stage {fwd_stage_id}")
        mpu.set_spiral_pipeline_parallel_forward_virtual_rank(fwd_stage_id)
        # input_tensor_ckpt_dst = mpu.get_pipeline_model_parallel_world_size() - mpu.get_pipeline_model_parallel_rank() - 1 # TODO (mcrl) can be moved out of fwd for loop

        # fetch fwd stage
        # TODO (mcrl) on-demand fetch currently
        assert hasattr(model[fwd_stage_id], "spiral_forward_stage_id") and getattr(model[fwd_stage_id], "spiral_forward_stage_id") == fwd_stage_id, \
            "Forward stage ID mismatch between virtual rank and model."
        model[fwd_stage_id].spiral_fetch(async_op=False)

        # fwd microbatches
        for i in range(num_microbatches):
            spiral_print(f" microbatch {i}")
            torch.cuda.nvtx.range_push(f'f[{fwd_stage_id}]m[{i}]')

            # set input tensor
            if mpu.is_pipeline_first_stage():
                input_tensor = None
            else:
                input_tensor, fwd_wait_handles = spiral_p2p.recv_input_tensor(tensor_shape, 
                                                            dtype, 
                                                            batch_p2p_comm=batch_p2p_comm,
                                                            overlap_p2p_comm=overlap_p2p_comm,
                                                            timers=timers)

            # wait for recv input tensor
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            # input ckpt
            # p2p_communication.send_ckpt(input_tensor, input_tensor_ckpt_dst, timers=timers)
                    
            output_tensor = forward_step(forward_step_func,
                                        data_iterator[fwd_stage_id],
                                        model[fwd_stage_id],
                                        num_microbatches,
                                        input_tensor,
                                        forward_data_store,
                                        timers,
                                        collect_non_loss_data,
                                        dtype,
                                        enable_autocast)
            
            # send output tensor
            if not mpu.is_pipeline_last_stage():
                fwd_wait_handles = spiral_p2p.send_output_tensor(output_tensor, batch_p2p_comm=False, timers=timers)

            # wait for send output tensor
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            torch.cuda.nvtx.range_pop()
        # end fwd microbatches
            
        model[fwd_stage_id].spiral_free()
        mpu.set_spiral_pipeline_parallel_forward_virtual_rank(None)

    if forward_only:
        return forward_data_store
    # end fwd
    
    # bwd
    for bwd_stage_id in range(mpu.get_spiral_pipeline_parallel_backward_virtual_size() - 1, -1, -1):
        spiral_print(f"Start bwd stage {bwd_stage_id}")
        mpu.set_spiral_pipeline_parallel_backward_virtual_rank(bwd_stage_id)

        # fetch bwd stage
        # TODO (mcrl) on-demand fetch currently
        assert hasattr(model[-bwd_stage_id-1], "spiral_backward_stage_id") and getattr(model[-bwd_stage_id-1], "spiral_backward_stage_id") == bwd_stage_id, \
            "Backward stage ID mismatch between virtual rank and model."
        model[-bwd_stage_id-1].spiral_fetch(async_op=False)

        # NOTE (mcrl) temporary code
        # TODO (mcrl) remove after receive ckpts
        input_tensor_ckpt = torch.ones(tensor_shape, dtype=torch.float, device=torch.cuda.current_device(), requires_grad=True) if not mpu.is_pipeline_first_stage() else None

        if input_tensor_ckpt == None:
            spiral_print(f"input_tensor_ckpt is None ; 1st stage {mpu.is_pipeline_first_stage()}")
        else:
            spiral_print(f"input_tensor_ckpt is not None")

        # bwd microbatches
        for i in range(num_microbatches):
            spiral_print(f" microbatch {i}")
            torch.cuda.nvtx.range_push(f'b[{bwd_stage_id}]m[{i}]')

            # set output tensor grad
            if mpu.is_pipeline_last_stage():
                output_tensor_grad = None
            else:
                output_tensor_grad, bwd_wait_handles = spiral_p2p.recv_output_tensor_grad(tensor_shape, 
                                                            dtype, 
                                                            batch_p2p_comm=batch_p2p_comm,
                                                            overlap_p2p_comm=overlap_p2p_comm,
                                                            timers=timers)
            
            # wait for recv output tensor grad
            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            output_tensor = forward_step(forward_step_func,
                                         data_iterator[bwd_stage_id], # TODO (mcrl) problematic, since this assumes forward virtual size == backward virtual size, as len == forward virtual size
                                         model[-bwd_stage_id-1],
                                         num_microbatches,
                                         input_tensor_ckpt,
                                         [],
                                         timers,
                                         collect_non_loss_data,
                                         dtype,
                                         enable_autocast)
            
            input_tensor_grad = backward_step(grad_scaler,
                                              input_tensor_ckpt,
                                              output_tensor,
                                              output_tensor_grad,
                                              model_type,
                                              timers,
                                              deallocate_pipeline_outputs)
            
            # send input tensor grad
            if not mpu.is_pipeline_first_stage():
                bwd_wait_handles = spiral_p2p.send_input_tensor_grad(input_tensor_grad,
                                                                     overlap_p2p_comm=overlap_p2p_comm,
                                                                     batch_p2p_comm=batch_p2p_comm,
                                                                     timers=timers)
            
            # wait for send input tensor grad
            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            torch.cuda.nvtx.range_pop()
        # end bwd microbatches

        model[-bwd_stage_id-1].spiral_offload_grad()
        model[-bwd_stage_id-1].spiral_free()
        mpu.set_spiral_pipeline_parallel_backward_virtual_rank(None)
    # end bwd
    
    return forward_data_store