# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import psutil
import os


# For logging to a file
fh = None

# For lazy import with printflock()
fcntl = None

# for debug purposes map module and param objects to their fully qualified names
module_names = {}
param_names = {}


def debug_extract_module_and_param_names(model):
    # extract the fully qualified names as soon as the model is acquired
    global module_names
    global param_names
    # XXX: can probably make a map of param2module and vice-versa
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}


def debug_module2name(module):
    if module in module_names:
        return module_names[module]
    else:
        return "unknown"


def debug_module2name_id(module):
    if hasattr(module, "spiral_forward_stage_id") and hasattr(
        module, "spiral_backward_stage_id"
    ):
        return f"name={debug_module2name(module)} fid={module.spiral_forward_stage_id} bid={module.spiral_backward_stage_id}"
    else:
        return f"name={debug_module2name(module)} id={module.id}"


def debug_module2name_class(module):
    return f"name={debug_module2name(module)} {module.__class__.__name__}"


def debug_module2class_id(module):
    if hasattr(module, "spiral_forward_stage_id") and hasattr(
        module, "spiral_backward_stage_id"
    ):
        return f"class={module.__class__.__name__} fid={module.spiral_forward_stage_id} bid={module.spiral_backward_stage_id}"
    else:
        return f"class={module.__class__.__name__} id={module.id}"


def debug_param2name(param):
    if param in param_names:
        return param_names[param]
    else:
        return "unknown"


def debug_param2name_id(param):
    return f"name={debug_param2name(param)} id={param.spiral_id}"


def debug_param2name_id_shape(param):
    return (
        f"name={debug_param2name(param)} id={param.spiral_id} shape={param.data.shape}"
    )


def debug_param2name_id_shape_device(param):
    return f"name={debug_param2name(param)} id={param.spiral_id} shape={param.data.shape} device={param.device}"


def debug_param2name_id_numel(param):
    return f"name={debug_param2name(param)} id={param.spiral_id} numel={param.numel()}"


def debug_param2name_id_shape_status(param):
    return f"name={debug_param2name(param)} id={param.spiral_id} shape={param.data.shape} status={param.spiral_status}"


def debug_param2id_shape_status(param):
    return f"id={param.spiral_id} gpu_shape={param.data.shape} cpu_shape={param.spiral_tensor.shape} status={param.spiral_status}"


def debug_param2id_numel_dataptr(param):
    msg = f"id={param.spiral_id} numel={param.numel()} spiral_tensor.numel={param.spiral_tensor.numel()}"
    if hasattr(param, "grad") and param.grad is not None:
        msg += f" grad.numel={param.grad.numel()} grad.data_ptr={hex(param.grad.data_ptr())}"
    if hasattr(param, "main_grad") and param.main_grad is not None:
        msg += f" main_grad.numel={param.main_grad.numel()} main_grad.data_ptr={hex(param.main_grad.data_ptr())}"
    if hasattr(param.spiral_tensor, "grad") and param.spiral_tensor.grad is not None:
        msg += f" spiral_tensor.grad.numel={param.spiral_tensor.grad.numel()} spiral_tensor.grad.data_ptr={hex(param.spiral_tensor.grad.data_ptr())}"
    return msg


def printflock(*msgs):
    """

    For printing messages for all concurrent gpus w/o getting interleaved text.

    This is useful when debugging issues where multi-gpus don't sync.

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            printflock(f"[{rank}] {message}")
    3. run the program and you get both logs non-interleaved

    But this makes it very difficult to make sense of the output, so the ``log_rank_file`` helper
    function might be more useful, as it's easier to send each log stream into a separate file and
    then compare those.

    """
    global fcntl
    if fcntl is None:
        import fcntl

    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


fh = None


def log_rank_file(rank, *msgs):
    """
    Print to a log file of the given rank

    This is useful for debugging hanging in sync processes. Here is a possible workflow:

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions of print_rank_0 in those files with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            log_rank_file(rank, message)

    3. run the program
    4. fix up the expected differences, e.g. different cuda numbers ::

        perl -pi -e 's|cuda:1|cuda:0|' log_rank_*

    5. now diff and see where names and ids diverge - you will find where the gpus don't do the same
    work (e.g. when some layers get conditionally skipped on one gpu but not all)

        diff -u log_rank_0.txt log_rank_1.txt | less

    """
    global fh
    if fh is None:
        fh = open(f"log_rank_{rank}.txt", "w")
    for m in msgs:
        fh.write(f"{m}\n")
    fh.flush()


def print_backward_tensors(tensor):

    def _print_bwd_tensors(grad_fn):
        print(f"Backward tensors in {grad_fn}")
        for funcs in grad_fn.next_functions:
            if funcs[0]:
                try:
                    tensor = getattr(funcs[0], "variable")
                    print(funcs[0])
                    print(
                        f"Tensor - id: {id(tensor)}, shape: {tensor.shape}, data: {tensor}, grad: {tensor.grad}"
                    )
                except AttributeError as e:
                    _print_bwd_tensors(funcs[0])

    if hasattr(tensor, "grad_fn"):
        _print_bwd_tensors(tensor.grad_fn)


def spiral_print(message):
    _DEBUG_PID = True

    if torch.distributed.is_initialized():
        prefix = f"[Spiral] " + f"[{torch.distributed.get_rank()}] "
        if _DEBUG_PID:
            prefix += f"[pid={os.getpid()}] "
        message = prefix + message
    else:
        prefix = f"[Spiral] "
        if _DEBUG_PID:
            prefix += f"[pid={os.getpid()}] "
        message = prefix + message
    print(message)

    if torch.distributed.is_initialized():
        global fh
        if fh is None:
            fh = open(f"log_rank_{torch.distributed.get_rank()}.txt", "w")
        fh.write(message + "\n")
        fh.flush()


def spiral_report_memory(message, gpu=True, cpu=True):
    """SpiralPipe memory report"""
    giga_bytes = 1024**3

    string = message
    if gpu:
        string += " | GPU: {} GB".format(
            round(torch.cuda.memory_allocated() / giga_bytes, 2)
        )
        string += " | MAX_GPU: {} GB".format(
            round(torch.cuda.max_memory_allocated() / giga_bytes, 2)
        )
    if cpu:
        rss_used = psutil.Process(os.getpid()).memory_info().rss
        string += " | CPU: {} GB".format(round(rss_used / giga_bytes, 2))
    spiral_print(string)
