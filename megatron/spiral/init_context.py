# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# Code modified from DeepSpeed

from typing import Callable, Iterable
from enum import Enum
import functools
from contextlib import contextmanager

import nvtx
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from megatron import get_args
from megatron.spiral import get_thunder_group
from megatron.spiral.debug import (
    spiral_print,
    spiral_report_memory,
    debug_module2class_id,
)
from megatron.spiral.utils import is_spiral_param
import megatron.spiral.build_state as sbs


spiral_init_context = 0
top_level_context = None

_orig_torch_tensor = torch.tensor
_orig_torch_empty = torch.empty
_orig_torch_zeros = torch.zeros
_orig_torch_ones = torch.ones
_orig_torch_full = torch.full
_orig_torch_arange = torch.arange
_orig_torch_eye = torch.eye
_orig_torch_randn = torch.randn


class SpiralParamStatus(Enum):
    # parameter is fully present on local device and ready for use
    ACTIVE = 1

    # parameter is in CPU
    REMOTE = 2

    # parameter is not available
    UNAVAILABLE = 3


def wrapper_for_fp_tensor_constructor(
    fn: Callable, target_fp_dtype: torch.dtype
) -> Callable:

    def wrapped_fn(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            kwargs["device"] = torch.cuda.current_device()
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.data = tensor.data.to(target_fp_dtype)

        return tensor

    return wrapped_fn


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:

    def new_tensor(cls, *args, **kwargs) -> Tensor:
        device = torch.cuda.current_device()
        if not args:
            args = (0,)
        tensor = _orig_torch_empty(0, device=device).new_empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)

        return tensor

    return new_tensor


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    num_spiral_parameters = 0
    num_spiral_elements = 0

    def __init__(self, enabled=True, dtype=None):
        """A context to enable massive model construction for training with PP.
        Models are automatically partitioned across the system and converted to half precision.

        Args:
            enabled (bool, optional): If ``False``, this context has no effect. Defaults to ``True``.
            dtype (torch.dtype, optional): Data type of the model.
        """

        self.enabled = enabled
        self.dtype = dtype
        assert self.dtype in [
            torch.half,
            torch.bfloat16,
            torch.float,
        ], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.bfloat16, torch.float]"

    def __enter__(self):
        if not self.enabled:
            return

        global spiral_init_context
        if spiral_init_context == 0:
            self.patch_init_and_builtins()
            global top_level_context
            top_level_context = self
        spiral_init_context += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        global spiral_init_context
        spiral_init_context -= 1

        # Exiting the top level context
        if spiral_init_context == 0:
            self.unpatch_init_and_builtins()
            global top_level_context
            top_level_context = None

            billion_elems = (
                InsertPostInitMethodToModuleSubClasses.num_spiral_elements / 1e9
            )
            num_params = InsertPostInitMethodToModuleSubClasses.num_spiral_parameters
            spiral_print(
                f"finished initializing spiral params = {num_params}, elems = {billion_elems:.2f}B"
            )

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    def patch_init_and_builtins(self):

        def apply(orig_module_apply_fn: Callable) -> Callable:
            def get_wrapped_fn_to_apply(fn_to_apply: Callable) -> Callable:
                if hasattr(fn_to_apply, "wrapped"):
                    return fn_to_apply

                @functools.wraps(fn_to_apply)
                def wrapped_fn_to_apply(module_to_apply_fn_to: Module) -> None:
                    params_to_apply_fn_to: Iterable[Parameter] = [
                        p
                        for p in module_to_apply_fn_to.parameters(recurse=False)
                        if is_spiral_param(p)
                    ]

                    # Copy parameters to local device
                    for param in params_to_apply_fn_to:
                        param.fetch()
                    # Apply function
                    fn_to_apply(module_to_apply_fn_to)
                    # Copy parameters back to remote device
                    for param in params_to_apply_fn_to:
                        param.offload()

                wrapped_fn_to_apply.wrapped = True

                return wrapped_fn_to_apply

            @functools.wraps(orig_module_apply_fn)
            def wrapped_apply(module: Module, fn_to_apply: Callable) -> None:
                orig_module_apply_fn(
                    module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply

        def offload_after(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                is_child_module = False
                if not hasattr(module, "_spiral_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_spiral_child_entered", True)

                    # attach spiral module recurse attributes
                    num_module_spiral_parameters_before = InsertPostInitMethodToModuleSubClasses.num_spiral_parameters

                f(module, *args, **kwargs)

                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_spiral_child_entered")
                    self._post_init_method(module)

                    # attach spiral module recurse attributes
                    num_module_spiral_parameters_after = InsertPostInitMethodToModuleSubClasses.num_spiral_parameters
                    setattr(module, "num_spiral_params_recurse",
                            num_module_spiral_parameters_after - num_module_spiral_parameters_before)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = offload_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls._old_init = cls.__init__
            cls.__init__ = offload_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _enable_class(subclass)

        # holding onto some methods so we can put them back the way they were in __exit__
        torch.nn.modules.module.Module._old_init_subclass = (
            torch.nn.modules.module.Module.__init_subclass__
        )
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(
            _init_subclass)
        if SpiralInitContext.override_module_apply:
            torch.nn.modules.module.Module.apply = apply(
                torch.nn.modules.module.Module._old_apply
            )

        self._add_tensor_creation_wrappers()

        self.patched = True

    def unpatch_init_and_builtins(self):
        if self.patched:

            def _disable_class(cls):
                cls.__init__ = cls._old_init

            for subclass in get_all_subclasses(torch.nn.modules.module.Module):
                _disable_class(subclass)

            # putting methods back the way we found them
            torch.nn.modules.module.Module.__init_subclass__ = (
                torch.nn.modules.module.Module._old_init_subclass
            )
            if SpiralInitContext.override_module_apply:
                torch.nn.modules.module.Module.apply = (
                    torch.nn.modules.module.Module._old_apply
                )

            self._remove_tensor_creation_wrappers()

            self.patched = False

    def _add_tensor_creation_wrappers(self):
        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.tensor = wrapper_for_fp_tensor_constructor(
            _orig_torch_tensor, self.dtype)
        torch.empty = wrapper_for_fp_tensor_constructor(
            _orig_torch_empty, self.dtype)
        torch.zeros = wrapper_for_fp_tensor_constructor(
            _orig_torch_zeros, self.dtype)
        torch.ones = wrapper_for_fp_tensor_constructor(
            _orig_torch_ones, self.dtype)
        torch.full = wrapper_for_fp_tensor_constructor(
            _orig_torch_full, self.dtype)
        torch.arange = wrapper_for_fp_tensor_constructor(
            _orig_torch_arange, self.dtype)
        torch.eye = wrapper_for_fp_tensor_constructor(
            _orig_torch_eye, self.dtype)
        torch.randn = wrapper_for_fp_tensor_constructor(
            _orig_torch_randn, self.dtype)

    def _remove_tensor_creation_wrappers(self):
        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.tensor = _orig_torch_tensor
        torch.empty = _orig_torch_empty
        torch.zeros = _orig_torch_zeros
        torch.ones = _orig_torch_ones
        torch.full = _orig_torch_full
        torch.arange = _orig_torch_arange
        torch.eye = _orig_torch_eye
        torch.randn = _orig_torch_randn

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass


class SpiralInitContext(InsertPostInitMethodToModuleSubClasses):
    override_module_apply = False  # unused but kept for future

    def __init__(
        self,
        enabled=True,
        dtype=None,
    ):
        """A context to enable massive model construction for training with PP.
        Models are automatically partitioned across the system and converted to half precision.

        Args:
          enabled (bool, optional): If ``False``, this context has no effect. Defaults to ``True``.
          dtype (torch.dtype, optional): Data type of the model.
        """

        self.local_device = torch.cuda.current_device()
        self.remote_device = torch.device("cpu")
        super().__init__(enabled=enabled, dtype=dtype)

    def _post_init_method(self, module):
        num_spiral_params = 0

        # Attach spiral stage attribute
        from megatron.core import mpu

        if not hasattr(module, "spiral_forward_stage_id"):
            setattr(
                module,
                "spiral_forward_stage_id",
                mpu.get_spiral_forward_virtual_rank(),
            )
        if not hasattr(module, "spiral_backward_stage_id"):
            setattr(
                module,
                "spiral_backward_stage_id",
                mpu.get_spiral_backward_virtual_rank(),
            )
        assert (
            module.spiral_forward_stage_id is not None
            or module.spiral_backward_stage_id is not None
        ), f"{module.__class__.__name__} is neither forward nor backward stage"

        # Convert module's parameters
        for param in module.parameters(recurse=False):
            # TODO (SpiralPipe) Currently, all params are converted to spiral params.
            # Modify this when selectively converting params to spiral params is required
            if not is_spiral_param(param):
                self._convert_to_spiral_param(param)

            if is_spiral_param(param):
                num_spiral_params += 1

            if get_args().spiral_remap:
                # Conditionally ordered offload and free
                # fwd stage: free; since its spiral_tensor will be re-mapped
                # bwd stage: offload -> free; its data will be used
                if is_spiral_param(param):
                    if module.spiral_forward_stage_id is not None:
                        param.free()
                    elif module.spiral_backward_stage_id is not None:
                        param.offload()
                        param.free()
            else:
                param.offload()
                param.free()

        # Attach spiral module methods
        module.spiral_fetch = lambda *args, **kwargs: self._fetch_module(
            module, *args, **kwargs
        )
        module.spiral_offload = lambda *args, **kwargs: self._offload_module(
            module, *args, **kwargs
        )
        module.spiral_free = lambda *args, **kwargs: self._free_module(
            module, *args, **kwargs
        )
        module.spiral_offload_grad = lambda *args, **kwargs: self._offload_grad_module(
            module, *args, **kwargs
        )
        module.spiral_save_params_info = lambda *args, **kwargs: self._save_module_params_info(
            module, *args, **kwargs
        )
        module.spiral_remap = lambda *args, **kwargs: self._remap_module(
            module, *args, **kwargs
        )

        # Attach spiral module attributes
        setattr(module, "num_spiral_params", num_spiral_params)

    def _convert_to_spiral_param(self, param):
        """Converts a parameter to a SpiralPipe parameter.

        NOTE (SpiralPipe) .to() changes dataptr while .copy_() preserves. .empty() also changes dataptr
                    Be aware not to allocate redundant memory and provoke unnecessary garbage collection for allocators

        NOTE (SpiralPipe) Currently, all spiral_tensor is pinned by default.
                    This is because Spiral allocator memory is registered as pinned memory.
                    As a result, pin_memory argument should not be passed to Tensor creation APIs.
                    Although there is no max pinned memory limit in CUDA, there are possible overheads(https://stackoverflow.com/questions/22300100/about-pinned-memory-in-cuda-is-there-an-upper-limit-on-it)
        """
        InsertPostInitMethodToModuleSubClasses.num_spiral_parameters += 1
        InsertPostInitMethodToModuleSubClasses.num_spiral_elements += param.numel()

        param.spiral_id = sbs.get_add_spiral_next_param_number_to_build()
        param.spiral_status = SpiralParamStatus.ACTIVE
        param.spiral_shape = param.shape
        param.spiral_stride = param.stride()
        param.spiral_storage_offset = param.storage_offset()
        param.spiral_numel = param.numel()
        param.spiral_tensor = torch.empty(
            0,
            dtype=param.dtype,
            device=self.remote_device,
            # NOTE (SpiralPipe) pin_memory arg should not be passed
        )

        def _free_data(param: Parameter) -> None:
            """Free weight data of a parameter."""
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            if param.spiral_tensor.numel() == param.spiral_numel:
                param.spiral_status = SpiralParamStatus.REMOTE
            else:
                param.spiral_status = SpiralParamStatus.UNAVAILABLE

        def _offload_data(param, non_blocking=False):
            """Offload weight data of a parameter to remote device."""
            assert param.spiral_status == SpiralParamStatus.ACTIVE
            assert param.spiral_tensor is not None, "Offload tensor is None"

            if param.spiral_tensor.numel() == 0:
                if get_args().spiral_remap:
                    assert sbs.get_spiral_backward_stage_build_phase() is not None, \
                        "Offloading to empty spiral tensor should be executed only once during SpiralPipe with remapping backward stage build phase"
                else:
                    assert sbs.get_spiral_forward_stage_build_phase() is not None, \
                        "Offloading to empty spiral tensor should be executed only once during SpiralPipe w/o remapping forward stage build phase"
                param.spiral_tensor = param.data.to(
                    device=self.remote_device, non_blocking=non_blocking
                )  # NOTE (SpiralPipe) pin_memory() should not be called
            else:
                assert (
                    param.spiral_tensor.shape == param.data.shape
                ), f"Offload tensor shape mismatch ({param.spiral_tensor.shape} != {param.data.shape})"
                param.spiral_tensor.copy_(
                    param.data, non_blocking=non_blocking)
            if not non_blocking:
                # NOTE: for non-blocking offload, spiral_status should be changed after waiting in the caller
                param.spiral_status = SpiralParamStatus.REMOTE

        def _fetch_data(param, non_blocking=False):
            assert param.spiral_status == SpiralParamStatus.REMOTE
            assert param.spiral_tensor is not None, "Fetch tensor is None"

            if param.numel() == 0:
                if get_thunder_group().IsParamDataLocal(param.spiral_id):
                    param.data = param.spiral_tensor.to(
                        device=self.local_device, non_blocking=non_blocking
                    ).view(param.spiral_shape)
                else:
                    param.data = torch.empty(
                        param.spiral_shape, device=self.local_device)
                    get_thunder_group().FetchRemoteParam(
                        param.spiral_id,
                        non_blocking,
                        param.data.data_ptr()
                    )
            else:
                assert (
                    param.spiral_tensor.shape == param.data.shape
                ), f"Fetch tensor shape mismatch ({param.spiral_tensor.shape} != {param.data.shape})"
                param.data.copy_(param.spiral_tensor,
                                 non_blocking=non_blocking)
            if not non_blocking:
                # NOTE: for non-blocking fetch, spiral_status should be changed after waiting in the caller
                param.spiral_status = SpiralParamStatus.ACTIVE

        def _offload_grad(param, non_blocking=False):
            """Offload a gradient to remote device."""

            if hasattr(param, "main_grad") and getattr(param, "main_grad") is not None:
                if (
                    hasattr(param.spiral_tensor, "main_grad")
                    and getattr(param.spiral_tensor, "main_grad") is not None
                ):
                    # NOTE (SpiralPipe) Only check for numel, since main_grad shape "may" differ on GPU and CPU depending on optimizer
                    assert (
                        param.main_grad.numel() == param.spiral_tensor.main_grad.numel()
                    ), "Main grad numel mismatch"
                    param.spiral_tensor.main_grad.copy_(
                        param.main_grad, non_blocking=non_blocking
                    )
                else:
                    param.spiral_tensor.main_grad = param.main_grad.to(
                        self.remote_device, non_blocking=non_blocking
                    )

            if hasattr(param, "grad") and getattr(param, "grad") is not None:
                if (
                    hasattr(param.spiral_tensor, "grad")
                    and getattr(param.spiral_tensor, "grad") is not None
                ):
                    # NOTE (SpiralPipe) Only check for numel, since grad shape "may" differ on GPU and CPU depending on optimizer
                    assert (
                        param.grad.numel() == param.spiral_tensor.grad.numel()
                    ), "Grad numel mismatch"
                    param.spiral_tensor.grad.copy_(
                        param.grad, non_blocking=non_blocking
                    )
                else:
                    param.spiral_tensor.grad = param.grad.to(
                        self.remote_device, non_blocking=non_blocking
                    )

        param.free = lambda *args, **kwargs: _free_data(param, *args, **kwargs)
        param.offload = lambda *args, **kwargs: _offload_data(
            param, *args, **kwargs)
        param.fetch = lambda *args, **kwargs: _fetch_data(
            param, *args, **kwargs)
        param.offload_grad = lambda *args, **kwargs: _offload_grad(
            param, *args, **kwargs
        )

    @ nvtx.annotate("fetch_module", color="orange")
    def _fetch_module(self, module, non_blocking=False):
        spiral_report_memory(
            f"before fetch module {debug_module2class_id(module)}")
        for param in module.parameters(recurse=True):
            if is_spiral_param(param):
                param.fetch(non_blocking=non_blocking)
        spiral_report_memory(
            f"after fetch module {debug_module2class_id(module)}")

    @ nvtx.annotate("offload_module", color="yellow")
    def _offload_module(self, module, non_blocking=False):
        spiral_report_memory(
            f"before offload module {debug_module2class_id(module)}")
        for param in module.parameters(recurse=True):
            if is_spiral_param(param):
                param.offload(non_blocking=non_blocking)
        spiral_report_memory(
            f"after offload module {debug_module2class_id(module)}")

    @ nvtx.annotate("free_module", color="darkgreen")
    def _free_module(self, module):
        spiral_report_memory(
            f"before free module {debug_module2class_id(module)}")
        for param in module.parameters(recurse=True):
            if is_spiral_param(param):
                param.free()
        spiral_report_memory(
            f"after free module {debug_module2class_id(module)}")

    @ nvtx.annotate("offload_grad_module", color="white")
    def _offload_grad_module(self, module, non_blocking=False):
        spiral_report_memory(
            f"before offload grad module {debug_module2class_id(module)}"
        )
        for param in module.parameters(recurse=True):
            if is_spiral_param(param):
                param.offload_grad(non_blocking=non_blocking)
        spiral_report_memory(
            f"after offload grad module {debug_module2class_id(module)}"
        )

    # NOTE (SpiralPipe) Building w/o remapping skips this function call, as it does not remap param data, hence no need to save param data info
    def _save_module_params_info(self, module):
        for param in module.parameters(recurse=True):
            if is_spiral_param(param):
                get_thunder_group().SetParamDataInfo(
                    param.spiral_id,
                    (
                        param.spiral_tensor.data_ptr()
                        if isinstance(param.spiral_tensor, torch.Tensor)
                        else param.spiral_tensor.data.ptr
                    ),
                    param.spiral_tensor.numel() * param.spiral_tensor.element_size(),
                )

    # NOTE (SpiralPipe) Prior to call, must reset spiral build state's "forward number of spiral params allocated". (Look at training.py for example)
    # If reset_spiral_forward_stage_build_phase_num_spiral_params_allocated() hasn't been called, spiral_ids will be incorrectly reassigned.

    def _remap_module(self, module):
        if get_args().spiral_remap:
            for param in module.parameters(recurse=True):
                if is_spiral_param(param):
                    param.spiral_id = (
                        sbs.get_add_spiral_next_param_number_to_build()
                    )
                    get_thunder_group().RemapParamData(
                        param.spiral_tensor,
                        param.spiral_id,
                        param.spiral_shape,
                        param.spiral_stride,
                        param.spiral_storage_offset,
                    )

                    if param.spiral_status == SpiralParamStatus.UNAVAILABLE:
                        param.spiral_status = SpiralParamStatus.REMOTE
                else:
                    print("[DY] NO Spiral Param")

        else:
            for param in module.parameters(recurse=True):
                if is_spiral_param(param):
                    param.spiral_id = sbs.get_add_spiral_next_param_number_to_build()


@ contextmanager
def patch_extra_repr():
    """A context to patch the ``extra_repr`` method of all subclasses of ``torch.nn.Module`` to include spiral information of the module.
    """
    try:
        def _extra_repr(self):
            spiral_id_generator = (
                p.spiral_id for p in self.parameters(recurse=False) if is_spiral_param(p)
            )
            first_spiral_id = next(spiral_id_generator, None)
            last_spiral_id = None if first_spiral_id is None else first_spiral_id + \
                self.num_spiral_params - 1
            return (
                f"fid={self.spiral_forward_stage_id}"
                + f", bid={self.spiral_backward_stage_id}"
                + (f", lid={self.layer_number}" if hasattr(self,
                   "layer_number") else "")
                # + f", spiral_params={self.num_spiral_params}"
                # + f", spiral_params_recurse={self.num_spiral_params_recurse}"
                + f", spiral_ids={first_spiral_id}..{last_spiral_id}"
            )

        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            subclass._old_extra_repr = subclass.extra_repr
            subclass.extra_repr = _extra_repr
        yield
    finally:
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            subclass.extra_repr = subclass._old_extra_repr
            del subclass._old_extra_repr
