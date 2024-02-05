# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# Code modified from DeepSpeed

from typing import Callable, Iterable
from enum import Enum
import functools

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from megatron.spiral.debug import spiral_print
from megatron.spiral.utils import is_spiral_param


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

    # parameter is being fetched
    INFLIGHT = 3


def wrapper_for_fp_tensor_constructor(fn: Callable, target_fp_dtype: torch.dtype) -> Callable:

    def wrapped_fn(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            kwargs['device'] = torch.cuda.current_device()
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.data = tensor.data.to(target_fp_dtype)

        return tensor

    return wrapped_fn


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:

    def new_tensor(cls, *args, **kwargs) -> Tensor:
        device = torch.cuda.current_device()
        if not args:
            args = (0, )
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
    num_module_parameters = 0
    num_module_elements = 0

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
                InsertPostInitMethodToModuleSubClasses.num_module_elements / 1e9
            )
            num_params = (
                InsertPostInitMethodToModuleSubClasses.num_module_parameters
            )
            spiral_print(
                f"finished initializing model - num_params = {num_params}, num_elems = {billion_elems:.2f}B"
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
                    params_to_apply_fn_to: Iterable[Parameter] = [p for p in module_to_apply_fn_to.parameters(recurse=False) if is_spiral_param(p)]

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
                orig_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply

        def offload_after(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                is_child_module = False
                if not hasattr(module, "_spiral_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_spiral_child_entered", True)

                f(module, *args, **kwargs)

                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_spiral_child_entered")
                    self._post_init_method(module)

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
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
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
            _orig_torch_tensor, self.dtype
        )
        torch.empty = wrapper_for_fp_tensor_constructor(
            _orig_torch_empty, self.dtype
        )
        torch.zeros = wrapper_for_fp_tensor_constructor(
            _orig_torch_zeros, self.dtype
        )
        torch.ones = wrapper_for_fp_tensor_constructor(_orig_torch_ones, self.dtype)
        torch.full = wrapper_for_fp_tensor_constructor(_orig_torch_full, self.dtype)
        torch.arange = wrapper_for_fp_tensor_constructor(
            _orig_torch_arange, self.dtype
        )
        torch.eye = wrapper_for_fp_tensor_constructor(_orig_torch_eye, self.dtype)
        torch.randn = wrapper_for_fp_tensor_constructor(
            _orig_torch_randn, self.dtype
        )


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
    override_module_apply = False # unused but kept for future

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

        # Attach spiral stage attribute
        from megatron.core import mpu
        if not hasattr(module, "is_spiral_pipeline_parallel_forward_stage"):
            setattr(module, "is_spiral_pipeline_parallel_forward_stage", mpu.is_spiral_pipeline_parallel_forward_stage())
        if not hasattr(module, "is_spiral_pipeline_parallel_backward_stage"):
            setattr(module, "is_spiral_pipeline_parallel_backward_stage", mpu.is_spiral_pipeline_parallel_backward_stage())
        assert module.is_spiral_pipeline_parallel_forward_stage ^ module.is_spiral_pipeline_parallel_backward_stage, f"{module.__class__.__name__} is neither forward nor backward stage"

        # Convert and offload module's parameters
        for param in module.parameters(recurse=False):
            InsertPostInitMethodToModuleSubClasses.num_module_parameters += 1
            InsertPostInitMethodToModuleSubClasses.num_module_elements += param.numel()

            # TODO (mcrl) Currently, all params are converted to spiral params. 
            # Modify this when selectively converting params to spiral params is required
            if not self._is_spiral_param(param):
                self._convert_to_spiral_param(param)

            if self._is_spiral_param(param):
                # TODO (mcrl) select offload or borrow
                param.offload()
                # param.borrow()
                param.free()
    

    def _is_spiral_param(self, param):
        if not torch.is_tensor(param):
            return False
        return hasattr(param, "spiral_tensor")


    def _convert_to_spiral_param(self, param):
        """Converts a parameter to a SpiralPipe parameter."""
        param.spiral_status = SpiralParamStatus.ACTIVE # After original __init__, all params initialized active
        param.spiral_shape = param.shape
        param.spiral_numel = param.numel()
        param.spiral_tensor = None # Stores copy of the tensor on remote device

        def _free_param(param: Parameter) -> None:
            """Free underlying storage of a parameter."""
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            if param.spiral_tensor is not None:
                param.spiral_status = SpiralParamStatus.REMOTE

        def _offload_param(param):
            """Offload a parameter to remote device."""
            assert param.spiral_status == SpiralParamStatus.ACTIVE

            param.spiral_tensor = param.data.to(self.remote_device)
            param.spiral_status = SpiralParamStatus.REMOTE

        def _borrow_param(param):
            param.spiral_tensor = torch.empty_like(param.data)
            # TODO (mcrl) borrow tensor
            param.spiral_status = SpiralParamStatus.REMOTE

        # TODO (mcrl) implement async fetch
        def _fetch_param(param, async_op=False):
            assert param.spiral_status == SpiralParamStatus.REMOTE
            assert param.spiral_tensor is not None
            assert param.spiral_tensor.shape == param.spiral_shape

            param.data = param.spiral_tensor.to(self.local_device).view(param.spiral_shape)
            param.spiral_status = SpiralParamStatus.ACTIVE

        param.free = lambda: _free_param(param)
        param.offload = lambda: _offload_param(param)
        param.borrow = lambda: _borrow_param(param)
        param.fetch = lambda: _fetch_param(param)