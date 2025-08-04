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
from megatron.spipe import get_thunder_group
from megatron.spipe.initialize import get_thunder_cuda_manager
from megatron.spipe.debug import (
    spipe_print,
    spipe_report_memory,
    debug_module2class_id,
)
from megatron.spipe.utils import is_spipe_param
import megatron.spipe.build_state as sbs


spipe_init_context = 0
top_level_context = None


class SPipeParamStatus(Enum):
    # parameter is fully present on local device and ready for use
    GPU = 1

    # parameter is in CPU memory of local/remote host
    CPU = 2


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
    num_spipe_parameters = 0
    num_spipe_elements = 0

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

        global spipe_init_context
        if spipe_init_context == 0:
            self.patch_init_and_builtins()
            global top_level_context
            top_level_context = self
        spipe_init_context += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        global spipe_init_context
        spipe_init_context -= 1

        # Exiting the top level context
        if spipe_init_context == 0:
            self.unpatch_init_and_builtins()
            global top_level_context
            top_level_context = None

            billion_elems = (
                InsertPostInitMethodToModuleSubClasses.num_spipe_elements / 1e9
            )
            num_params = InsertPostInitMethodToModuleSubClasses.num_spipe_parameters
            spipe_print(
                f"finished initializing spipe params = {num_params}, elems = {billion_elems:.2f}B"
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
                        if is_spipe_param(p)
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
                orig_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply

        def offload_after(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                is_child_module = False
                if not hasattr(module, "_spipe_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_spipe_child_entered", True)

                    # attach spipe module recurse attributes
                    num_module_spipe_parameters_before = InsertPostInitMethodToModuleSubClasses.num_spipe_parameters

                f(module, *args, **kwargs)

                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_spipe_child_entered")
                    self._post_init_method(module)

                    # attach spipe module recurse attributes
                    num_module_spipe_parameters_after = InsertPostInitMethodToModuleSubClasses.num_spipe_parameters
                    setattr(module, "num_spipe_params_recurse", num_module_spipe_parameters_after - num_module_spipe_parameters_before)

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
        if SPipeInitContext.override_module_apply:
            torch.nn.modules.module.Module.apply = apply(
                torch.nn.modules.module.Module._old_apply
            )

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
            if SPipeInitContext.override_module_apply:
                torch.nn.modules.module.Module.apply = (
                    torch.nn.modules.module.Module._old_apply
                )

            self.patched = False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass


class SPipeInitContext(InsertPostInitMethodToModuleSubClasses):
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
        num_spipe_params = 0

        # Attach spipe stage attribute
        from megatron.core import mpu

        if not hasattr(module, "spipe_forward_stage_id"):
            setattr(
                module,
                "spipe_forward_stage_id",
                mpu.get_spipe_forward_virtual_rank(),
            )
        if not hasattr(module, "spipe_backward_stage_id"):
            setattr(
                module,
                "spipe_backward_stage_id",
                mpu.get_spipe_backward_virtual_rank(),
            )
        assert (
            module.spipe_forward_stage_id is not None
            or module.spipe_backward_stage_id is not None
        ), f"{module.__class__.__name__} is neither forward nor backward stage"

        # Convert module's parameters
        for param in module.parameters(recurse=False):
            # TODO (SPipe) Currently, all params are converted to spipe params.
            # Modify this when selectively converting params to spipe params is required
            if not is_spipe_param(param):
                self._convert_to_spipe_param(param)

            if is_spipe_param(param):
                num_spipe_params += 1

            if get_args().spipe_remap:
                # Conditionally ordered offload and free
                # fwd stage: free; since its spipe_tensor will be re-mapped
                # bwd stage: offload -> free; its data will be used
                if is_spipe_param(param):
                    if module.spipe_forward_stage_id is not None:
                        param.free()
                    elif module.spipe_backward_stage_id is not None:
                        param.offload()
                        param.free()
            else:
                param.offload()
                param.free()

        # Attach spipe module methods
        module.spipe_fetch = lambda *args, **kwargs: self._fetch_module(
            module, *args, **kwargs
        )
        module.spipe_offload = lambda *args, **kwargs: self._offload_module(
            module, *args, **kwargs
        )
        module.spipe_free = lambda *args, **kwargs: self._free_module(
            module, *args, **kwargs
        )
        module.spipe_offload_grad = lambda *args, **kwargs: self._offload_grad_module(
            module, *args, **kwargs
        )
        module.spipe_free_grad = lambda *args, **kwargs: self._free_grad_module(
            module, *args, **kwargs
        )
        module.spipe_save_params_info = lambda *args, **kwargs: self._save_module_params_info(
            module, *args, **kwargs
        )
        module.spipe_remap = lambda *args, **kwargs: self._remap_module(
            module, *args, **kwargs
        )
        module.spipe_assert_free_grad = lambda *args, **kwargs: self._assert_free_grad_module(
            module, *args, **kwargs
        )

        # Attach spipe module attributes
        setattr(module, "num_spipe_params", num_spipe_params)

    def _convert_to_spipe_param(self, param):
        """Converts a parameter to a SPipe parameter.

        NOTE (SPipe) .to() changes dataptr while .copy_() preserves. .empty() also changes dataptr
                    Be aware not to allocate redundant memory and provoke unnecessary garbage collection for allocators

        NOTE (SPipe) Currently, all spipe_tensor is pinned by default.
                    This is because spipe allocator memory is registered as pinned memory.
                    As a result, pin_memory argument should not be passed to Tensor creation APIs.
                    Although there is no max pinned memory limit in CUDA, there are possible overheads(https://stackoverflow.com/questions/22300100/about-pinned-memory-in-cuda-is-there-an-upper-limit-on-it)
        """
        InsertPostInitMethodToModuleSubClasses.num_spipe_parameters += 1
        InsertPostInitMethodToModuleSubClasses.num_spipe_elements += param.numel()

        param.spipe_id = sbs.get_add_spipe_next_param_number_to_build()
        param.spipe_status = SPipeParamStatus.GPU if param.is_cuda else SPipeParamStatus.CPU
        param.spipe_shape = param.shape
        param.spipe_stride = param.stride()
        param.spipe_storage_offset = param.storage_offset()
        param.spipe_numel = param.numel()
        param.spipe_tensor = torch.empty(
            0,
            dtype=param.dtype,
            device=self.remote_device,
            # NOTE (SPipe) pin_memory arg should not be passed
        )

        def _free_data(param: Parameter) -> None:
            """Free weight data of a parameter."""
            param.data = torch.empty(0, dtype=param.dtype, device=self.local_device)
            param.spipe_status = SPipeParamStatus.CPU

        def _offload_data(param, non_blocking=False):
            """Offload weight data of a parameter to remote device."""
            assert param.spipe_tensor is not None, "Offload tensor is None"

            if param.spipe_tensor.numel() == 0:
                if get_args().spipe_remap:
                    assert sbs.get_spipe_backward_stage_build_phase() is not None, \
                        "Offloading to empty spipe tensor should be executed only once during SPipe with remapping backward stage build phase"
                    # NOTE (SPipe) pin_memory arg should not be passed when offloading to shared memory, since it transfers the data to pinned memory region
                    param.spipe_tensor = torch.empty(param.spipe_shape, device=self.remote_device, dtype=param.dtype)
                else:
                    assert sbs.get_spipe_forward_stage_build_phase() is not None, \
                        "Offloading to empty spipe tensor should be executed only once during SPipe w/o remapping forward stage build phase"
                    # NOTE (SPipe) pin_memory arg can be passed, since it is not offloading to shared memory
                    param.spipe_tensor = torch.empty(param.spipe_shape, device=self.remote_device, dtype=param.dtype, pin_memory=True)
                param.spipe_tensor.copy_(param.data, non_blocking=non_blocking)
            else:
                assert (
                    param.spipe_tensor.shape == param.data.shape
                ), f"Offload tensor shape mismatch ({param.spipe_tensor.shape} != {param.data.shape})"
                param.spipe_tensor.copy_(param.data, non_blocking=non_blocking)

        def _fetch_data(param, non_blocking=False):
            # If parameter is already in gpu memory, skip fetch.
            # NOTE: Most cases should naturally fetch from CPU in spipe case.
            if param.spipe_status == SPipeParamStatus.GPU:
                return

            assert param.spipe_tensor is not None, "Fetch tensor is None"

            if param.numel() == 0:
                if not get_args().spipe_remap or get_thunder_group().IsParamDataLocal(param.spipe_id):
                    param.data = param.spipe_tensor.to(
                        device=self.local_device, non_blocking=non_blocking
                    ).view(param.spipe_shape)
                else:
                    param.data = torch.empty(param.spipe_shape, device=self.local_device, dtype=param.dtype)
                    get_thunder_group().FetchRemoteParam(
                        param.spipe_id,
                        non_blocking,
                        param.data.data_ptr()
                    )
            else:
                assert (
                    param.spipe_tensor.shape == param.data.shape
                ), f"Fetch tensor shape mismatch ({param.spipe_tensor.shape} != {param.data.shape})"
                if not get_args().spipe_remap or get_thunder_group().IsParamDataLocal(param.spipe_id):
                    param.data.copy_(param.spipe_tensor, non_blocking=non_blocking)
                else:
                    param.data = torch.empty(param.spipe_shape, device=self.local_device, dtype=param.dtype)
                    get_thunder_group().FetchRemoteParam(
                        param.spipe_id,
                        non_blocking,
                        param.data.data_ptr()
                    )
            if not non_blocking:
                # NOTE: for non-blocking fetch, spipe_status should be changed after waiting in the caller
                param.spipe_status = SPipeParamStatus.GPU

        def _offload_grad(param, non_blocking=False):
            """Offload a gradient to remote device."""

            # Determine whether the params have main-grad field
            params_have_main_grad = False
            if get_args().DDP_impl == "local":
                params_have_main_grad = True

            # Always offload param.grad/param.main_grad into param.spipe_tensor.grad
            if params_have_main_grad:
                assert hasattr(param, "main_grad") and getattr(param, "main_grad") is not None

                if (
                    hasattr(param.spipe_tensor, "grad")
                    and getattr(param.spipe_tensor, "grad") is not None
                ):
                    # NOTE (SPipe) Corresponds to when optimizer sets `set_to_none=False`
                    # NOTE (SPipe) Only check for numel, since main_grad shape "may" differ on GPU and CPU depending on optimizer
                    assert (
                        param.main_grad.numel() == param.spipe_tensor.grad.numel()
                    ), "param.main_grad and param.spipe_tensor.grad numel mismatch"
                    param.spipe_tensor.grad.copy_(
                        param.main_grad, non_blocking=non_blocking
                    )
                else:
                    # NOTE (SPipe) Corresponds to when optimizer sets `set_to_none=True`
                    param.spipe_tensor.grad = torch.empty(param.main_grad.shape, device=self.remote_device, dtype=param.main_grad.dtype, pin_memory=True)
                    param.spipe_tensor.grad.copy_(
                        param.main_grad, non_blocking=non_blocking
                    )
            else:
                assert hasattr(param, "grad") and getattr(param, "grad") is not None

                if (
                    hasattr(param.spipe_tensor, "grad")
                    and getattr(param.spipe_tensor, "grad") is not None
                ):
                    # NOTE (SPipe) Corresponds to when optimizer sets `set_to_none=False`
                    # NOTE (SPipe) Only check for numel, since grad shape "may" differ on GPU and CPU depending on optimizer
                    assert (
                        param.grad.numel() == param.spipe_tensor.grad.numel()
                    ), "param.grad and param.spipe_tensor.grad numel mismatch"
                    param.spipe_tensor.grad.copy_(
                        param.grad, non_blocking=non_blocking
                    )
                else:
                    # NOTE (SPipe) Corresponds to when optimizer sets `set_to_none=True`
                    param.spipe_tensor.grad = torch.empty(param.grad.shape, device=self.remote_device, dtype=param.grad.dtype, pin_memory=True)
                    param.spipe_tensor.grad.copy_(
                        param.grad, non_blocking=non_blocking
                    )

        def _free_grad(param: Parameter) -> None:
            """Free grad of a parameter."""
            # Determine whether the params have main-grad field
            params_have_main_grad = False
            if get_args().DDP_impl == "local":
                params_have_main_grad = True

            if params_have_main_grad:
                assert hasattr(param, "main_grad")
                # Ensure the tensor memory is not reused for another tensor until all work queued on stream is complete
                if param.main_grad is not None:
                    param.main_grad.record_stream(get_thunder_cuda_manager().Stream('offload'))
                param.main_grad = None
            if hasattr(param, "grad"):
                # Ensure the tensor memory is not reused for another tensor until all work queued on stream is complete
                if param.grad is not None:
                    param.grad.record_stream(get_thunder_cuda_manager().Stream('offload'))
                param.grad = None

        def _assert_free_grad(param: Parameter) -> None:
            """Assert free grad of a parameter."""
            # Determine whether the params have main-grad field
            params_have_main_grad = False
            if get_args().DDP_impl == "local":
                params_have_main_grad = True

            if params_have_main_grad:
                assert hasattr(param, "main_grad")
                assert param.main_grad is None
            if hasattr(param, "grad"):
                assert param.grad is None

        param.free = lambda *args, **kwargs: _free_data(param, *args, **kwargs)
        param.offload = lambda *args, **kwargs: _offload_data(param, *args, **kwargs)
        param.fetch = lambda *args, **kwargs: _fetch_data(param, *args, **kwargs)
        param.offload_grad = lambda *args, **kwargs: _offload_grad(
            param, *args, **kwargs
        )
        param.free_grad = lambda *args, **kwargs: _free_grad(param, *args, **kwargs)

        # checker methods
        param.assert_free_grad = lambda *args, **kwargs: _assert_free_grad(param, *args, **kwargs)

    @nvtx.annotate("fetch_module", color="orange")
    def _fetch_module(self, module, non_blocking=False):
        spipe_report_memory(f"before fetch module {debug_module2class_id(module)}")
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                param.fetch(non_blocking=non_blocking)
        spipe_report_memory(f"after fetch module {debug_module2class_id(module)}")

    @nvtx.annotate("offload_module", color="yellow")
    def _offload_module(self, module, non_blocking=False):
        spipe_report_memory(f"before offload module {debug_module2class_id(module)}")
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                param.offload(non_blocking=non_blocking)
        spipe_report_memory(f"after offload module {debug_module2class_id(module)}")

    @nvtx.annotate("free_module", color="darkgreen")
    def _free_module(self, module):
        spipe_report_memory(f"before free module {debug_module2class_id(module)}")
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                param.free()
        spipe_report_memory(f"after free module {debug_module2class_id(module)}")

    @nvtx.annotate("offload_grad_module", color="white")
    def _offload_grad_module(self, module, non_blocking=False):
        spipe_report_memory(
            f"before offload grad module {debug_module2class_id(module)}"
        )
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                param.offload_grad(non_blocking=non_blocking)
        spipe_report_memory(
            f"after offload grad module {debug_module2class_id(module)}"
        )

    @nvtx.annotate("free_grad_module", color="white")
    def _free_grad_module(self, module):
        spipe_report_memory(
            f"before free grad module {debug_module2class_id(module)}"
        )
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                param.free_grad()
        spipe_report_memory(
            f"after free grad module {debug_module2class_id(module)}"
        )

    def _assert_free_grad_module(self, module):
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                param.assert_free_grad()

    # NOTE (SPipe) Building w/o remapping skips this function call, as it does not remap param data, hence no need to save param data info
    def _save_module_params_info(self, module):
        for param in module.parameters(recurse=True):
            if is_spipe_param(param):
                get_thunder_group().SetParamDataInfo(
                    param.spipe_id,
                    (
                        param.spipe_tensor.data_ptr()
                        if isinstance(param.spipe_tensor, torch.Tensor)
                        else param.spipe_tensor.data.ptr
                    ),
                    param.spipe_tensor.numel() * param.spipe_tensor.element_size(),
                )

    # NOTE (SPipe) Prior to call, must reset spipe build state's "forward number of spipe params allocated". (Look at training.py for example)
    # If reset_spipe_forward_stage_build_phase_num_spipe_params_allocated() hasn't been called, spipe_ids will be incorrectly reassigned.
    def _remap_module(self, module):
        if get_args().spipe_remap:
            for param in module.parameters(recurse=True):
                if is_spipe_param(param):
                    param.spipe_id = (
                        sbs.get_add_spipe_next_param_number_to_build()
                    )
                    if get_thunder_group().IsParamDataLocal(param.spipe_id):
                        get_thunder_group().RemapLocalParamData(
                            param.spipe_tensor,
                            param.spipe_id,
                            param.spipe_shape,
                            param.spipe_stride,
                            param.spipe_storage_offset,
                        )
                    param.spipe_status = SPipeParamStatus.CPU
        else:
            for param in module.parameters(recurse=True):
                if is_spipe_param(param):
                    param.spipe_id = sbs.get_add_spipe_next_param_number_to_build()


@contextmanager
def patch_extra_repr():
    """A context to patch the ``extra_repr`` method of all subclasses of ``torch.nn.Module`` to include spipe information of the module.
    """
    try:
        def _extra_repr(self):
            spipe_id_generator = (
                p.spipe_id for p in self.parameters(recurse=False) if is_spipe_param(p)
            )
            first_spipe_id = next(spipe_id_generator, None)
            last_spipe_id = None if first_spipe_id is None else first_spipe_id + self.num_spipe_params - 1
            return (
                f"fid={self.spipe_forward_stage_id}"
                + f", bid={self.spipe_backward_stage_id}"
                + (f", lid={self.layer_number}" if hasattr(self, "layer_number") else "")
                # + f", spipe_params={self.num_spipe_params}"
                # + f", spipe_params_recurse={self.num_spipe_params_recurse}"
                + f", spipe_ids={first_spipe_id}..{last_spipe_id}"
            )

        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            subclass._old_extra_repr = subclass.extra_repr
            subclass.extra_repr = _extra_repr
        yield
    finally:
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            subclass.extra_repr = subclass._old_extra_repr
            del subclass._old_extra_repr


def set_module_spipe_status(module, status: SPipeParamStatus):
    for param in module.parameters(recurse=True):
        if is_spipe_param(param):
            assert hasattr(param, "spipe_status"), "spipe_status not found in param"
            setattr(param, "spipe_status", status)