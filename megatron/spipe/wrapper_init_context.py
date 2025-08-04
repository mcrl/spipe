from typing import Callable, Iterable
import functools

import torch


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        self.patch_init_and_builtins()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unpatch_init_and_builtins()

    def patch_init_and_builtins(self):

        def copy_spipe_attrs_after(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                f(module, *args, **kwargs)
                if hasattr(module, "module"):
                    self._post_init_method(module)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = copy_spipe_attrs_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls._old_init = cls.__init__
            cls.__init__ = copy_spipe_attrs_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _enable_class(subclass)

        # holding onto some methods so we can put them back the way they were in __exit__
        torch.nn.modules.module.Module._old_init_subclass = (
            torch.nn.modules.module.Module.__init_subclass__
        )
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

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

            self.patched = False


class SPipeWrapperInitContext(InsertPostInitMethodToModuleSubClasses):

    SPIPE_INFIX = "spipe"

    def __init__(self, enabled=True):
        """A contex to enable copy of attributes from the wrapped module to the wrapper module.
        All attributes that contain the SPIPE_INFIX will be copied from the wrapped module to the wrapper module.
        """
        super().__init__(enabled=enabled)

    def _post_init_method(self, module):
        assert hasattr(module, "module"), "module must have a .module attribute"
        for name, value in getattr(module, "module").__dict__.items():
            if self.SPIPE_INFIX in name:
                setattr(
                    module, name, value
                )  # copy attributes with name xx${SPIPE_INFIX}xx to the wrapper module
