from collections import deque

import torch

from megatron import get_args, get_num_microbatches
from megatron.core.utils import get_attr_wrapped_model
from megatron.model.module import MegatronModule

from megatron.spipe.debug import spipe_print


class SPipePhaseList(MegatronModule):

    def __init__(
        self,
        modules,
        save_input_tensors=False,
        save_output_tensors=False,
    ):
        """A wrapper for a list of modules that have the same stage number but built on different build phase.

        Must be initialized under SPipeInitContext in order to have spipe attributes set properly, and have spipe functions that propagate to the wrapped modules.

        Modules in `modules` must be initialized in this ctor, in order to correctly walk through the recursive init process of SPipeInitContext.

        The ctor takes the following arguments:

        modules (required): list of modules to be wrapped

        save_input_tensors (optional): whether to save input tensors of each phase module

        save_output_tensors (optional): whether to save output tensors of each phase module
        """
        args = get_args()
        super(SPipePhaseList, self).__init__(
            share_word_embeddings=not args.untie_embeddings_and_output_weights
        )
        self.module_list = torch.nn.ModuleList(modules)

        for module in self.module_list:
            if save_input_tensors:
                module.spipe_input_tensors = deque(maxlen=get_num_microbatches())
            else:
                module.spipe_input_tensors = None
            if save_output_tensors:
                module.spipe_output_tensors = deque(maxlen=get_num_microbatches())
            else:
                module.spipe_output_tensors = None

    def forward(self, *args, **kwargs):
        for idx, module in enumerate(self.module_list):
            # Run forward of each phase module
            # For phase 0, input tensor is already set from megatron/core/pipeline_parallel/schedules.py L224 set_input_tensor()
            output_tensor = module(*args, **kwargs)
            self._set_phase_output_tensor(idx, output_tensor)
            if idx + 1 < len(self.module_list):
                self._set_phase_input_tensor(idx + 1, output_tensor)
        return output_tensor

    # NOTE (SPipe) Do not rename this function
    def set_input_tensor(self, input_tensor):
        if len(self.module_list) > 0:
            self._set_phase_input_tensor(0, input_tensor)

    def _set_phase_input_tensor(self, phase_id, input_tensor):
        # wrap input_tensor as list is required by TransformerLanguageModel.set_input_tensor
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        # Set input tensor of the phase module
        set_input_tensor = get_attr_wrapped_model(
            self.module_list[phase_id], "set_input_tensor"
        )
        set_input_tensor(input_tensor)
        # Save unwrapped input tensor ckpt
        # TODO (SPipe) Current POC saves all input tensors, where it can be optimized to save only the necessary input tensor ckpts
        if hasattr(self.module_list[phase_id], "spipe_input_tensors") and isinstance(
            getattr(self.module_list[phase_id], "spipe_input_tensors"), deque
        ):
            self.module_list[phase_id].spipe_input_tensors.append(input_tensor[0])

    def _set_phase_output_tensor(self, phase_id, output_tensor):
        if hasattr(self.module_list[phase_id], "spipe_output_tensors") and isinstance(
            getattr(self.module_list[phase_id], "spipe_output_tensors"), deque
        ):
            self.module_list[phase_id].spipe_output_tensors.append(output_tensor)

    def empty_input_tensors(self):
        for module in self.module_list:
            if hasattr(module, "spipe_input_tensors") and isinstance(
                getattr(module, "spipe_input_tensors"), deque
            ):
                module.spipe_input_tensors.clear()

    def empty_output_tensors(self):
        for module in self.module_list:
            if hasattr(module, "spipe_output_tensors") and isinstance(
                getattr(module, "spipe_output_tensors"), deque
            ):
                module.spipe_output_tensors.clear()

    def __getitem__(self, idx):
        return self.module_list[idx]