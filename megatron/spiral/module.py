from collections import deque

import torch

from megatron import get_args, get_num_microbatches
from megatron.core.utils import get_attr_wrapped_model
from megatron.model.module import MegatronModule

from megatron.spiral.debug import spiral_print


class SpiralPhaseList(MegatronModule):
    def __init__(self, modules):
        args = get_args()
        super(SpiralPhaseList, self).__init__(
            share_word_embeddings=not args.untie_embeddings_and_output_weights
        )
        self.module_list = torch.nn.ModuleList(modules)

        for module in self.module_list:
            if not hasattr(module, "spiral_input_tensor_ckpts"):
                module.spiral_input_tensor_ckpts = deque(maxlen=get_num_microbatches())

    def forward(self, *args, **kwargs):
        for idx, module in enumerate(self.module_list):
            # Run forward of each phase module
            output_tensor = module(*args, **kwargs)
            # Set input tensor of next phase module
            if idx + 1 < len(self.module_list):
                self._set_phase_input_tensor(idx + 1, output_tensor)
        return output_tensor

    def set_input_tensor(self, input_tensor):
        if len(self.module_list) > 0:
            self._set_phase_input_tensor(0, input_tensor)

    def empty_input_tensor_ckpts(self):
        for module in self.module_list:
            module.spiral_input_tensor_ckpts.clear()

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
        # TODO (mcrl): Current POC saves all input tensors, where it can be optimized to save only the necessary input tensor ckpts
        # TODO (mcrl): Skip when forward_only
        self.module_list[phase_id].spiral_input_tensor_ckpts.append(input_tensor[0])

    def __getitem__(self, idx):
        return self.module_list[idx]
