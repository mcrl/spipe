import torch

from megatron import get_args
from megatron.core.utils import get_attr_wrapped_model
from megatron.model.module import MegatronModule


class SpiralPhaseList(MegatronModule):
    def __init__(self, modules):
        args = get_args()
        super(SpiralPhaseList, self).__init__(
            share_word_embeddings=not args.untie_embeddings_and_output_weights
        )
        self.module_list = torch.nn.ModuleList(modules)

    def set_input_tensor(self, input_tensor):
        if len(self.module_list) > 0:
            set_input_tensor = get_attr_wrapped_model(
                self.module_list[0], "set_input_tensor"
            )
            set_input_tensor(input_tensor)

    def forward(self, *args, **kwargs):
        for idx, module in enumerate(self.module_list):
            # Run forward of each phase module
            output_tensor = module(*args, **kwargs)
            # put output_tensor to next phase's input_tensor
            if idx + 1 < len(self.module_list):
                if not isinstance(output_tensor, list):
                    output_tensor = [output_tensor]
                set_input_tensor = get_attr_wrapped_model(
                    self.module_list[idx + 1], "set_input_tensor"
                )
                set_input_tensor(output_tensor)

        return output_tensor
