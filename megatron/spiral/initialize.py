import torch

from megatron.spiral.debug import spiral_print

import spiral_helper

global thunder_group


class SpiralBackend:
    def __init__(self, ranks):
        global thunder_group
        thunder_group = spiral_helper.Comm(sorted(ranks))


def get_thunder_group():
    assert thunder_group is not None, "thunder_group is not initialized"
    return thunder_group
