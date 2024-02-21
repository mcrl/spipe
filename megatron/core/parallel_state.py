# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import torch
from typing import Optional

from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
_SPIRAL_EMBEDDING_GROUP = None
_SPIRAL_EMBEDDING_GROUP_GLOO = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
_SPIRAL_POSITION_EMBEDDING_GROUP = None
_SPIRAL_POSITION_EMBEDDING_GROUP_GLOO = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# FP8 amax reduction group.
_AMAX_REDUCTION_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None
_SPIRAL_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None
_SPIRAL_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# Whether spiral pipeline parallel is enable or not
_SPIRAL_PIPELINE_PARALLEL = None
_SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK = None # should set rank value only when active
_SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK = None # should set rank value only when active
_SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE = None
_SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE = None

# Below spiral variables are lazily set after spiral backend init using CommInfo
_SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE = None
_SPIRAL_PIPELINE_PARALLEL_INTRA_RANK = None
_SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    spiral_pipeline_parallel: bool = False,
    spiral_pipeline_parallel_forward_virtual_size: Optional[int] = None,
    spiral_pipeline_parallel_backward_virtual_size: Optional[int] = None,
    use_fp8: bool = False,
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        spiral_pipeline_parallel (bool, default = False):
            Whether to use spiral pipeline parallel.

        spiral_pipeline_parallel_forward_virtual_size (int, optional):
            The number of forward virtual stages that each spiral pipeline rank will have

        spiral_pipeline_parallel_backward_virtual_size (int, optional):
            The number of backward virtual stages that each spiral pipeline rank will have

        use_fp8 (bool, default = False):
            Construct GPU groups needed for FP8 training, namely for
            amax reduction across the product of the data-parallel and
            tensor-parallel groups.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    data_parallel_size: int = world_size // (tensor_model_parallel_size *
                                             pipeline_model_parallel_size)

    num_tensor_model_parallel_groups: int  = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError("pipeline-model-parallel size should be greater than 2 with "
                               "interleaved schedule")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if spiral_pipeline_parallel:
        if virtual_pipeline_model_parallel_size is not None:
            raise RuntimeError("virtual pipeline parallel is not compatible with spiral pipeline parallel")
        global _SPIRAL_PIPELINE_PARALLEL
        _SPIRAL_PIPELINE_PARALLEL = True
        global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK
        global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK
        global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE
        global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE
        _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK = None
        _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK = None
        if spiral_pipeline_parallel_forward_virtual_size is not None:
            _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE = spiral_pipeline_parallel_forward_virtual_size
        if spiral_pipeline_parallel_backward_virtual_size is not None:
            _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE = spiral_pipeline_parallel_backward_virtual_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _SPIRAL_EMBEDDING_GROUP
    global _SPIRAL_EMBEDDING_GROUP_GLOO
    global _EMBEDDING_GLOBAL_RANKS
    global _SPIRAL_EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _SPIRAL_POSITION_EMBEDDING_GROUP
    global _SPIRAL_POSITION_EMBEDDING_GROUP_GLOO
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    global _SPIRAL_POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank],
                                       ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank]]
            if _SPIRAL_PIPELINE_PARALLEL:
                spiral_embedding_ranks = [ranks[0], ranks[-1]]
                spiral_position_embedding_ranks = [ranks[0], ranks[-1]] # since forward and backward both have a prep stage
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        if _SPIRAL_PIPELINE_PARALLEL:
            group = torch.distributed.new_group(spiral_embedding_ranks)
            group_gloo = torch.distributed.new_group(spiral_embedding_ranks, backend="gloo")
            if rank in spiral_embedding_ranks:
                _SPIRAL_EMBEDDING_GROUP = group
                _SPIRAL_EMBEDDING_GROUP_GLOO = group_gloo
            if rank in ranks:
                _SPIRAL_EMBEDDING_GLOBAL_RANKS = spiral_embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        if _SPIRAL_PIPELINE_PARALLEL:
            group = torch.distributed.new_group(spiral_position_embedding_ranks)
            group_gloo = torch.distributed.new_group(spiral_position_embedding_ranks, backend="gloo")
            if rank in spiral_position_embedding_ranks:
                _SPIRAL_POSITION_EMBEDDING_GROUP = group
                _SPIRAL_POSITION_EMBEDDING_GROUP_GLOO = group_gloo
            if rank in ranks:
                _SPIRAL_POSITION_EMBEDDING_GLOBAL_RANKS = spiral_position_embedding_ranks

    # Build the FP8 groups.
    global _AMAX_REDUCTION_GROUP
    assert _AMAX_REDUCTION_GROUP is None, \
        'FP8 amax reduction group is already initialized'
    if use_fp8:
        amax_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_amax_groups: int = world_size // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _AMAX_REDUCTION_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo():
    """Get the data parallel group-gloo the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP_GLOO is not None, \
        'data parallel group-gloo is not initialized'
    return _DATA_PARALLEL_GROUP_GLOO


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_spiral_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _SPIRAL_EMBEDDING_GROUP is not None, \
        'spiral embedding group is not initialized'
    return _SPIRAL_EMBEDDING_GROUP


def get_spiral_embedding_group_gloo():
    """Get the embedding group-gloo the caller rank belongs to."""
    assert _SPIRAL_EMBEDDING_GROUP_GLOO is not None, \
        'spiral embedding group-gloo is not initialized'
    return _SPIRAL_EMBEDDING_GROUP_GLOO


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, \
        'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_spiral_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _SPIRAL_POSITION_EMBEDDING_GROUP is not None, \
        'spiral position embedding group is not initialized'
    return _SPIRAL_POSITION_EMBEDDING_GROUP


def get_spiral_position_embedding_group_gloo():
    """Get the position embedding group-gloo the caller rank belongs to."""
    assert _SPIRAL_POSITION_EMBEDDING_GROUP_GLOO is not None, \
        'spiral position embedding group-gloo is not initialized'
    return _SPIRAL_POSITION_EMBEDDING_GROUP_GLOO


def get_amax_reduction_group():
    """Get the FP8 amax reduction group the caller rank belongs to."""
    assert _AMAX_REDUCTION_GROUP is not None, \
        'FP8 amax reduction group is not initialized'
    return _AMAX_REDUCTION_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size

def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""

    # For spiral pipeline, lowest idx forward virtual stage and lowest idx backward virtual stage is the first stage
    if is_spiral_pipeline_parallel():
        _is_forward_virtual_first_stage = get_spiral_pipeline_parallel_forward_virtual_rank() == 0 and \
        get_pipeline_model_parallel_rank() == 0
        _is_backward_virtual_first_stage = get_spiral_pipeline_parallel_backward_virtual_rank() == 0 and \
        get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_world_size() - 1
        return _is_forward_virtual_first_stage or _is_backward_virtual_first_stage

    # NOTE (mcrl) below is original logic
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""

    # For spiral pipeline, highest idx forward virtual stage and highest idx backward virtual stage is the last stage
    if is_spiral_pipeline_parallel():
        _is_forward_virtual_last_stage = get_spiral_pipeline_parallel_forward_virtual_rank() == get_spiral_pipeline_parallel_forward_virtual_size() - 1 and \
        get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_world_size() - 1
        _is_backward_virtual_last_stage = get_spiral_pipeline_parallel_backward_virtual_rank() == get_spiral_pipeline_parallel_backward_virtual_size() - 1 and \
        get_pipeline_model_parallel_rank() == 0
        return _is_forward_virtual_last_stage or _is_backward_virtual_last_stage

    # NOTE (mcrl) below is original logic
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and \
            is_pipeline_stage_after_split(rank+1)


def is_spiral_pipeline_parallel():
    """Return true if spiral pipeline parallel is enabled."""
    if _SPIRAL_PIPELINE_PARALLEL is None:
        return False
    return True


def is_spiral_pipeline_parallel_forward_stage():
    """Return true if current stage is a forward stage in spiral pipeline parallel."""
    if not is_spiral_pipeline_parallel():
        raise RuntimeError("is_spiral_pipeline_parallel_forward_stage is only valid when spiral pipeline parallel is enabled")
    return get_spiral_pipeline_parallel_forward_virtual_rank() is not None and \
        get_spiral_pipeline_parallel_backward_virtual_rank() is None


def is_spiral_pipeline_parallel_backward_stage():
    """Return true if current stage is a backward stage in spiral pipeline parallel."""
    if not is_spiral_pipeline_parallel():
        raise RuntimeError("is_spiral_pipeline_parallel_backward_stage is only valid when spiral pipeline parallel is enabled")
    return get_spiral_pipeline_parallel_backward_virtual_rank() is not None and \
        get_spiral_pipeline_parallel_forward_virtual_rank() is None


def get_spiral_pipeline_parallel_forward_virtual_rank():
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK
    return _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK


def set_spiral_pipeline_parallel_forward_virtual_rank(rank):
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK = rank


def get_spiral_pipeline_parallel_backward_virtual_rank():
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK
    return _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK


def set_spiral_pipeline_parallel_backward_virtual_rank(rank):
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK = rank


def get_spiral_pipeline_parallel_forward_virtual_size():
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE
    return _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE


def set_spiral_pipeline_parallel_forward_virtual_size(size):
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE = size


def get_spiral_pipeline_parallel_backward_virtual_size():
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE
    return _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE


def set_spiral_pipeline_parallel_backward_virtual_size(size):
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_SIZE = size


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the virtual pipeline-parallel world size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, \
        "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]

def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER

def get_spiral_pipeline_parallel_intra_rank():
    assert _SPIRAL_PIPELINE_PARALLEL_INTRA_RANK is not None
    return _SPIRAL_PIPELINE_PARALLEL_INTRA_RANK

def set_spiral_pipeline_parallel_intra_rank(rank):
    global _SPIRAL_PIPELINE_PARALLEL_INTRA_RANK
    _SPIRAL_PIPELINE_PARALLEL_INTRA_RANK = rank

def get_spiral_pipeline_parallel_intra_size():
    assert _SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE is not None
    return _SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE

def set_spiral_pipeline_parallel_intra_size(size):
    global _SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE
    _SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE = size

def get_spiral_pipeline_parallel_is_host_leader():
    assert _SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER is not None
    return _SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER

def set_spiral_pipeline_parallel_is_host_leader(is_leader: bool):
    global _SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER
    _SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER = is_leader


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_GLOO
    _DATA_PARALLEL_GROUP_GLOO = None

    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _EMBEDDING_GLOBAL_RANKS
    _EMBEDDING_GLOBAL_RANKS = None
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    _POSITION_EMBEDDING_GLOBAL_RANKS = None

    global _AMAX_REDUCTION_GROUP
    _AMAX_REDUCTION_GROUP = None

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

    # SpiralPipe
    global _SPIRAL_PIPELINE_PARALLEL
    _SPIRAL_PIPELINE_PARALLEL = None
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_RANK = None
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_RANK = None
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_VIRTUAL_SIZE = None
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_WORLD_SIZE
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_VIRTUAL_WORLD_SIZE = None

    global _SPIRAL_EMBEDDING_GROUP
    _SPIRAL_EMBEDDING_GROUP = None
    global _SPIRAL_EMBEDDING_GROUP_GLOO
    _SPIRAL_EMBEDDING_GROUP_GLOO = None
    global _SPIRAL_POSITION_EMBEDDING_GROUP
    _SPIRAL_POSITION_EMBEDDING_GROUP = None
    global _SPIRAL_POSITION_EMBEDDING_GROUP_GLOO
    _SPIRAL_POSITION_EMBEDDING_GROUP_GLOO = None
    global _SPIRAL_EMBEDDING_GLOBAL_RANKS
    _SPIRAL_EMBEDDING_GLOBAL_RANKS = None
    global _SPIRAL_POSITION_EMBEDDING_GLOBAL_RANKS
    _SPIRAL_POSITION_EMBEDDING_GLOBAL_RANKS = None

    global _SPIRAL_PIPELINE_PARALLEL_INTRA_RANK
    _SPIRAL_PIPELINE_PARALLEL_INTRA_RANK = None
    global _SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE
    _SPIRAL_PIPELINE_PARALLEL_INTRA_SIZE = None
    global _SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER
    _SPIRAL_PIPELINE_PARALLEL_IS_HOST_LEADER = None