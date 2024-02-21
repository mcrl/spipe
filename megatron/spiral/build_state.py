import numpy as np

from megatron.core import mpu
from megatron.spiral.utils import lcm

""" Spiral states only used and valid during model initialization. """
_SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE = None
_SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE = None

_SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT = None
_SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED = None
_SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED = None


def initialize_spiral_build_state():
    global _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED
    _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT = {
        s: 0 for s in range(get_spiral_pipeline_parallel_total_build_phase_size() * mpu.get_pipeline_model_parallel_world_size())
    }
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED = np.zeros(
        (
            mpu.get_spiral_pipeline_parallel_forward_virtual_size(),
            get_spiral_pipeline_parallel_forward_stage_build_phase_size(),
        ),
        dtype=np.uintc,
    )
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED = np.zeros(
        (
            mpu.get_spiral_pipeline_parallel_backward_virtual_size(),
            get_spiral_pipeline_parallel_backward_stage_build_phase_size(),
        ),
        dtype=np.uintc,
    )


def get_add_spiral_pipeline_parallel_next_param_number_to_build(incr=1):
    """Get next param number to build"""
    if _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE is not None:
        num_params_allocated = _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED[
            mpu.get_spiral_pipeline_parallel_forward_virtual_rank(),
            _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE,
        ]
        _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED[
            mpu.get_spiral_pipeline_parallel_forward_virtual_rank(),
            _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE,
        ] += incr
    if _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE is not None:
        num_params_allocated = _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED[
            mpu.get_spiral_pipeline_parallel_backward_virtual_rank(),
            _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE,
        ]
        _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE_NUM_SPIRAL_PARAMS_ALLOCATED[
            mpu.get_spiral_pipeline_parallel_backward_virtual_rank(),
            _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE,
        ] += incr
    return get_spiral_pipeline_parallel_aggregate_num_spiral_params() + num_params_allocated


def get_spiral_pipeline_parallel_forward_stage_build_phase():
    """Current phase of forward stage build"""
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
    return _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE


def set_spiral_pipeline_parallel_forward_stage_build_phase(phase):
    """Set phase of forward stage build"""
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
    assert (
        _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE is None
    ), "Can't set forward stage build phase when backward stage build phase is not None"
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE = phase


def get_spiral_pipeline_parallel_backward_stage_build_phase():
    """Current phase of backward stage build"""
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
    return _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE


def set_spiral_pipeline_parallel_backward_stage_build_phase(phase):
    """Set phase of backward stage build"""
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
    assert (
        _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE is None
    ), "Can't set backward stage build phase when forward stage build phase is not None"
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE = phase


def get_spiral_pipeline_parallel_forward_stage_build_phase_size():
    """Number of phases to build for a forward stage"""
    return (
        get_spiral_pipeline_parallel_total_build_phase_size()
        // mpu.get_spiral_pipeline_parallel_forward_virtual_size()
    )


def get_spiral_pipeline_parallel_backward_stage_build_phase_size():
    """Number of phases to build for a backward stage"""
    return (
        get_spiral_pipeline_parallel_total_build_phase_size()
        // mpu.get_spiral_pipeline_parallel_backward_virtual_size()
    )


def get_spiral_pipeline_parallel_total_build_phase_size():
    """Number of total phases to build for a pass"""
    return lcm(
        mpu.get_spiral_pipeline_parallel_forward_virtual_size(),
        mpu.get_spiral_pipeline_parallel_backward_virtual_size(),
    )


def get_spiral_pipeline_parallel_aggregate_num_spiral_params():
    """Get the aggregate number of spiral parameters before the current build phase"""
    global _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT
    assert (
        _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT is not None
    ), "Number of spiral parameters for each phase is not set. "
    return sum(
        _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT[i]
        for i in range(get_spiral_pipeline_parallel_global_build_phase())
    )


def get_spiral_pipeline_parallel_num_spiral_params():
    """Get the number of spiral parameters for the current build phase

    Returns None if build phase is not set or the value in dict is not set."""
    global _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT
    assert (
        _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT is not None
    ), "Number of spiral parameters for each phase is not set. "
    return _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT.get(
        get_spiral_pipeline_parallel_global_build_phase()
    )


def get_spiral_pipeline_parallel_global_build_phase():
    """Get the global build phase number"""
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
    if _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE is not None:
        return (
            get_spiral_pipeline_parallel_forward_stage_build_phase_size()
            * mpu.get_pipeline_model_parallel_world_size()
            * mpu.get_spiral_pipeline_parallel_forward_virtual_rank()
            + get_spiral_pipeline_parallel_forward_stage_build_phase_size()
            * mpu.get_pipeline_model_parallel_rank()
            + _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
        )
    elif _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE is not None:
        return (
            get_spiral_pipeline_parallel_backward_stage_build_phase_size()
            * mpu.get_pipeline_model_parallel_world_size()
            * mpu.get_spiral_pipeline_parallel_backward_virtual_rank()
            + get_spiral_pipeline_parallel_backward_stage_build_phase_size()
            * (mpu.get_pipeline_model_parallel_world_size() - mpu.get_pipeline_model_parallel_rank() - 1)
            + _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
        )
    else:
        raise RuntimeError("No build phase is set")


def get_spiral_pipeline_parallel_global_build_phase_num_spiral_params_dict():
    """Get the number of spiral parameters for each phase"""
    global _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT
    assert (
        _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT is not None
    ), "Number of spiral parameters for each phase is not set. "
    return _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT


def set_spiral_pipeline_parallel_global_build_phase_num_spiral_params_dict(dict):
    """Set the number of spiral parameters for each phase

    In this dict, the "phase" key is not local but global w.r.t. the entire pipeline ranks.
    So, querying the dict should be done with the calculated global phase number.
    """
    global _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT
    _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT = dict


# NOTE (mcrl): currently has no caller
def destroy_spiral_build_state():
    global _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE
    global _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE
    _SPIRAL_PIPELINE_PARALLEL_FORWARD_STAGE_BUILD_PHASE = None
    _SPIRAL_PIPELINE_PARALLEL_BACKWARD_STAGE_BUILD_PHASE = None

    global _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT
    _SPIRAL_PIPELINE_PARALLEL_GLOBAL_BUILD_PHASE_NUM_SPIRAL_PARAMS_DICT = None
