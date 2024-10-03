from .initialize import (
    SpiralBackend,
    get_thunder_group,
    get_thunder_cuda_manager,
)
from .init_context import SpiralInitContext, SpiralParamStatus
from .wrapper_init_context import SpiralWrapperInitContext
from .generic import ContextManagers
from .p2p_communication import (
    recv_input_tensor,
    send_output_tensor,
    recv_output_tensor_grad,
    send_input_tensor_grad,
)
from .schedule import (
    forward_backward_pipelining_with_spiral_remap,
    forward_backward_pipelining_with_spiral,
)
from .ckpt_schedule import (
    CkptSendRecvOp,
    CkptSendRecvType,
    CkptSendRecvSchedule,
)
from .ckpt_communication import (
    comm_input_ckpt,
)
from .optimizer import (
    SpiralStageOptimizer,
    SpiralStageOptimizerParamScheduler,
)
from .module import SpiralPhaseList
from .cpu_adam import SpiralCPUAdam

from .build_state import (
    reset_spiral_forward_stage_build_phase_num_spiral_params_allocated,
    reset_spiral_backward_stage_build_phase_num_spiral_params_allocated,
    get_spiral_forward_stage_build_phase,
    set_spiral_forward_stage_build_phase,
    get_spiral_backward_stage_build_phase,
    set_spiral_backward_stage_build_phase,
    get_spiral_forward_stage_build_phase_size,
    get_spiral_backward_stage_build_phase_size,
    get_spiral_total_build_phase_size,
    get_spiral_aggregate_num_spiral_params,
    get_spiral_num_spiral_params,
    get_spiral_global_build_phase,
    get_spiral_global_build_phase_num_spiral_params_dict,
    set_spiral_global_build_phase_num_spiral_params_dict,
    get_pp_rank_for_fwd_phase,
    fwd_phase2local_stage_phase,
    destroy_spiral_build_state,
)

from .debug import (
    debug_extract_module_and_param_names,
    debug_module2name,
    debug_module2name_id,
    debug_module2name_class,
    debug_module2class_id,
    debug_param2name,
    debug_param2name_id,
    debug_param2name_id_shape,
    debug_param2name_id_shape_device,
    debug_param2name_id_numel,
    debug_param2name_id_shape_status,
    debug_param2id_shape_status,
    printflock,
    spiral_print,
    spiral_report_memory,
)

from .utils import (
    is_spiral_param,
    num_spiral_params,
    lcm,
)

from .test import (
    test_spiral_report_memory,
    test_spiral_cuda_manager,
)
