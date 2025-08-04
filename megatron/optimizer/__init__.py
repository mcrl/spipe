# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import warnings

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args
from megatron.spipe.init_context import SPipeParamStatus
from megatron.spipe.utils import is_spipe_param

from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.spipe.optimizer.stage_optimizer import SPipeStageOptimizer
from megatron.spipe.optimizer.optimizer import SPipeFloat16Optimizer, DeepSpeedFloat16Optimizer, SPipeFP32Optimizer


def get_param_groups(modules,
                     no_weight_decay_cond,
                     scale_lr_cond,
                     lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate.

       if spipe, only the backward stages' params are considered.
    """
    args = get_args()

    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []

    for module in modules:

        # NOTE (SPipe) Skip module that is not bwd stage. A module with both fwd / bwd stage id is not skipped. (e.g., SPipe w/o remapping)
        if args.spipe:
            if not hasattr(module, "spipe_backward_stage_id"):
                warnings.warn("SPipe module should have spipe_backward_stage_id attr even if it is None (refer to _post_init_method in init_context.py). Lacking it highly suggests a critical bug.")
                continue
            if hasattr(module, "spipe_backward_stage_id") and getattr(module, "spipe_backward_stage_id") is None:
                continue

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            # TODO: This is for DeepSpeedCPUOptimizer which will be deprecated.
            if args.spipe and not args.spipe_stage_optimizer:
                # Only params converted to spipe param and currently placed on local CPU memory should enter,
                # as it is the necessary condition for backward stages
                assert (is_spipe_param(param))
                assert (param.spipe_status == SPipeParamStatus.CPU)
                assert (param.spipe_tensor.numel() == param.spipe_numel)
                param = param.spipe_tensor

            if no_weight_decay_cond is not None:
                if args.spipe:
                    warnings.warn("no_weight_decay_cond is not supported in SPipe. Consequencies are not known.")
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                shape = param.spipe_shape if args.spipe and args.spipe_stage_optimizer else param.shape
                no_wd = name.endswith(".bias") or len(shape) == 1

            if scale_lr_cond is not None:
                if args.spipe:
                    warnings.warn("scale_lr_cond is not supported in SPipe. Consequencies are not known.")
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif not no_wd and scale_lr:
                wd_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)
            else:
                no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append({'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups

def get_megatron_optimizer(model,
                           no_weight_decay_cond=None,
                           scale_lr_cond=None,
                           lr_mult=1.0):
    """
    - FP16 megatron optimizer : Float16OptimizerWithFloat16Params(FusedAdam)
    - FP32 megatron optimizer : FP32Optimizer(FusedAdam)
    - FP16 no staged optimizer : DeepSpeedFloat16Optimizer(DeepSpeedCPUAdam)
    - FP32 no staged optimizer : FP32Optimizer(DeepSpeedCPUAdam)
    - FP16 staged optimizer : SPipeStageOptimizer(...SPipeFloat16Optimizer(SPipeCPUAdam | SPipeGPUAdam | SPipeGPUChunkedAdam))
    - FP16 staged optimizer : SPipeStageOptimizer(...SPipeFP32Optimizer(SPipeCPUAdam | SPipeGPUAdam | SPipeGPUChunkedAdam))
    """
    args = get_args()

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if (
        args.spipe
        and args.spipe_stage_optimizer
    ):
        if not hasattr(model, "_spipe_optimizer_entered"):
            # NOTE (SPipe) top level model[], recursively collect optimizer for each **BWD** stage. FWD stages are skipped, even though they will naturally be skipped due to logic in get_param_groups, in order to prevent optimizer with empty param group.
            bwd_stage_optimizers = []
            for bwd_stage_id in range(args.spipe_backward_virtual_size):
                # NOTE (SPipe) SPipeStageOptimizer requires optimizer list to be sorted in **ascending** order of bwd stage id. SPipe w/o remapping has stage models that have both fwd/bwd id and hence in opposite bwd stage order w.r.t SPipe w/ remapping.
                if args.spipe_remap:
                    _idx_bwd_stage_id = -bwd_stage_id - 1
                else:
                    _idx_bwd_stage_id = bwd_stage_id
                # Attach a flag to second level modules (i.e., stage model) to stop recursive call
                setattr(model[_idx_bwd_stage_id], "_spipe_optimizer_entered", True)
                # Insert optimizer for each bwd stage
                opt_ty = get_megatron_optimizer(
                    model[_idx_bwd_stage_id],
                    no_weight_decay_cond,
                    scale_lr_cond,
                    lr_mult,
                )
                bwd_stage_optimizers.append(opt_ty)
            return SPipeStageOptimizer(bwd_stage_optimizers)
        else:
            # NOTE (SPipe) second level model of DDP class
            # Wrap itself in order to work as an iterable of size 1
            # This is necessary for backward compatibility with MegatronOptimizer, as well as get_param_groups
            model = [model]

    # Base optimizer.
    param_groups = get_param_groups(model,
                                    no_weight_decay_cond,
                                    scale_lr_cond,
                                    lr_mult)

    if args.spipe:
        assert args.optimizer == 'adam', 'SPipe only support Adam'

        if args.spipe_stage_optimizer:
            _unwrapped_model = model[0]
            assert hasattr(_unwrapped_model, "_spipe_optimizer_entered")
            delattr(_unwrapped_model, "_spipe_optimizer_entered")

            # NOTE (SPipe) spipe stage optimizer uses SPipeCPUAdam, which overlaps weight update with upstream bwd stage computation.
            # If --spipe-heterogeneous-optimizer is enabled, apply gpu optimizer to first stage.
            if args.spipe_heterogeneous_optimizer and _unwrapped_model.spipe_backward_stage_id == 0:
                if args.spipe_offload_optimizer:
                    from megatron.spipe.optimizer.gpu_chunked_adam import SPipeGPUChunkedAdam
                    inner_opt_ty = SPipeGPUChunkedAdam
                else:
                    from megatron.spipe.optimizer.gpu_adam import SPipeGPUAdam
                    inner_opt_ty = SPipeGPUAdam
            else:
                from megatron.spipe.optimizer.cpu_adam import SPipeCPUAdam
                inner_opt_ty = SPipeCPUAdam
                # NOTE (SPipe) `params` here annotates the "optimizer params". It is not always the same as the params referenced during training.
                # For SPipe, the optimizer params are the offloaded params and hence should only have grad field, while the params referred during can have both main_grad and grad field.
                # Hence, setting `params_have_main_grad` to True incurs explicit copy from main_grad into grad (optimizer.step() calls it), which is not necessary.
                params_have_main_grad = False
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            inner_opt_ty = DeepSpeedCPUAdam
            params_have_main_grad = False

        optimizer = inner_opt_ty(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    else:
        if args.optimizer == 'adam':
            optimizer = Adam(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps)
        elif args.optimizer == 'sgd':
            optimizer = SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.sgd_momentum)
        else:
            raise Exception('{} optimizer is not supported.'.format(
                args.optimizer))

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if args.fp16 or args.bf16 or args.use_distributed_optimizer:
        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)

        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        if args.use_distributed_optimizer:
            opt_ty = DistributedOptimizer
        elif args.spipe:
            opt_ty = SPipeFloat16Optimizer
            if not args.spipe_stage_optimizer:
                opt_ty = DeepSpeedFloat16Optimizer
        else:
            opt_ty = Float16OptimizerWithFloat16Params

        return opt_ty(optimizer,
                    args.clip_grad,
                    args.log_num_zeros_in_grad,
                    params_have_main_grad,
                    args.use_contiguous_buffers_in_local_ddp,
                    args.fp16,
                    args.bf16,
                    args.params_dtype,
                    grad_scaler,
                    model)

    # FP32.
    opt_ty = SPipeFP32Optimizer if args.spipe and args.spipe_stage_optimizer else FP32Optimizer
    return opt_ty(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad,
                         args.use_contiguous_buffers_in_local_ddp,
                         model)
