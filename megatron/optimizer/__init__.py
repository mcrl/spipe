# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import warnings

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args
from megatron.spiral.init_context import SpiralParamStatus
from megatron.spiral.utils import is_spiral_param
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from megatron.spiral.cpu_adam import SpiralCPUAdam

from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.spiral.optimizer import SpiralStageOptimizer


def get_param_groups(modules,
                     no_weight_decay_cond,
                     scale_lr_cond,
                     lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate.
    """
    args = get_args()

    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []

    for module in modules:

        # NOTE (SpiralPipe) Skip module that is not bwd stage. A module with both fwd / bwd stage id is not skipped. (e.g., SpiralPipe w/o remapping)
        if args.spiral:
            if not hasattr(module, "spiral_backward_stage_id"):
                warnings.warn("SpiralPipe module should have spiral_backward_stage_id attr even if it is None (refer to _post_init_method in init_context.py). Lacking it highly suggests a critical bug.")
                continue
            if hasattr(module, "spiral_backward_stage_id") and getattr(module, "spiral_backward_stage_id") is None:
                continue

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if args.spiral:
                assert (is_spiral_param(param))
                if param.spiral_status == SpiralParamStatus.REMOTE:
                    param = param.spiral_tensor

            if no_weight_decay_cond is not None:
                if args.spiral:
                    warnings.warn("no_weight_decay_cond is not supported in SpiralPipe. Consequencies are not known.")
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                if args.spiral:
                    warnings.warn("scale_lr_cond is not supported in SpiralPipe. Consequencies are not known.")
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
    args = get_args()

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if (
        args.spiral
        and args.spiral_stage_optimizer
        and args.spiral_backward_virtual_size > 1
    ):
        if not hasattr(model, "_spiral_optimizer_entered"):
            # NOTE (SpiralPipe) top level model[], recursively collect optimizer for each **BWD** stage. FWD stages are skipped, even though they will naturally be skipped due to logic in get_param_groups, in order to prevent optimizer with empty param group.
            optimizer_list = []
            for bwd_stage_id in range(args.spiral_backward_virtual_size):
                # NOTE (SpiralPipe) SpiralStageOptimizer requires optimizer list to be sorted in **ascending** order of bwd stage id. SpiralPipe w/o remapping has stage models that have both fwd/bwd id and hence in opposite bwd stage order w.r.t SpiralPipe w/ remapping.
                if args.spiral_remap:
                    _idx_bwd_stage_id = -bwd_stage_id - 1
                else:
                    _idx_bwd_stage_id = bwd_stage_id
                setattr(model[_idx_bwd_stage_id], "_spiral_optimizer_entered", True)
                optimizer_list.append(
                    get_megatron_optimizer(
                        model[_idx_bwd_stage_id],
                        no_weight_decay_cond,
                        scale_lr_cond,
                        lr_mult,
                    )
                )
            return SpiralStageOptimizer(
                optimizer_list,
                args.clip_grad,
                args.log_num_zeros_in_grad,
                params_have_main_grad,
                args.use_contiguous_buffers_in_local_ddp,
                model,
            )
        else:
            # NOTE (SpiralPipe) second level model of DDP class
            # Wrap itself in order to work as an iterable of size 1

            # TODO (SpiralPipe) SpiralStageOptimizer should be refactored inheritance in order to allow cleaner logic here (with L149)
            # delattr(model, "_spiral_optimizer_entered")
            model = [model]

    # Base optimizer.
    param_groups = get_param_groups(model,
                                    no_weight_decay_cond,
                                    scale_lr_cond,
                                    lr_mult)

    if args.spiral:
        assert args.optimizer == 'adam', 'SpiralPipe only support Adam'
        optimizer = SpiralCPUAdam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
        # optimizer = DeepSpeedCPUAdam(
        #     param_groups,
        #     lr=args.lr,
        #     weight_decay=args.weight_decay,
        #     betas=(args.adam_beta1, args.adam_beta2),
        #     eps=args.adam_eps,
        # )
        # TODO (SpiralPipe) SpiralStageOptimizer should be refactored inheritance in order to allow cleaner logic here
        if args.spiral_stage_optimizer:
            _unwrapped_model = model[0]
            assert hasattr(_unwrapped_model, "_spiral_optimizer_entered")
            delattr(_unwrapped_model, "_spiral_optimizer_entered")
            return optimizer
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
        if args.spiral:
            raise RuntimeError("SpiralPipe currently only supports FP32 optimizer.")
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
        opt_ty = DistributedOptimizer \
            if args.use_distributed_optimizer else \
            Float16OptimizerWithFloat16Params
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
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad,
                         args.use_contiguous_buffers_in_local_ddp,
                         model)
