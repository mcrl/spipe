#!/bin/bash

DS_CONFIG="slurm-$SLURM_JOB_ID-ds_config.json"
PROFILE_OUTPUT="slurm-$SLURM_JOB_ID-profile.out"

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GBS,
  "train_micro_batch_size_per_gpu": $MBS,
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true
  },
  "bf16": {
    "enabled": false
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 3,
    "top_modules": 1,
    "output_file": "$PROFILE_OUTPUT"
  }
}
EOT

EXTRA_ARGS="
    --deepspeed \
    --deepspeed_config=$DS_CONFIG \
    --zero-stage=3 \
    --no-pipeline-parallel \
    --checkpoint-activations \
    --recompute-granularity full \
    --recompute-method uniform \
    --deepspeed-activation-checkpointing \
    --megatron-mpi
"