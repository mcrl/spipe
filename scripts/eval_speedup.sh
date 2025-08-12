#!/bin/bash

RUN=${SPIPE_ROOT:-.}/examples/run.sh
TRAIN_ITER=2
LOG_ITER=2

# Fig 9: Speedups of DeepSpeed, Mobius, Megatron, and SPipe (Cluster V100)
# Fig 10: Speedups of DeepSpeed, Mobius, Megatron, and SPipe (Cluster RTX 3090)

for PARTITION in spipe-v100 spipe-3090; do # TODO: adjust to real partition names

    ## Model size: 10B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 1 $RUN -j infinity -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 1 $RUN -j 1f1b_offload -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 10B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 1 $RUN -j infinity -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 1 $RUN -j 1f1b_offload -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 19B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 1 $RUN -j infinity -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 1 $RUN -j 1f1b_offload -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 19B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 1 $RUN -j infinity -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 1 $RUN -j 1f1b_offload -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 30B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 2 $RUN -j infinity -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 2 $RUN -j 1f1b_offload -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 30B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 2 $RUN -j infinity -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 2 $RUN -j 1f1b_offload -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 40B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 2 $RUN -j infinity -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 2 $RUN -j 1f1b_offload -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 40B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 2 $RUN -j infinity -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 2 $RUN -j 1f1b_offload -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 52B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 4 $RUN -j infinity -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 4 $RUN -j 1f1b_offload -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 52B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 4 $RUN -j infinity -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 4 $RUN -j 1f1b_offload -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 69B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 4 $RUN -j infinity -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 4 $RUN -j 1f1b_offload -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 69B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 4 $RUN -j infinity -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 4 $RUN -j 1f1b_offload -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 88B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 8 $RUN -j infinity -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 8 $RUN -j 1f1b_offload -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 88B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 8 $RUN -j infinity -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 8 $RUN -j 1f1b_offload -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2 -o stage

    ## Model size: 110B / Sequence length: 1024
    sbatch -J DeepSpeed -p $PARTITION -N 8 $RUN -j infinity -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 8 $RUN -j 1f1b_offload -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 1024 -v 1 -w 16 -u 2 -o stage

    ## Model size: 110B / Sequence length: 2048
    sbatch -J DeepSpeed -p $PARTITION -N 8 $RUN -j infinity -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2
    sbatch -J Megatron -p $PARTITION -N 8 $RUN -j 1f1b_offload -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2
    sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -z 2048 -v 1 -w 16 -u 2 -o stage

done
