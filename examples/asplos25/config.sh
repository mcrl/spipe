#!/bin/bash

while getopts "s:t:l:f:b:m:g:" opt
do
    case "$opt" in
        s ) MODEL_SIZE="$OPTARG" ;;
        t ) TRAIN_ITER="$OPTARG" ;;
        l ) LOG_ITER="$OPTARG" ;;
        f ) FWD_STAGE="$OPTARG" ;;
        b ) BWD_STAGE="$OPTARG" ;;
        m ) MBS="$OPTARG" ;;
        g ) GBS="$OPTARG" ;;
    esac
done

# Modify this file for custom configuration

## conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spiral

# MPI
MPIRUN=$(which mpirun)
MPI_OPTIONS="-mca btl ^openib -mca pml ucx"
GPUS_PER_NODE=4
NP=$(( $GPUS_PER_NODE * $SLURM_JOB_NUM_NODES ))
UNWRAPPED_NODELIST=$(scontrol show hostnames $SLURM_NODELIST) # b3 b4
HOSTS=$(for node in $UNWRAPPED_NODELIST; do echo -n "$node:$GPUS_PER_NODE,"; done | sed 's/,$//') # b3:2,b4:2

# Source code
export MEGATRON_PATH=$HOME/asplos2025/Megatron-LM-mcrl
FUSED_KERNEL_LOCK=${MEGATRON_PATH}/megatron/fused_kernels/build/lock

# nsys
NSYS_ENABLE=NO
NSYS=$(which nsys)
NSYS_OUTPUT=${MEGATRON_PATH}/logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}

# Data and tokenizer files
DATA_PATH=/data/z0/heehoon/openwebtext-mg/openwebtext_text_document
VOCAB_FILE=/home/n0/yujin/tmp/tokenizer/megatron/gpt2-vocab.json
MERGE_FILE=/home/n0/yujin/tmp/tokenizer/megatron/gpt2-merges.txt

# Model spec
LAYER=32
HIDDEN=1024
HEAD=16

if [ $MODEL_SIZE -eq 0 ]; then
    # 0.4B
    LAYER=32
    HIDDEN=1024
    HEAD=16
    SHMEM_BUFFER_SIZE=$(( 4 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 1 ]; then
    # 0.9B
    LAYER=32
    HIDDEN=1536
    HEAD=24
    SHMEM_BUFFER_SIZE=$(( 8 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 2 ]; then
    # 1.8B
    LAYER=32
    HIDDEN=2160
    HEAD=24
    SHMEM_BUFFER_SIZE=$(( 12 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 3 ]; then
    # 3.2B
    LAYER=32
    HIDDEN=2880
    HEAD=32
    SHMEM_BUFFER_SIZE=$(( 16 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 6 ]; then
    # 6.4B
    LAYER=32
    HIDDEN=4096
    HEAD=32
    SHMEM_BUFFER_SIZE=$(( 32 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 14 ]; then
    # 14B
    LAYER=48
    HIDDEN=5040
    HEAD=48
    SHMEM_BUFFER_SIZE=$(( 64 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 30 ]; then
    # 30B
    LAYER=48
    HIDDEN=7200
    HEAD=60
    SHMEM_BUFFER_SIZE=$(( 128 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 51 ]; then
    # 51B
    LAYER=64
    HIDDEN=8192
    HEAD=64
    SHMEM_BUFFER_SIZE=$(( 212 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 71 ]; then
    # 71B
    LAYER=72
    HIDDEN=7200
    HEAD=72
    SHMEM_BUFFER_SIZE=$(( 292 / $SLURM_JOB_NUM_NODES * 2**30 ))
elif [ $MODEL_SIZE -eq 88 ]; then
    # 88B
    LAYER=80
    HIDDEN=9600
    HEAD=80
    SHMEM_BUFFER_SIZE=$(( 360 / $SLURM_JOB_NUM_NODES * 2**30 ))
fi

SEQ=2048
POS=2048

# Micro Batch size
MBS=${MBS:=1}
GBS=${GBS:=$(( $MBS * $NP ))}

# iteration
TRAIN_ITER=${TRAIN_ITER:=101}
LOG_ITER=${LOG_ITER:=11}
EVAL_ITER=0

# config for spiral training
SPIRAL_FWD=${FWD_STAGE:=1}
SPIRAL_BWD=${BWD_STAGE:=2}
SPIRAL_STAGE_OPTIMIZER=YES
SPIRAL_STAGE_OPTIMIZER_POOL_SIZE=0
SPIRAL_SHMEM_NAME=/spiral-${USER}
SPIRAL_SHMEM_BUFFER_SIZE=${SHMEM_BUFFER_SIZE:=$(( 64 * 2**30 ))}
SPIRAL_SHMEM_HEADER_SIZE=$(( 1 * 2**30 ))
SPIRAL_DEBUG_BACKEND=NO

# config for interleaving
INTERLEAVE_VIRTUAL_SIZE=${FWD_STAGE:=2}

# Print configuration
echo -e "===========Script Configuration==========="
echo -e "JOB_TYPE=${JOB_TYPE}\nJOB_NAME=${SLURM_JOB_NAME}\nHOSTS=${HOSTS}\nNSYS_ENABLE=${NSYS_ENABLE}"
echo -e "MODEL_SIZE=${MODEL_SIZE}\nMBS=${MBS}\nGBS=${GBS}"
echo -e "TRAIN_ITER=${TRAIN_ITER}\nLOG_ITER=${LOG_ITER}\nEVAL_ITER=${EVAL_ITER}"
echo -e "SPIRAL_FWD=${SPIRAL_FWD}\nSPIRAL_BWD=${SPIRAL_BWD}"
echo -e "SPIRAL_STAGE_OPTIMIZER=${SPIRAL_STAGE_OPTIMIZER}(pool_size=${SPIRAL_STAGE_OPTIMIZER_POOL_SIZE})\nSPIRAL_DEBUG_BACKEND=${SPIRAL_DEBUG_BACKEND}"
echo -e "INTERLEAVE_VIRTUAL_SIZE=${INTERLEAVE_VIRTUAL_SIZE}"
echo "==========================================="
