#!/bin/bash

while getopts "j:n:s:t:l:f:b:m:g:x:" opt
do
    case "$opt" in
        j ) JOB_TYPE="$OPTARG" ;;
        n ) JOB_NAME="$OPTARG" ;;
        s ) MODEL_SIZE="$OPTARG" ;;
        t ) TRAIN_ITER="$OPTARG" ;;
        l ) LOG_ITER="$OPTARG" ;;
        f ) FWD_STAGE="$OPTARG" ;;
        b ) BWD_STAGE="$OPTARG" ;;
        m ) MBS="$OPTARG" ;;
        g ) GBS="$OPTARG" ;;
        x ) NRR="$OPTARG" ;;
    esac
done

## conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.4-cuda-12.4-python-3.8  

# MPI
MPIRUN=$(which mpirun)
MPI_OPTIONS="-mca btl ^openib -mca pml ucx"
GPUS_PER_NODE=4
NP=$(( $GPUS_PER_NODE * $SLURM_JOB_NUM_NODES ))
UNWRAPPED_NODELIST=$(scontrol show hostnames $SLURM_NODELIST) # b3 b4
HOSTS=$(for node in $UNWRAPPED_NODELIST; do echo -n "$node:$GPUS_PER_NODE,"; done | sed 's/,$//') # b3:2,b4:2

MEGATRON_DEEPSPEED=0
if [[ "$JOB_TYPE" == *"zero3"* || "$JOB_TYPE" == *"infinity"* || "$JOB_TYPE" == *"deepspeed"* ]]; then
    MEGATRON_DEEPSPEED=1
fi

NO_PIPELINE_PARALLEL=0
if [[ "$JOB_TYPE" == *"zero3"* || "$JOB_TYPE" == *"infinity"* ]]; then
    NO_PIPELINE_PARALLEL=1
fi

# Source code
if [[ $MEGATRON_DEEPSPEED -eq 1 ]]; then
    export MEGATRON_PATH=${HOME}/spipe/Megatron-LM-mcrl/external/Megatron-DeepSpeed
else
    export MEGATRON_PATH=${HOME}/spipe/Megatron-LM-mcrl
fi
FUSED_KERNEL_LOCK=${MEGATRON_PATH}/megatron/fused_kernels/build/lock

# nsys
NSYS_ENABLE=NO
NSYS=$(which nsys)
NSYS_OUTPUT=${MEGATRON_PATH}/logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}

# Data and tokenizer files
DATA_PATH=/shared/s1/lab08/junyeol/openwebtext/openwebtext_text_document
VOCAB_FILE=/shared/s1/lab08/junyeol/megatron-deepspeed-data/gpt/gpt2-vocab.json
MERGE_FILE=/shared/s1/lab08/junyeol/megatron-deepspeed-data/gpt/gpt2-merges.txt

# Job type and model name
JOB_TYPE=${JOB_TYPE:="spiral"}
JOB_NAME=${JOB_NAME:="opt"}

# Micro Batch size
MBS=${MBS:=1}
GBS=${GBS:=$(( $MBS * $NP ))}

# iteration
TRAIN_ITER=${TRAIN_ITER:=100}
LOG_ITER=${LOG_ITER:=10}
SKIP_TRAIN_ITER_ZERO_TIMING=YES
EVAL_ITER=0

# config for spiral training
SPIRAL_FWD=${FWD_STAGE:=2}
SPIRAL_BWD=${BWD_STAGE:=2}
SPIRAL_SHMEM_NAME=/spiral-${USER}
SPIRAL_SHMEM_BUFFER_SIZE=${SHMEM_BUFFER_SIZE:=$(( 64 * 2**30 ))}
SPIRAL_SHMEM_HEADER_SIZE=$(( 1 * 2**30 ))
SPIRAL_DEBUG_BACKEND=NO

# config for interleaving
INTERLEAVE_VIRTUAL_SIZE=${FWD_STAGE:=2}

# Print configuration
echo -e "===========Script Configuration==========="
echo -e "JOB_TYPE=${JOB_TYPE}\nJOB_NAME=${JOB_NAME}\nHOSTS=${HOSTS}\nNSYS_ENABLE=${NSYS_ENABLE}"
echo -e "MODEL_SIZE=${MODEL_SIZE}\nMBS=${MBS}\nGBS=${GBS}"
echo -e "TRAIN_ITER=${TRAIN_ITER}\nLOG_ITER=${LOG_ITER}(skip0=${SKIP_TRAIN_ITER_ZERO_TIMING})\nEVAL_ITER=${EVAL_ITER}"
echo -e "SPIRAL_FWD=${SPIRAL_FWD}\nSPIRAL_BWD=${SPIRAL_BWD}"
echo -e "INTERLEAVE_VIRTUAL_SIZE=${INTERLEAVE_VIRTUAL_SIZE}"
echo "==========================================="