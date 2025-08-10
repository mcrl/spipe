#!/bin/bash

while getopts "j:n:s:t:l:f:b:m:g:o:q:u:v:w:x:y:z:" opt
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
        o ) OPTIMIZER="$OPTARG" ;;
        q ) BLOCK_PREFETCH="$OPTARG" ;;
        u ) OMP_NUM_THREADS="$OPTARG" ;;
        v ) ACTV_P2P="$OPTARG" ;;
        w ) INIT_LOSS_SCALE_POWER="$OPTARG" ;;
        x ) CROSS_MAPPING="$OPTARG" ;;
        y ) SYNC_CKPT_COMMUNICATION="$OPTARG" ;;
        z ) SEQ="$OPTARG" ;;
    esac
done

## conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-pytorch-2.4-cuda-12.4-python-3.8}

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
    export MEGATRON_PATH=${SPIPE_ROOT:-${HOME}/spipe-aec/spipe}/external/Megatron-DeepSpeed
else
    export MEGATRON_PATH=${SPIPE_ROOT:-${HOME}/spipe-aec/spipe}
fi
FUSED_KERNEL_LOCK=${MEGATRON_PATH}/megatron/fused_kernels/build/lock

# nsys
NSYS_ENABLE=NO
NSYS=$(which nsys)
NSYS_OUTPUT=${MEGATRON_PATH}/logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}

# Data and tokenizer files
DATA_PATH=${DATA_PATH:-\
/shared/s1/lab08/junyeol/openwebtext/openwebtext_text_document}
VOCAB_FILE=${VOCAB_FILE:-\
/shared/s1/lab08/junyeol/megatron-deepspeed-data/gpt/gpt2-vocab.json}
MERGE_FILE=${MERGE_FILE:-\
/shared/s1/lab08/junyeol/megatron-deepspeed-data/gpt/gpt2-merges.txt}

# Job type and model name
JOB_TYPE=${JOB_TYPE:="spipe"}
JOB_NAME=${JOB_NAME:="opt"}

# Micro Batch size
MBS=${MBS:=1}
GBS=${GBS:=$(( $MBS * $NP ))}

# Training configs
SEQ=${SEQ:=4096}
INIT_LOSS_SCALE_POWER=${INIT_LOSS_SCALE_POWER:=32}
OMP_NUM_THREADS=${OMP_NUM_THREADS:=1}

# iteration
TRAIN_ITER=${TRAIN_ITER:=100}
LOG_ITER=${LOG_ITER:=10}
SKIP_TRAIN_ITER_ZERO_TIMING=YES
EVAL_ITER=0

# config for spipe training
SPIPE_FWD=${FWD_STAGE:=2}
SPIPE_BWD=${BWD_STAGE:=2}
SPIPE_SHMEM_NAME=/spipe-${USER}
SPIPE_SHMEM_BUFFER_SIZE=${SHMEM_BUFFER_SIZE:=$(( 48 * 2**30 ))}
SPIPE_SHMEM_HEADER_SIZE=$(( 1 * 2**30 ))
SPIPE_DEBUG_BACKEND=NO
SPIPE_STAGE_OPTIMIZER_POOL_SIZE=1

SPIPE_HETERO_OPTIMIZER=NO
SPIPE_OFFLOAD_OPTIMIZER=NO
if [[ "$OPTIMIZER" == *"stage"* ]]; then
    SPIPE_STAGE_OPTIMIZER=YES
    if [[ "$OPTIMIZER" == *"hetero"* ]]; then
        SPIPE_HETERO_OPTIMIZER=YES
    fi
    if [[ "$OPTIMIZER" == *"offload"* ]]; then
        SPIPE_OFFLOAD_OPTIMIZER=YES
    fi
else
    SPIPE_STAGE_OPTIMIZER=NO
fi

if [[ "$CROSS_MAPPING" == "1" ]]; then
    SPIPE_CROSS_MAPPING=YES
else
    SPIPE_CROSS_MAPPING=NO
fi

if [[ "$SYNC_CKPT_COMMUNICATION" == "1" ]]; then
    SPIPE_SYNC_CKPT_COMMUNICATION=YES
else
    SPIPE_SYNC_CKPT_COMMUNICATION=NO
fi

if [[ "$ACTV_P2P" == "1" ]]; then
    SPIPE_ACTV_P2P=YES
else
    SPIPE_ACTV_P2P=NO
fi

if [[ "$BLOCK_PREFETCH" == "1" ]]; then
    SPIPE_BLOCK_PREFETCH=YES
else
    SPIPE_BLOCK_PREFETCH=NO
fi

# config for interleaving
INTERLEAVE_VIRTUAL_SIZE=${FWD_STAGE:=2}

# Print configuration
echo -e "===========Script Configuration==========="
echo -e "JOB_TYPE=${JOB_TYPE}\nJOB_NAME=${JOB_NAME}\nHOSTS=${HOSTS}\nNSYS_ENABLE=${NSYS_ENABLE}"
echo -e "MODEL_SIZE=${MODEL_SIZE}\nMBS=${MBS}\nGBS=${GBS}"
echo -e "TRAIN_ITER=${TRAIN_ITER}\nLOG_ITER=${LOG_ITER}(skip0=${SKIP_TRAIN_ITER_ZERO_TIMING})\nEVAL_ITER=${EVAL_ITER}"
echo -e "SPIPE_STAGE_OPTIMIZER=${SPIPE_STAGE_OPTIMIZER}(omp=${OMP_NUM_THREADS},pool=${SPIPE_STAGE_OPTIMIZER_POOL_SIZE})"
echo -e "SPIPE_ACTV_P2P=${SPIPE_ACTV_P2P}"
echo -e "SPIPE_CROSS_MAPPING=${SPIPE_CROSS_MAPPING}"
echo -e "SPIPE_SYNC_CKPT_COMMUNICATION=${SPIPE_SYNC_CKPT_COMMUNICATION}"
echo -e "SPIPE_BLOCK_PREFETCH=${SPIPE_BLOCK_PREFETCH}"
echo -e "SPIPE_FWD=${SPIPE_FWD}\nSPIPE_BWD=${SPIPE_BWD}"
echo -e "INTERLEAVE_VIRTUAL_SIZE=${INTERLEAVE_VIRTUAL_SIZE}"
echo "==========================================="
