#!/bin/bash

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
SEQ=1024
POS=1024

# Micro Batch size
MBS=1

# iteration
TRAIN_ITER=100
LOG_ITER=10
EVAL_ITER=0

# config for spiral training
SPIRAL_FWD=1
SPIRAL_BWD=3
SPIRAL_STAGE_OPTIMIZER=YES
SPIRAL_DEBUG_BACKEND=NO

# config for interleaving
INTERLEAVE_VIRTUAL_SIZE=2

# Print configuration
echo -e "===========Script Configuration==========="
echo -e "JOB_NAME=${SLURM_JOB_NAME}\nHOSTS=${HOSTS}\nNSYS_ENABLE=${NSYS_ENABLE}"
echo -e "LAYER=${LAYER}\nHIDDEN=${HIDDEN}\nHEAD=${HEAD}\nMBS=${MBS}"
echo -e "TRAIN_ITER=${TRAIN_ITER}\nLOG_ITER=${LOG_ITER}\nEVAL_ITER=${EVAL_ITER}"
echo -e "SPIRAL_FWD=${SPIRAL_FWD}\nSPIRAL_BWD=${SPIRAL_BWD}"
echo -e "SPIRAL_STAGE_OPTIMIZER=${SPIRAL_STAGE_OPTIMIZER}\nSPIRAL_DEBUG_BACKEND=${SPIRAL_DEBUG_BACKEND}"
echo -e "INTERLEAVE_VIRTUAL_SIZE=${INTERLEAVE_VIRTUAL_SIZE}"
echo "==========================================="
