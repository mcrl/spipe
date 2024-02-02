#!/bin/bash

# Activate conda env
source ~/anaconda3/etc/profile.d/conda.sh # change
conda activate Megatron-cuda11.7 # change

MPI_OPTIONS="-mca btl ^openib -mca pml ucx"

# Set ENVs
MACHINE="B" # change
HOSTS="b4" # change
GPUS_PER_NODE=4 # change

# Set ENVS for MPI processes
export NCCL_LIB_PATH="$HOME/asplos2025/nccl-branches/nccl-$(echo $MACHINE | tr '[:upper:]' '[:lower:]')/build/lib/" # change
export LD_LIBRARY_PATH=${NCCL_LIB_PATH}

mpirun -n $GPUS_PER_NODE -x LD_LIBRARY_PATH $MPI_OPTIONS \
    python $(dirname "$0")/test_allocator.py
