#!/bin/bash
export SPIPE_AEC_ROOT=$HOME/spipe-aec
export SPIPE_ROOT=$SPIPE_AEC_ROOT/spipe
export CUDA_ROOT=/usr/local/cuda
export UCX_ROOT=$SPIPE_AEC_ROOT/ucx/build
export MPI_ROOT=$SPIPE_AEC_ROOT/ompi/build

export PATH="$MPI_ROOT/bin:$UCX_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_ROOT/lib:$UCX_ROOT/lib:$LD_LIBRARY_PATH"

conda activate spipe-pact
