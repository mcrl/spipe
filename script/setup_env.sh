#!/bin/bash
export SPIPE_AEC_ROOT=$HOME/spipe-ace
export SPIPE_ROOT=$SPIPE_AEC_ROOT/spipe
export CUDA_ROOT=/usr/local/cuda
export UCX_ROOT=$SPIPE_AEC_ROOT/ucx/build
export MPI_ROOT=$SPIPE_AEC_ROOT/ompi/build
conda activate spipe-pact
