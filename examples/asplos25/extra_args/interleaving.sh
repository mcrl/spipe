#!/bin/bash

EXTRA_ARGS="
    --num-layers-per-virtual-pipeline-stage $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --overlap-p2p-communication \
    --megatron-mpi
"