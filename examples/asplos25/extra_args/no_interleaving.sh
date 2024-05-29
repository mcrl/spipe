#!/bin/bash

EXTRA_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers $(($LAYER/$NP)) \
    --megatron-mpi
"