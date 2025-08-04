#!/bin/bash

EXTRA_ARGS="
    --num-layers-per-virtual-pipeline-stage $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --overlap-p2p-communication \
    --megatron-mpi
"