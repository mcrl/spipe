#!/bin/bash

EXTRA_ARGS="
    --spiral \
    --spiral-mobius \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_FWD \
    --spiral-recompute-activations \
    --spiral-overlap-offload-grad \
    --overlap-p2p-communication \
    --megatron-mpi
"

if [ ${SPIRAL_DEBUG_BACKEND} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-debug-backend"
fi

if [ ${SPIRAL_STAGE_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+="
    --spiral-stage-optimizer \
    --spiral-stage-optimizer-pool-size 0
    "
fi

if [ ${SPIRAL_HETERO_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-heterogeneous-optimizer"
fi