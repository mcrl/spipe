#!/bin/bash

EXTRA_ARGS="
    --spiral \
    --spiral-1f1b \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_FWD \
    --spiral-overlap-offload-grad \
    --spiral-actv-p2p \
    --spiral-log-gpu-pipeline-latency
    --overlap-p2p-communication \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --megatron-mpi \
"

if [ ${SPIRAL_DEBUG_BACKEND} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-debug-backend"
fi

if [ ${SPIRAL_STAGE_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+="
    --spiral-stage-optimizer \
    --spiral-stage-optimizer-pool-size $SPIRAL_STAGE_OPTIMIZER_POOL_SIZE"
fi