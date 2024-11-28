#!/bin/bash

EXTRA_ARGS="
    --spiral \
    --spiral-mobius \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_FWD \
    --spiral-overlap-offload-grad \
    --spiral-log-gpu-pipeline-latency \
    --overlap-p2p-communication \
    --megatron-mpi
"

if [ ${SPIRAL_DEBUG_BACKEND} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-debug-backend"
fi

if [ ${SPIRAL_STAGE_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+="
    --spiral-stage-optimizer \
    --spiral-stage-optimizer-pool-size $SPIRAL_STAGE_OPTIMIZER_POOL_SIZE"
fi