#!/bin/bash

EXTRA_ARGS="
    --spiral \
    --spiral-mobius \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_FWD \
    --spiral-recompute-activations \
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

if [ ${SPIRAL_HETERO_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-heterogeneous-optimizer"
fi

if [ ${SPIRAL_OFFLOAD_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-offload-optimizer"
fi

if [ ${SPIRAL_ACTV_P2P} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-actv-p2p"
fi