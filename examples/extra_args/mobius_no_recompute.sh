#!/bin/bash

EXTRA_ARGS="
    --spipe \
    --spipe-mobius \
    --spipe-forward-virtual-size $SPIPE_FWD \
    --spipe-backward-virtual-size $SPIPE_FWD \
    --spipe-overlap-offload-grad \
    --spipe-log-gpu-pipeline-latency \
    --overlap-p2p-communication \
    --megatron-mpi
"

if [ ${SPIPE_DEBUG_BACKEND} == "YES" ]; then
    EXTRA_ARGS+=" --spipe-debug-backend"
fi

if [ ${SPIPE_STAGE_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+="
    --spipe-stage-optimizer \
    --spipe-stage-optimizer-pool-size $SPIPE_STAGE_OPTIMIZER_POOL_SIZE"
fi