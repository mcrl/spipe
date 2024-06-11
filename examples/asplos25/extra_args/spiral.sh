#!/bin/bash

EXTRA_ARGS="
    --spiral \
    --spiral-remap \
    --spiral-shared-memory-name $SPIRAL_SHMEM_NAME \
    --spiral-shared-memory-buffer-size $SPIRAL_SHMEM_BUFFER_SIZE \
    --spiral-shared-memory-header-size $SPIRAL_SHMEM_HEADER_SIZE \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_BWD \
    --spiral-overlap-offload-grad \
    --spiral-stage-optimizer \
    --spiral-stage-optimizer-pool-size 0 \
    --spiral-recompute-activations \
    --spiral-cross-mapping \
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
