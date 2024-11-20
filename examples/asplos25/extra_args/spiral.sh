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
    --spiral-recompute-activations \
    --spiral-ckpt-comm-threshold 2 \
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

if [ ${SPIRAL_OFFLOAD_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-offload-optimizer"
fi

if [ ${SPIRAL_CROSS_MAPPING} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-cross-mapping"
fi

if [ ${SPIRAL_SYNC_CKPT_COMMUNICATION} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-sync-ckpt-communication"
fi

if [ ${SPIRAL_ACTV_P2P} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-actv-p2p"
fi