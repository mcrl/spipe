#!/bin/bash

#SBATCH -J spiral
#SBATCH --mincpus=4
#SBATCH --mem=0
#SBATCH --exclusive

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi

. ${HOME}/lib/spipe/env.sh

# Configuration for custom env
. $(dirname "${SCRIPT_PATH}")/config.sh

# Configuration for spiral training
EXTRA_ARGS="
    --spiral \
    --spiral-remap \
    --spiral-shared-memory-name $SPIRAL_SHMEM_NAME \
    --spiral-shared-memory-buffer-size $SPIRAL_SHMEM_BUFFER_SIZE \
    --spiral-shared-memory-header-size $SPIRAL_SHMEM_HEADER_SIZE \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_BWD \
    --spiral-recompute-activations \
    --overlap-p2p-communication \
    --megatron-mpi
"

if [ ${SPIRAL_STAGE_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-stage-optimizer"
    if [ -n ${SPIRAL_STAGE_OPTIMIZER_POOL_SIZE} ]; then
        EXTRA_ARGS+=" --spiral-stage-optimizer-pool-size ${SPIRAL_STAGE_OPTIMIZER_POOL_SIZE}"
    fi
fi

if [ ${SPIRAL_DEBUG_BACKEND} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-debug-backend"
fi

# Run script
. $(dirname "${SCRIPT_PATH}")/run_llama.sh

exit 0
