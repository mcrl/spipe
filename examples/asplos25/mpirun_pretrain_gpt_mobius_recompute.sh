#!/bin/bash

#SBATCH -J mobius-recompute
#SBATCH --mincpus=4
#SBATCH --mem=0
#SBATCH --exclusive

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi

# Configuration for custom env
. $(dirname "${SCRIPT_PATH}")/config.sh

# Configuration for mobius-recompute training
EXTRA_ARGS="
    --spiral \
    --spiral-forward-virtual-size $SPIRAL_FWD \
    --spiral-backward-virtual-size $SPIRAL_FWD \
    --spiral-recompute-activations \
    --overlap-p2p-communication \
    --megatron-mpi
"

if [ ${SPIRAL_STAGED_OPTIMIZER} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-stage-optimizer"
fi

if [ ${SPIRAL_DEBUG_BACKEND} == "YES" ]; then
    EXTRA_ARGS+=" --spiral-debug-backend"
fi

# Run script
. $(dirname "${SCRIPT_PATH}")/run.sh

exit 0
