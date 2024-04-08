#!/bin/bash

#SBATCH -J interleaving
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
    --num-layers-per-virtual-pipeline-stage $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --overlap-p2p-communication \
    --megatron-mpi
"

# Run script
. $(dirname "${SCRIPT_PATH}")/run.sh

exit 0
