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
JOB_TYPE="interleaving"
JOB_NAME="llama2"
. $(dirname "${SCRIPT_PATH}")/config.sh

# Configuration for mobius-recompute training
EXTRA_ARGS="
    --num-layers-per-virtual-pipeline-stage $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers $(($LAYER/$NP/$INTERLEAVE_VIRTUAL_SIZE)) \
    --overlap-p2p-communication \
    --megatron-mpi
"

# Run script
. $(dirname "${SCRIPT_PATH}")/run_llama2.sh

exit 0
