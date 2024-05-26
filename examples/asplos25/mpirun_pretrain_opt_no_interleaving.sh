#!/bin/bash

#SBATCH -J no-interleaving
#SBATCH --mincpus=4
#SBATCH --mem=0
#SBATCH --exclusive

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi

# Configuration for custom env
JOB_TYPE="no-interleaving-recompute"
JOB_NAME="gpt"
. $(dirname "${SCRIPT_PATH}")/config.sh

# Configuration for mobius-recompute training
EXTRA_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers $(($LAYER/$NP)) \
    --megatron-mpi
"

# Run script
. $(dirname "${SCRIPT_PATH}")/run_opt.sh

exit 0
