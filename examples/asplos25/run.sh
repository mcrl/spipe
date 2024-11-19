#!/bin/bash

#SBATCH -J spiral
#SBATCH --mincpus=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --gres=gpu:4

ulimit -v unlimited

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi

# Configure env
. $(dirname "${SCRIPT_PATH}")/config.sh

# Configure model args
. $(dirname "${SCRIPT_PATH}")/models/${JOB_NAME}.sh

# Configure args
. $(dirname "${SCRIPT_PATH}")/args.sh

# Configure extra args
. $(dirname "${SCRIPT_PATH}")/extra_args/${JOB_TYPE}.sh

# Remove Megatron lockfile
if [ -n "$FUSED_KERNEL_LOCK" ] && [ -f "${FUSED_KERNEL_LOCK}" ]; then
    rm "${FUSED_KERNEL_LOCK}" 2>/dev/null || true # Ignore timing issue
fi

if [ -n "${SPIRAL_SHMEM_NAME}" ] && [ -e "/dev/shm${SPIRAL_SHMEM_NAME}" ]; then
    if [ ! -r "/dev/shm${SPIRAL_SHMEM_NAME}" ] || [ ! -w "/dev/shm${SPIRAL_SHMEM_NAME}" ]; then
        echo "Permission error: /dev/shm${SPIRAL_SHMEM_NAME} exists already and is not readable/writable"
        exit 1
    fi
fi

# Configure exec cmd
EXEC_CMD="python ${MEGATRON_PATH}/pretrain_gpt.py ${EXTRA_ARGS} ${DISTRIBUTED_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${MIXED_PRECISION_ARGS} ${LOGGING_ARGS}"

if [ ${NSYS_ENABLE} == "YES" ]; then
    EXEC_CMD="${NSYS} profile -t cuda,nvtx -o ${NSYS_OUTPUT}_%q{OMPI_COMM_WORLD_RANK} --force-overwrite true ${EXEC_CMD}"
fi

# Run script
${MPIRUN} --bind-to none --report-bindings -npernode $GPUS_PER_NODE -host $HOSTS $MPI_OPTIONS ${EXEC_CMD}

exit 0