#!/bin/bash

#SBATCH -J spipe
#SBATCH --mincpus=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --output=results/slurm-%j-%x.out

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
if [ -n "${FUSED_KERNEL_LOCK}" ]; then
    max_attempts=30
    attempts=0

    while [ -f "${FUSED_KERNEL_LOCK}" ]; do
        attempts=$((attempts + 1))
        echo "Waiting for ${FUSED_KERNEL_LOCK} to be removed... Attempt ${attempts}/${max_attempts}"
        sleep 10
        if [ "${attempts}" -ge "${max_attempts}" ] && [ -f "${FUSED_KERNEL_LOCK}" ]; then
            echo "File ${FUSED_KERNEL_LOCK} still exists after ${max_attempts} attempts. Force remove."
            rm "${FUSED_KERNEL_LOCK}" 2>/dev/null || true
        fi
    done
fi

if [ -n "${SPIPE_SHMEM_NAME}" ] && [ -e "/dev/shm${SPIPE_SHMEM_NAME}" ]; then
    if [ ! -r "/dev/shm${SPIPE_SHMEM_NAME}" ] || [ ! -w "/dev/shm${SPIPE_SHMEM_NAME}" ]; then
        echo "Permission error: /dev/shm${SPIPE_SHMEM_NAME} exists already and is not readable/writable"
        exit 1
    fi
fi

# Configure exec cmd
EXEC_CMD="python ${MEGATRON_PATH}/pretrain_gpt.py ${EXTRA_ARGS} ${DISTRIBUTED_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${MIXED_PRECISION_ARGS} ${LOGGING_ARGS}"

if [ ${NSYS_ENABLE} == "YES" ]; then
    EXEC_CMD="${NSYS} profile -t cuda,nvtx -o ${NSYS_OUTPUT}_\$OMPI_COMM_WORLD_RANK --force-overwrite true ${EXEC_CMD}"
fi

# Configure numactl
# Check the partition name and configure numactl accordingly
if [ "$SLURM_JOB_PARTITION" == "spipe-3090" ]; then
    EXEC_CMD="numactl --cpunodebind \$((OMPI_COMM_WORLD_LOCAL_RANK / 2)) --membind \$((OMPI_COMM_WORLD_LOCAL_RANK / 2)) ${EXEC_CMD}"
else
    EXEC_CMD="numactl --cpunodebind \$((3 - \$OMPI_COMM_WORLD_LOCAL_RANK)) --membind \$((3 - \$OMPI_COMM_WORLD_LOCAL_RANK)) ${EXEC_CMD}"
fi


# Remove newline
EXEC_CMD=$(echo "$EXEC_CMD" | tr '\n' ' ')

# Run script
mpirun --bind-to none --report-bindings -npernode $GPUS_PER_NODE -host $HOSTS $MPI_OPTIONS -x OMP_NUM_THREADS=$OMP_NUM_THREADS \
    bash -c "${EXEC_CMD}"

exit 0
