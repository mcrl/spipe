#!/bin/bash

ulimit -v unlimited

## torch dist.
export MASTER_ADDR=$(echo $UNWRAPPED_NODELIST | awk '{print $1}')
export MASTER_PORT=$(comm -23 <(seq 10000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep -v '^\s*$' | sort -u) | shuf -n 1)
export CUDA_DEVICE_MAX_CONNECTIONS=1


DISTRIBUTED_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size $NP \
    --distributed-backend nccl \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT
"

GPT_ARGS="
    --no-initialization \
    --untie-embeddings-and-output-weights \
    --sequence-parallel \
    --num-layers $LAYER \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEAD \
    --seq-length $SEQ \
    --max-position-embeddings $POS \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --train-iters $TRAIN_ITER \
    --log-interval $LOG_ITER \
    --eval-iters $EVAL_ITER \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 0.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-gradient-accumulation-fusion \
    --no-contiguous-buffers-in-local-ddp
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

LOGGING_ARGS="
    --log-throughput
"

if [ -n "$FUSED_KERNEL_LOCK" ] && [ -f "${FUSED_KERNEL_LOCK}" ]; then
    rm ${FUSED_KERNEL_LOCK}
fi

if [ -n "${SPIRAL_SHMEM_NAME}" ] && [ -e "/dev/shm${SPIRAL_SHMEM_NAME}" ]; then
    if [ ! -r "/dev/shm${SPIRAL_SHMEM_NAME}" ] || [ ! -w "/dev/shm${SPIRAL_SHMEM_NAME}" ]; then
        echo "Permission error: /dev/shm${SPIRAL_SHMEM_NAME} exists already and is not readable/writable"
        exit 1
    fi
fi

if [ ${SKIP_TRAIN_ITER_ZERO_TIMING} == "YES" ]; then
    LOGGING_ARGS+=" --skip-train-iter-zero-timing"
fi

EXEC_CMD="python ${MEGATRON_PATH}/pretrain_gpt.py ${EXTRA_ARGS} ${DISTRIBUTED_ARGS} ${GPT_ARGS} ${DATA_ARGS} ${LOGGING_ARGS}"

if [ ${NSYS_ENABLE} == "YES" ]; then
    EXEC_CMD="${NSYS} profile -t cuda,nvtx -o ${NSYS_OUTPUT}_%q{OMPI_COMM_WORLD_RANK} --force-overwrite true ${EXEC_CMD}"
fi

${MPIRUN} -np $NP -host $HOSTS $MPI_OPTIONS ${EXEC_CMD}
