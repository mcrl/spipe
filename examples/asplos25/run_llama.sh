#!/bin/bash

ulimit -v unlimited

## torch dist.
export MASTER_ADDR=$(echo $UNWRAPPED_NODELIST | awk '{print $1}')
export MASTER_PORT=$(comm -23 <(seq 10000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep -v '^\s*$' | sort -u) | shuf -n 1)
export CUDA_DEVICE_MAX_CONNECTIONS=1

HIDDEN=2048 # e.g. llama-13b: 5120
FFN_HIDDEN=5504 # e.g. llama-13b: 13824
LAYER=24 # e.g. llama-13b: 40
HEAD=16 # e.g. llama-13b: 40
SEQ=2048
NUM_KV_HEADS=4 # llama2 70B uses GQA

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=20
WEIGHT_DECAY=0.1
GRAD_CLIP=0

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
    --ffn-hidden-size $FFN_HIDDEN \
    --num-attention-heads $HEAD \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr $LR \
    --train-iters $TRAIN_ITER \
    --log-interval $LOG_ITER \
    --eval-iters $EVAL_ITER \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr $MIN_LR \
    --weight-decay $WEIGHT_DECAY \
    --clip-grad $GRAD_CLIP \
    --lr-warmup-iters $LR_WARMUP_STEPS \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-gradient-accumulation-fusion \
    --no-contiguous-buffers-in-local-ddp
"

LLAMA_ARGS="
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --disable-bias-linear \
    --normalization rmsnorm \
    --num-key-value-heads $NUM_KV_HEADS \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
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

EXEC_CMD="python ${MEGATRON_PATH}/pretrain_gpt.py ${EXTRA_ARGS} ${DISTRIBUTED_ARGS} ${GPT_ARGS} ${LLAMA_ARGS} ${DATA_ARGS} ${LOGGING_ARGS}"

if [ ${NSYS_ENABLE} == "YES" ]; then
    EXEC_CMD="${NSYS} profile -t cuda,nvtx -o ${NSYS_OUTPUT}_%q{OMPI_COMM_WORLD_RANK} --force-overwrite true ${EXEC_CMD}"
fi

${MPIRUN} -np $NP -host $HOSTS $MPI_OPTIONS ${EXEC_CMD}
