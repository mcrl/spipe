#!/bin/bash

## torch dist.
export MASTER_ADDR=$(echo $UNWRAPPED_NODELIST | awk '{print $1}')
export MASTER_PORT=$(comm -23 <(seq 10000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep -v '^\s*$' | sort -u) | shuf -n 1)
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCH_NCCL_USE_COMM_NONBLOCKING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

DISTRIBUTED_ARGS="
    --tensor-model-parallel-size 1 \
    --distributed-backend nccl \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    --distributed-timeout-minutes 1
"
if [[ $NO_PIPELINE_PARALLEL -eq 1 ]]; then
    DISTRIBUTED_ARGS="
        --pipeline-model-parallel-size 1 \
        ${DISTRIBUTED_ARGS}
    "
else
    DISTRIBUTED_ARGS="
        --pipeline-model-parallel-size $NP \
        ${DISTRIBUTED_ARGS}
    "
fi

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --num-workers 0
"

MIXED_PRECISION_ARGS="
    --fp16
"

LOGGING_ARGS="
    --no-refresh-btw-log-intervals
"

if [ ${SKIP_TRAIN_ITER_ZERO_TIMING} == "YES" ]; then
    LOGGING_ARGS+=" --skip-train-iter-zero-timing"
fi