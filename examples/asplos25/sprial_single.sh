#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
GPUS_PER_NODE=4
MASTER_ADDR=b3
MASTER_PORT=9901
NNODES=1
NODE_RANK=0

# Data and tokenizer files.
DATA_PATH=<path to megatron processed data>
VOCAB_FILE=<path to vocab file>
MERGE_FILE=<path to merges file>

ITERATION=5

MBS=2
BS=$((4 * MBS)) # 4 is micro batch num

N_LAYER=32
N_EMBD=1024
N_HEAD=16

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --num-layers-per-virtual-pipeline-stage 4 \
    --spiral-pipeline-parallel \
    --sequence-parallel \
    --num-layers $N_LAYER \
    --hidden-size $N_EMBD \
    --num-attention-heads $N_HEAD \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --lr 0.00015 \
    --train-iters $ITERATION \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval $ITERATION \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters $ITERATION
"

nsys profile --trace=nvtx --force-overwrite true -o spiral_single.nsys-rep \
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
