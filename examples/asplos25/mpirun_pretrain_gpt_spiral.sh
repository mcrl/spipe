#!/bin/bash

# Activate conda env
source ~/anaconda3/etc/profile.d/conda.sh # change
conda activate Megatron-cuda11.7 # change

MPI_OPTIONS="-mca btl ^openib -mca pml ucx"

# Set ENVs
MACHINE="B" # change
HOSTS="b4" # change

# Set ENVS for MPI processes
export NCCL_LIB_PATH="$HOME/asplos2025/nccl-branches/nccl-$(echo $MACHINE | tr '[:upper:]' '[:lower:]')/build/lib/" # change
export LD_LIBRARY_PATH=${NCCL_LIB_PATH}

export MASTER_ADDR="b4"
export MASTER_PORT=6003
export GPUS_PER_NODE=3
export CUDA_VISIBLE_DEVICES=1,2,3

export TP_SIZE=1
export PP_SIZE=$GPUS_PER_NODE

# Runs the "270M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

MICRO_BSZ=1 # change
GLOBAL_BSZ=4 # change

DDP_IMPL=torch # local or torch

LOG_INTERVAL=100
EXIT_INTERVAL=10 # for profiling purposes

# Config file path
MODEL_PATH=/home/n0/yujin/tmp/model/gpt-like-270m
VOCAB_FILE=/home/n0/yujin/tmp/tokenizer/megatron/gpt2-vocab.json
MERGE_FILE=/home/n0/yujin/tmp/tokenizer/megatron/gpt2-merges.txt

DATASET_NAME=openwebtext
DATASET_CONFIG=plan_text
DATA_PATH=/data/z0/heehoon/openwebtext-mg/openwebtext_text_document

MEGATRON_PATH=$HOME/asplos2025/Megatron-LM-mcrl

# TODO (mcrl) fp16 currently removed. no-initialization is added.
GPT_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --spiral-pipeline-parallel \
    --no-initialization \
    --untie-embeddings-and-output-weights \
    --distributed-backend nccl \
    --overlap-p2p-communication \
    --sequence-parallel \
    --num-layers 6 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BSZ \
    --global-batch-size $GLOBAL_BSZ \
    --lr 0.00015 \
    --train-iters 500 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

mpirun -n $GPUS_PER_NODE -x LD_LIBRARY_PATH $MPI_OPTIONS -x CUDA_VISIBLE_DEVICES \
    python $MEGATRON_PATH/pretrain_gpt.py --megatron-mpi $GPT_ARGS $DATA_ARGS $OUTPUT_ARGS --load $MODEL_PATH