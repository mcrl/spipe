#!/bin/bash

MPIRUN=/usr/local/bin/mpirun
MPI_OPTIONS="-mca btl ^openib -mca pml ucx"
MEGATRON_PATH=$HOME/asplos2025/Megatron-LM-mcrl

# Change for your system

## conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Megatron-cuda11.7

## mpi
NP=4
GPUS_PER_NODE=4
HOSTS="b4:${GPUS_PER_NODE}" # b3:2,b4:2

## torch dist.
export MASTER_ADDR="b4"
export MASTER_PORT=6003
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

## Config file path
MODEL_PATH=/home/n0/yujin/tmp/model/gpt-like-270m
VOCAB_FILE=/home/n0/yujin/tmp/tokenizer/megatron/gpt2-vocab.json
MERGE_FILE=/home/n0/yujin/tmp/tokenizer/megatron/gpt2-merges.txt

DATASET_NAME=openwebtext
DATASET_CONFIG=plan_text
DATA_PATH=/data/z0/heehoon/openwebtext-mg/openwebtext_text_document

# --spiral-stage-optimizer
SPIRAL_ARGS="
    --spiral \
    --spiral-forward-virtual-size 2 \
    --spiral-backward-virtual-size 2 \
    --spiral-recompute-activations \
    --megatron-mpi
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size $NP \
    --no-initialization \
    --untie-embeddings-and-output-weights \
    --distributed-backend nccl \
    --overlap-p2p-communication \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --lr 0.00015 \
    --train-iters 3 \
    --eval-iters 0 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 0.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-gradient-accumulation-fusion
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

${MPIRUN} -np $NP -host $HOSTS $MPI_OPTIONS \
    python $MEGATRON_PATH/pretrain_gpt.py $SPIRAL_ARGS $GPT_ARGS $DATA_ARGS --load $MODEL_PATH
