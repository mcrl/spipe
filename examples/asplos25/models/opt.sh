#!/bin/bash

# Model spec
if [ $MODEL_SIZE -eq 0 ]; then
    # 350M
    LAYER=24
    HIDDEN=1024
    HEAD=16
elif [ $MODEL_SIZE -eq 1 ]; then
    # 1.3B
    LAYER=24
    HIDDEN=2048
    HEAD=32
elif [ $MODEL_SIZE -eq 2 ]; then
    # 2.7B
    LAYER=32
    HIDDEN=2560
    HEAD=32
elif [ $MODEL_SIZE -eq 6 ]; then
    # 6.7B
    LAYER=32
    HIDDEN=4096
    HEAD=32
elif [ $MODEL_SIZE -eq 13 ]; then
    # 13B
    LAYER=40
    HIDDEN=5120
    HEAD=40
elif [ $MODEL_SIZE -eq 30 ]; then
    # 30B
    LAYER=48
    HIDDEN=7168
    HEAD=56
elif [ $MODEL_SIZE -eq 52 ]; then
    # 52B
    LAYER=64
    HIDDEN=8192
    HEAD=64
elif [ $MODEL_SIZE -eq 81 ]; then
    # 81B
    LAYER=64
    HIDDEN=10240
    HEAD=64
elif [ $MODEL_SIZE -eq 121 ]; then
    # 121B
    LAYER=96
    HIDDEN=10240
    HEAD=96
elif [ $MODEL_SIZE -eq 175 ]; then
    # 175B
    LAYER=96
    HIDDEN=12288
    HEAD=96
else
    LAYER=32
    HIDDEN=1024
    HEAD=16
fi

MODEL_ARGS="
    --no-initialization \
    --untie-embeddings-and-output-weights \
    --sequence-parallel \
    --num-layers $LAYER \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEAD \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
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
    --no-contiguous-buffers-in-local-ddp \
    --initial-loss-scale-power $INIT_LOSS_SCALE_POWER
"