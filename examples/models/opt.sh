#!/bin/bash

# Model spec
if [ $MODEL_SIZE -eq 10 ]; then
    # 10B
    LAYER=48
    HIDDEN=4096
    HEAD=32
elif [ $MODEL_SIZE -eq 19 ]; then
    # 19B
    LAYER=48
    HIDDEN=5632
    HEAD=44
elif [ $MODEL_SIZE -eq 30 ]; then
    # 30B
    LAYER=96
    HIDDEN=5120
    HEAD=40
elif [ $MODEL_SIZE -eq 35 ]; then
    # 35B
    LAYER=96
    HIDDEN=5504
    HEAD=32
elif [ $MODEL_SIZE -eq 40 ]; then
    # 40B
    LAYER=96
    HIDDEN=5888
    HEAD=46
elif [ $MODEL_SIZE -eq 52 ]; then
    # 52B
    LAYER=96
    HIDDEN=6656
    HEAD=52
elif [ $MODEL_SIZE -eq 60 ]; then
    # 60B
    LAYER=96
    HIDDEN=7168
    HEAD=56
elif [ $MODEL_SIZE -eq 67 ]; then
    # 67B
    LAYER=192
    HIDDEN=5376
    HEAD=48
elif [ $MODEL_SIZE -eq 69 ]; then
    # 69B
    LAYER=96
    HIDDEN=7680
    HEAD=48
elif [ $MODEL_SIZE -eq 74 ]; then
    # 74B
    LAYER=192
    HIDDEN=5632
    HEAD=44
elif [ $MODEL_SIZE -eq 77 ]; then
    # 77B
    LAYER=192
    HIDDEN=5760
    HEAD=48
elif [ $MODEL_SIZE -eq 80 ]; then
    # 88B
    LAYER=192
    HIDDEN=5888
    HEAD=46
elif [ $MODEL_SIZE -eq 88 ]; then
    # 88B
    LAYER=192
    HIDDEN=6144
    HEAD=48
elif [ $MODEL_SIZE -eq 110 ]; then
    # 110B
    LAYER=192
    HIDDEN=6912
    HEAD=54
else
    echo "Unsupported model size"
    HIDDEN=2048 
    LAYER=24
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
    --initial-loss-scale $((2**$INIT_LOSS_SCALE_POWER))
"