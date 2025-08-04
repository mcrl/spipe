#!/bin/bash

# Model spec
if [ $MODEL_SIZE -eq 10 ]; then
    # 10B
    LAYER=48
    HIDDEN=4096
    FFN_HIDDEN=11008
    HEAD=2
    NUM_KV_HEADS=2
elif [ $MODEL_SIZE -eq 19 ]; then
    # 19B
    LAYER=48
    HIDDEN=5632
    FFN_HIDDEN=13312
    HEAD=4
    NUM_KV_HEADS=4
elif [ $MODEL_SIZE -eq 30 ]; then
    # 30B
    LAYER=96
    HIDDEN=5120
    FFN_HIDDEN=12800
    HEAD=4
    NUM_KV_HEADS=4
elif [ $MODEL_SIZE -eq 40 ]; then
    # 40B
    LAYER=96
    HIDDEN=5888
    FFN_HIDDEN=15360
    HEAD=4
    NUM_KV_HEADS=4
elif [ $MODEL_SIZE -eq 52 ]; then
    # 52B
    LAYER=96
    HIDDEN=6656
    FFN_HIDDEN=17920
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 69 ]; then
    # 69B
    LAYER=96
    HIDDEN=7680
    FFN_HIDDEN=20480
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 88 ]; then
    # 88B
    LAYER=192
    HIDDEN=6144
    FFN_HIDDEN=16384
    HEAD=16
    NUM_KV_HEADS=16
elif [ $MODEL_SIZE -eq 110 ]; then
    # 110B
    LAYER=192
    HIDDEN=6912
    FFN_HIDDEN=18432
    HEAD=16
    NUM_KV_HEADS=16
else
    echo "Unsupported model size"
    HIDDEN=2048 
    FFN_HIDDEN=5504
    LAYER=24
    HEAD=16
    NUM_KV_HEADS=16
fi

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=20
WEIGHT_DECAY=0.1
GRAD_CLIP=0

MODEL_ARGS="
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
    --adam-beta2 0.95 \
    --initial-loss-scale $((2**$INIT_LOSS_SCALE_POWER))
"

MODEL_ARGS="${MODEL_ARGS} ${LLAMA_ARGS}"