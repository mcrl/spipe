#!/bin/bash

# Model spec
if [ $MODEL_SIZE -eq 1 ]; then
    # 1.3B
    LAYER=24
    HIDDEN=2048
    FFN_HIDDEN=5504
    HEAD=16
    NUM_KV_HEADS=16
elif [ $MODEL_SIZE -eq 7 ]; then
    # 7B
    LAYER=32
    HIDDEN=4096
    FFN_HIDDEN=11008
    HEAD=32
    NUM_KV_HEADS=32
elif [ $MODEL_SIZE -eq 10 ]; then
    # 10B
    LAYER=48
    HIDDEN=4096
    FFN_HIDDEN=13564
    HEAD=2
    NUM_KV_HEADS=2
elif [ $MODEL_SIZE -eq 13 ]; then
    # 13B
    LAYER=40
    HIDDEN=4096
    FFN_HIDDEN=13564
    HEAD=2
    NUM_KV_HEADS=2
elif [ $MODEL_SIZE -eq 15 ]; then
    # 15B
    LAYER=48
    HIDDEN=5120
    FFN_HIDDEN=13564
    HEAD=2
    NUM_KV_HEADS=2
elif [ $MODEL_SIZE -eq 58 ]; then
    # 58B
    LAYER=128
    HIDDEN=6144
    FFN_HIDDEN=16384
    HEAD=4
    NUM_KV_HEADS=4
elif [ $MODEL_SIZE -eq 80 ]; then
    # 80B
    LAYER=128
    HIDDEN=7168
    FFN_HIDDEN=18944
    HEAD=4
    NUM_KV_HEADS=4
elif [ $MODEL_SIZE -eq 103 ]; then
    # 103B
    LAYER=128
    HIDDEN=8192
    FFN_HIDDEN=27648
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 131 ]; then
    # 131B
    LAYER=128
    HIDDEN=9216
    FFN_HIDDEN=30208
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 162 ]; then
    # 162B
    LAYER=128
    HIDDEN=10240
    FFN_HIDDEN=32768
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 154 ]; then
    # 154B
    LAYER=192
    HIDDEN=8192
    FFN_HIDDEN=27648
    HEAD=16
    NUM_KV_HEADS=16
elif [ $MODEL_SIZE -eq 196 ]; then
    # 196B
    LAYER=128
    HIDDEN=11264
    FFN_HIDDEN=35840
    HEAD=16
    NUM_KV_HEADS=16
elif [ $MODEL_SIZE -eq 233 ]; then
    # 233B
    LAYER=128
    HIDDEN=12288
    FFN_HIDDEN=39424
    HEAD=16
    NUM_KV_HEADS=16
else
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