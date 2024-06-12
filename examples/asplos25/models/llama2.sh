#!/bin/bash

# Model spec
if [ $MODEL_SIZE -eq 7 ]; then
    # 7B
    LAYER=32
    HIDDEN=4096
    FFN_HIDDEN=11008
    HEAD=32
    NUM_KV_HEADS=32
elif [ $MODEL_SIZE -eq 13 ]; then
    # 13B
    LAYER=40
    HIDDEN=5120
    FFN_HIDDEN=13824
    HEAD=40
    NUM_KV_HEADS=40
elif [ $MODEL_SIZE -eq 52 ]; then
    # 52B
    LAYER=64
    HIDDEN=8192
    FFN_HIDDEN=22016
    HEAD=4
    NUM_KV_HEADS=4
elif [ $MODEL_SIZE -eq 81 ]; then
    # 81B
    LAYER=64
    HIDDEN=10240
    FFN_HIDDEN=27648
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 121 ]; then
    # 121B
    LAYER=96
    HIDDEN=10240
    FFN_HIDDEN=27648
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 175 ]; then
    # 127B
    LAYER=96
    HIDDEN=13312
    FFN_HIDDEN=35840
    HEAD=8
    NUM_KV_HEADS=8
elif [ $MODEL_SIZE -eq 200 ]; then
    # 200B
    LAYER=96
    HIDDEN=13312
    FFN_HIDDEN=35840
    HEAD=8
    NUM_KV_HEADS=8
else
    HIDDEN=2048 
    FFN_HIDDEN=5504
    LAYER=24
    HEAD=16
    NUM_KV_HEADS=16
fi

SEQ=4096
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
    --adam-beta2 0.95
"

MODEL_ARGS="${MODEL_ARGS} ${LLAMA_ARGS}"