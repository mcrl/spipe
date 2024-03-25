#!/bin/bash

ulimit -v unlimited

## torch dist.
export MASTER_ADDR=$(echo $UNWRAPPED_NODELIST | awk '{print $1}')
export MASTER_PORT=6000
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1


DISTRIBUTED_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size $NP \
    --distributed-backend nccl \
    --overlap-p2p-communication \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT
"

GPT_ARGS="
    --no-initialization \
    --untie-embeddings-and-output-weights \
    --sequence-parallel \
    --num-layers $LAYER \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEAD \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $(( $MBS * $NP )) \
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
    --no-contiguous-buffers-in-local-ddp
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

EXEC_CMD="python ${MEGATRON_PATH}/pretrain_gpt.py ${EXTRA_ARGS} ${DISTRIBUTED_ARGS} ${GPT_ARGS} ${DATA_ARGS}"

if [ ${NSYS_ENABLE} == "YES" ]; then
    EXEC_CMD="${NSYS} profile -t cuda,nvtx -o ${NSYS_OUTPUT}_%q{OMPI_COMM_WORLD_RANK} --force-overwrite true ${EXEC_CMD}"
fi

${MPIRUN} -np $NP -host $HOSTS $MPI_OPTIONS ${EXEC_CMD}
