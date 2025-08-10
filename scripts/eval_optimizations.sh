#!/bin/bash

RUN=./examples/run.sh
TRAIN_ITER=2
LOG_ITER=2
PARTITION=spipe-v100

# Fig 15: Impact of progressively adding system features to SPipe

sbatch -J CFG0 -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -y 1 -q 1
sbatch -J CFG1 -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -y 1
sbatch -J CFG2 -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -y 1
sbatch -J CFG3 -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -y 1
sbatch -J CFG4 -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2
sbatch -J CFG5 -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -o stage
# Ideal config. in Fig. 15 is CFG5 + skipping activation checkpoint communication. It's ran by commenting out the activation checkpoint communication code (function `comm_ckpt`) in SPipe schedule (`spipe_schedule.py`). 