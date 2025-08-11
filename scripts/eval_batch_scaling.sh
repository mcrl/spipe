#!/bin/bash

RUN=${SPIPE_ROOT:-.}/examples/run.sh
TRAIN_ITER=2
LOG_ITER=2
PARTITION=spipe-v100

# Fig 12: Effect of scaling micro-batch size (muBS)

## Model size: 10B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 32 -v 1 -w 16 -z 1024 -u 2

## Model size: 19B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 32 -v 1 -w 16 -z 1024 -u 2

## Model size: 30B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 64 -v 1 -w 16 -z 1024 -u 2

## Model size: 40B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 64 -v 1 -w 16 -z 1024 -u 2

## Model size: 52B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 128 -v 1 -w 16 -z 1024 -u 2

## Model size: 69B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 128 -v 1 -w 16 -z 1024 -u 2

## Model size: 88B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 256 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 256 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 256 -v 1 -w 16 -z 1024 -u 2

## Model size: 110B / muBS: 1->2->4->8

### SPipe
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 1 -g 256 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 4 -g 256 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 8 -g 256 -v 1 -w 16 -z 1024 -u 2

######################################################

# Fig 14: Effect of scaling mini-batch size (MBS)

## Model size: 10B / MBS: 8->16->24->32

### SPipe
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 24 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 10 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 24 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 10 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2

## Model size: 19B / MBS: 8->16->24->32

### SPipe
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 24 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 1 $RUN -j spipe -n llama2 -s 19 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 8 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 24 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 1 $RUN -j mobius -n llama2 -s 19 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2

## Model size: 30B / MBS: 16->32->48->64

### SPipe
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 48 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 30 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 48 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 30 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2

## Model size: 40B / MBS: 16->32->48->64

### SPipe
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 48 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 2 $RUN -j spipe -n llama2 -s 40 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 16 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 48 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 2 $RUN -j mobius -n llama2 -s 40 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2

## Model size: 52B / MBS: 32->64->96->128

### SPipe
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 96 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 52 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 96 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 52 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2

## Model size: 69B / MBS: 32->64->96->128

### SPipe
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 96 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 4 $RUN -j spipe -n llama2 -s 69 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 32 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 96 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 4 $RUN -j mobius -n llama2 -s 69 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2

## Model size: 88B / MBS: 64->128->192->256

### SPipe
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 192 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 88 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 192 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 88 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2

## Model size: 110B / MBS: 64->128->192->256

### SPipe
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 192 -v 1 -w 16 -z 1024 -u 2 -o stage
sbatch -J SPipe -p $PARTITION -N 8 $RUN -j spipe -n llama2 -s 110 -f 2 -b 6 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2 -o stage

### Mobius
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 64 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 128 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 192 -v 1 -w 16 -z 1024 -u 2
sbatch -J Mobius -p $PARTITION -N 8 $RUN -j mobius -n llama2 -s 110 -f 2 -b 2 -t $TRAIN_ITER -l $LOG_ITER -m 2 -g 256 -v 1 -w 16 -z 1024 -u 2
