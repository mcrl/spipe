#!/bin/bash
pip install numpy==1.24.3
pip install datasets nltk
export MEGATRON_PATH=$SPIPE_ROOT

python $SPIPE_ROOT/data/convert_dataset_hf_to_mg.py \
  --dataset_name openwebtext \
  --cache_dir $SPIPE_AEC_ROOT/workspace/datasets --preprocessing_num_workers 16 \
  --output $SPIPE_AEC_ROOT/workspace/datasets/openwebtext.jsonl

# preprocessing for GPT
python $SPIPE_ROOT/tools/preprocess_data.py \
  --input $SPIPE_AEC_ROOT/workspace/datasets/openwebtext.jsonl \
  --output-prefix $SPIPE_AEC_ROOT/workspace/datasets/openwebtext \
  --vocab-file $SPIPE_ROOT/data/gpt2-vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file $SPIPE_ROOT/data/gpt2-merges.txt \
  --append-eod \
  --workers 64 \
  --chunk-size 64
