#!/bin/bash

export CSQA_DIR=../datasets/csqa_new

/home/bill/miniconda3/bin/python run_csqa_bert.py \
     --bert_model bert-large-uncased \
     --do_train \
     --do_lower_case \
     --do_eval \
     --data_dir $CSQA_DIR \
     --train_batch_size 60 \
     --eval_batch_size 60 \
     --learning_rate 1e-4 \
     --num_train_epochs 100.0 \
     --max_seq_length 100 \
     --output_dir ./models/ \
     --save_model_name bert_large_1e-4 \
     --gradient_accumulation_steps 4
