#!/bin/bash

# 기본 설정
PROTOCOL_FILE="/home/woongjae/noise-tracing/wj_model/protocol/protocol_eval.txt"
BATCH_SIZE=128
NUM_CLASSES=10
MODEL_PATH="/home/woongjae/noise-tracing/muti-feature_fusion/out/best_model_old.pth"
SAVE_RESULTS="/home/woongjae/noise-tracing/muti-feature_fusion/results/eval_timit.txt"
EMBEDDING_SAVE_PATH="/home/woongjae/noise-tracing/wj_model/embeddings"
input_height=128   # freq axis for spec
input_width=126    # time axis
f0_len=126      # F0 time frame 수

echo "Evaluating classification model..."

CUDA_VISIBLE_DEVICES=1 python train.py \
    --is_eval \
    --protocol_file $PROTOCOL_FILE \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --model_path $MODEL_PATH \
    --save_results $SAVE_RESULTS \
    --input_height $input_height \
    --input_width $input_width \
    --f0_len $f0_len

echo "Results saved to ${SAVE_RESULTS}"
