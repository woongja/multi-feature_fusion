#!/bin/bash

# 기본 설정
PROTOCOL_FILE="/home/woongjae/noise-tracing/wj_model/mlaad_shortcutASV_result/protocol_mlaad.txt"
BATCH_SIZE=128
NUM_CLASSES=9
MODEL_PATH="/home/woongjae/noise-tracing/multi-feature_fusion/out/best_model_interspeech_2.pth"
RESULTS_FILE="/home/woongjae/noise-tracing/multi-feature_fusion/results/eval_interspeech_mlaad.txt"
EMBEDDING_SAVE_PATH="/home/woongjae/noise-tracing/wj_model/embeddings"
input_height=128   # freq axis for spec
input_width=126    # time axis
f0_len=126      # F0 time frame 수

echo "Evaluating classification model..."

CUDA_VISIBLE_DEVICES=0 python train_interspeech.py \
    --is_eval \
    --protocol_file $PROTOCOL_FILE \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --model_path $MODEL_PATH \
    --save_results $RESULTS_FILE \
    --input_height $input_height \
    --input_width $input_width \
    --f0_len $f0_len

echo "Results saved to ${RESULTS_FILE}"
