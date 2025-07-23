#!/bin/bash

# 기본 설정
BATCH_SIZE=128
num_classes=10
model_path="/home/woongjae/noise-tracing/wj_model/out/mfcc/3/best.pth"
# 결과 저장 디렉토리
embedding_save_path="/home/woongjae/noise-tracing/wj_model/embeddings"

protocol_file="/home/woongjae/noise-tracing/wj_model/protocol/protocol_eval.txt"
save_results="/home/woongjae/noise-tracing/wj_model/results/eval_cnn2d+lstm_3_result.txt"

echo "Evaluating classification model..."
CUDA_VISIBLE_DEVICES=0 python /home/woongjae/noise-tracing/wj_model/train_cnn_lstm.py \
    --is_eval \
    --protocol_file $protocol_file \
    --num_classes $num_classes \
    --batch_size $BATCH_SIZE \
    --model_path $model_path \
    --save_results $save_results

echo "Results saved to ${save_results}"
