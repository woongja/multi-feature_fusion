#!/bin/bash

# 기본 설정
PROTOCOL_FILE="/home/woongjae/multi-feature_fusion/Datasets/protocol_train_dev.txt"
PREPROCESSED_DIR="./preprocessed_data"
BATCH_SIZE=32
NUM_CLASSES=9
MODEL_PATH="out/best_model.pth"
RESULTS_FILE="eval_results.txt"
input_height=128   # freq axis for spec
input_width=126    # time axis
f0_len=126      # F0 time frame 수
NUM_WORKERS=8

# 모델 파일 존재 확인
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: 모델 파일을 찾을 수 없습니다: $MODEL_PATH"
    exit 1
fi

# 전처리된 데이터 존재 확인
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "Error: 전처리된 데이터를 찾을 수 없습니다: $PREPROCESSED_DIR"
    echo "먼저 전처리를 실행하세요."
    exit 1
fi

# Evaluation 실행
CUDA_VISIBLE_DEVICES=3 python train_interspeech_preprocessed.py \
    --is_eval \
    --preprocessed_dir $PREPROCESSED_DIR \
    --model_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --num_classes $NUM_CLASSES \
    --save_results $RESULTS_FILE \
    --input_height $input_height \
    --input_width $input_width \
    --f0_len $f0_len \
    --num_workers $NUM_WORKERS

echo "평가 완료! 결과는 $RESULTS_FILE 에 저장되었습니다."