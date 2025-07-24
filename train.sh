#!/bin/bash

# 기본 설정
PROTOCOL_FILE="/home/woongjae/noise-tracing/wj_model/protocol/protocol_train_dev.txt"
BATCH_SIZE=32
NUM_EPOCHS=100
LEARNING_RATE=1e-4
NUM_CLASSES=10
SAVE_PATH="out/best_model.pth"
EARLY_STOP_PATIENCE=5
LOG_DIR="runs/alexnetfusion"  # TensorBoard 로그 디렉토리
input_height=128   # freq axis for spec
input_width=126    # time axis
f0_len=126      # F0 time frame 수

# Training 실행
CUDA_VISIBLE_DEVICES=1 python train.py \
    --is_train \
    --protocol_file $PROTOCOL_FILE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_classes $NUM_CLASSES \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --save_path $SAVE_PATH \
    --log_dir $LOG_DIR \
    --input_height $input_height \
    --input_width $input_width \
    --f0_len $f0_len

echo "Results saved to ${SAVE_PATH}"