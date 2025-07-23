#!/bin/bash

# 기본 설정
PROTOCOL_FILE="/home/woongjae/noise-tracing/wj_model/protocol/protocol_train_dev.txt"
BATCH_SIZE=128
NUM_EPOCHS=1000
LEARNING_RATE=1e-5
NUM_CLASSES=10  # 라벨 클래스 개수
SAVE_PATH="out/mfcc/"
EARLY_STOP_PATIENCE=5
LOG_DIR="runs/mfcc"  # TensorBoard 로그 디렉토리
finetune=False  # 이어서 학습 여부


# Training 실행
CUDA_VISIBLE_DEVICES=2 python train_cnn_lstm.py \
    --protocol_file $PROTOCOL_FILE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_classes $NUM_CLASSES \
    --is_train True \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --save_path $SAVE_PATH \
    --log_dir $LOG_DIR

# 2. 이어서 학습 (예시: best_0.pth에서 이어서)
# CUDA_VISIBLE_DEVICES=1 python train_cnn_lstm.py \
#     --protocol_file $PROTOCOL_FILE \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --learning_rate $LEARNING_RATE \
#     --num_classes $NUM_CLASSES \
#     --finetune \
#     --model_path ${SAVE_PATH}best.pth \
#     --early_stop_patience $EARLY_STOP_PATIENCE \
#     --save_path $SAVE_PATH \
#     --log_dir $LOG_DIR