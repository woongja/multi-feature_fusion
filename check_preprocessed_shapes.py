import torch
import torch.nn as nn

# 실제 전처리된 데이터의 shape 확인
def calculate_conv_output_size():
    print("=== Spectrogram Branch ===")
    # Spec shape: [1, 1025, 126]
    spec_features = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    dummy_spec = torch.zeros(1, 1, 1025, 126)
    spec_out = spec_features(dummy_spec)
    spec_flatten = spec_out.view(1, -1)
    print(f"Spec input: {dummy_spec.shape}")
    print(f"Spec conv output: {spec_out.shape}")
    print(f"Spec flatten size: {spec_flatten.size(1)}")
    
    print("\n=== MFCC Branch ===")
    # MFCC shape: [1, 13, 401]
    mfcc_features = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
        nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
    )
    
    dummy_mfcc = torch.zeros(1, 1, 13, 401)
    mfcc_out = mfcc_features(dummy_mfcc)
    mfcc_flatten = mfcc_out.view(1, -1)
    print(f"MFCC input: {dummy_mfcc.shape}")
    print(f"MFCC conv output: {mfcc_out.shape}")
    print(f"MFCC flatten size: {mfcc_flatten.size(1)}")
    
    print("\n=== F0 Branch ===")
    # F0 shape: [1, 126]
    f0_conv = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )
    
    dummy_f0 = torch.zeros(1, 1, 126)
    f0_out = f0_conv(dummy_f0)
    f0_flatten = f0_out.view(1, -1)
    print(f"F0 input: {dummy_f0.shape}")
    print(f"F0 conv output: {f0_out.shape}")
    print(f"F0 flatten size: {f0_flatten.size(1)}")
    
    # FC 레이어 파라미터 계산
    print("\n=== FC Layer Parameters ===")
    spec_fc_params = spec_flatten.size(1) * 2048 + 2048 + 2048 * 1024 + 1024
    mfcc_fc_params = mfcc_flatten.size(1) * 2048 + 2048 + 2048 * 1024 + 1024
    f0_fc_params = f0_flatten.size(1) * 1024 + 1024
    
    print(f"Spec FC params: {spec_fc_params:,}")
    print(f"MFCC FC params: {mfcc_fc_params:,}")
    print(f"F0 FC params: {f0_fc_params:,}")
    print(f"Total FC params: {spec_fc_params + mfcc_fc_params + f0_fc_params:,}")

if __name__ == "__main__":
    calculate_conv_output_size()