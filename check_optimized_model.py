import torch
from models.AlexNet_fusion_optimized import FusionNet

def analyze_model_parameters(model):
    """모델의 파라미터 분석"""
    total_params = 0
    print("레이어별 파라미터 수:")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:<40} {param_count:>12,} {list(param.shape)}")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")  # float32 기준
    return total_params

if __name__ == "__main__":
    model = FusionNet(
        num_classes=10, 
        branch_output_dim=1024,
        spec_shape=(1, 1025, 126),
        mfcc_shape=(1, 13, 401),
        f0_len=126
    )
    
    analyze_model_parameters(model)