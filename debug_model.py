import torch
from models.AlexNet_fusion import FusionNet

# 현재 모델 생성
model = FusionNet(
    num_classes=10, 
    branch_output_dim=1024,
    spec_shape=(1, 128, 126),
    mfcc_shape=(1, 13, 126),
    f0_len=126
)

print("현재 모델의 키:")
for key in model.state_dict().keys():
    print(f"  {key}")

print("\n" + "="*50)

# 저장된 모델 로드
try:
    checkpoint = torch.load("/home/woongjae/noise-tracing/muti-feature_fusion/out/best_model_old.pth", map_location='cpu')
    print("저장된 모델의 키:")
    for key in checkpoint.keys():
        print(f"  {key}")
        
    print("\n" + "="*50)
    print("차이점:")
    
    current_keys = set(model.state_dict().keys())
    saved_keys = set(checkpoint.keys())
    
    print("현재 모델에만 있는 키:")
    for key in current_keys - saved_keys:
        print(f"  + {key}")
    
    print("저장된 모델에만 있는 키:")
    for key in saved_keys - current_keys:
        print(f"  - {key}")
        
except Exception as e:
    print(f"모델 로드 에러: {e}")