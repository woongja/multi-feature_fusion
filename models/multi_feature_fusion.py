import torch
import torch.nn as nn
import torch.nn.functional as F

# 최적화된 Spectrogram Branch
class SpectrogramBranch(nn.Module):
    def __init__(self, in_channels=1, output_dim=1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
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
        
        # Global Average Pooling 사용하여 파라미터 수 대폭 감소
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)  # [batch, 256, 1, 1]
        x = torch.flatten(x, 1)      # [batch, 256]
        x = self.fc(x)
        return x

# 최적화된 MFCC Branch
class MFCCBranch(nn.Module):
    def __init__(self, in_channels=1, output_dim=1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        
        # Global Average Pooling 사용
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)  # [batch, 256, 1, 1]
        x = torch.flatten(x, 1)      # [batch, 256]
        x = self.fc(x)
        return x

# F0 Branch (동일하게 유지)
class F0Branch(nn.Module):
    def __init__(self, input_len, output_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        dummy_input = torch.zeros(1, 1, input_len)
        out = self.conv(dummy_input)
        flatten_dim = out.shape[1] * out.shape[2]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 최적화된 Fusion 네트워크
class FusionNet(nn.Module):
    def __init__(self, num_classes, branch_output_dim=1024, spec_shape=(1,1025,126), mfcc_shape=(1,13,401), f0_len=126):
        super().__init__()
        self.spec_branch = SpectrogramBranch(in_channels=spec_shape[0], output_dim=branch_output_dim)
        self.mfcc_branch = MFCCBranch(in_channels=mfcc_shape[0], output_dim=branch_output_dim)
        self.f0_branch = F0Branch(f0_len, output_dim=branch_output_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(branch_output_dim * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, spec, mfcc, f0):
        spec_feat = self.spec_branch(spec)
        mfcc_feat = self.mfcc_branch(mfcc)
        f0_feat = self.f0_branch(f0)
        fused = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)
        out = self.classifier(fused)
        return out