import torch
import torch.nn as nn


class NoiseClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, seq_len):
        super().__init__()
        # 1D Conv Feature Extractor (입력: [B, T, F])
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=48, kernel_size=5, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 32, kernel_size=5, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=5, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.dropout = nn.Dropout(0.5)
        # Sequence Learning (LSTM)
        self.lstm = nn.LSTM(input_size=16, hidden_size=128, num_layers=3, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: [B, T, F]
        # x = x.permute(0,2,1)           # [B, F, T]
        x = self.cnn(x)                # [B, 16, T']
        x = x.permute(0,2,1)           # [B, T', 16]
        x = self.dropout(x)
        out, _ = self.lstm(x)          # [B, T', 128]
        x = out[:,-1,:]                # 마지막 타임스텝
        x = self.fc(x)                 # [B, num_classes]
        return x
    
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, label):
        return self.loss_fn(output, label)
    
class NoiseClassifier_2D(nn.Module):
    def __init__(self, num_classes, in_ch=1, n_mfcc=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            input_size=(n_mfcc//8)*256,  # CNN 마지막 채널*feature
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attn_fc = nn.Linear(256*2, 1)  # Attention layer
        self.fc = nn.Sequential(
            nn.Linear(256*2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, n_mfcc, T]
        x = self.cnn(x)  # [B, 256, n_mfcc//8, T//8]
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, -1)  # [B, T', C*F']
        lstm_out, _ = self.lstm(x)  # [B, T', 512]
        # Attention pooling
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)  # [B, T', 1]
        x = torch.sum(lstm_out * attn_weights, dim=1)  # [B, 512]
        x = self.fc(x)
        return x
