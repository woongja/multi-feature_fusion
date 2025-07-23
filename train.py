import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from datautils.data_multi_fusion import gen_list, MultiFeatureDataset
from models.AlexNet_fusion import FusionNet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys

label_mapping = {
        "clean": 0,
        "background_noise": 1,
        "background_music": 2,
        "overlapping_speech": 3,
        "white_noise": 4,
        "pink_noise": 5,
        "pitch_shift": 6,
        "time_stretch": 7,
        "auto_tune": 8,
        "reverberation": 9
}

class EarlyStop:
    def __init__(self, patience=5, delta=0, save_path='out_checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Model saved with validation loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch(model, train_loader, optimizer,criterion, device, log_file="train_debug_log.txt"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 로그 파일 초기화 (덮어쓰기)
    with open(log_file, "w") as f:
        f.write("Epoch Debug Logs\n")
        f.write("=" * 50 + "\n")

    for batch_idx, (spec, mfcc, f0, label) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
        spec, mfcc, f0, label = spec.to(device), mfcc.to(device), f0.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(spec, mfcc, f0)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        correct += (logits.argmax(dim=1) == label).sum().item()
        total += label.size(0)

        # 디버그 정보 저장
        with open(log_file, "a") as f:
            f.write(f"Batch {batch_idx + 1}:\n")
            f.write(f"  Logits shape: {logits.shape}, Labels shape: {label.shape}\n")
            f.write(f"  Logits sample: {logits[0].detach().cpu().numpy()}\n")
            f.write(f"  Label sample: {label[0].cpu().numpy()}\n")
            f.write(f"  Loss: {loss.item():.4f}\n")
            f.write("-" * 50 + "\n")

    return total_loss / total, (correct / total) * 100


def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for spec, mfcc, f0, label in tqdm(eval_loader, desc="Evaluating", ncols=100):
            spec, mfcc, f0, label = spec.to(device), mfcc.to(device), f0.to(device), label.to(device)
            logits = model(spec, mfcc, f0)
            loss = criterion(logits, label)

            total_loss += loss.item() * label.size(0)
            correct += (logits.argmax(dim=1) == label).sum().item()
            total += label.size(0)

    return total_loss / total, (correct / total) * 100

def produce_evaluation_file(dataset, model, device, save_path, batch_size):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    model.eval()

    index_to_label = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for spec, mfcc, f0, utt_ids in tqdm(data_loader, ncols=100):
            spec, mfcc, f0 = spec.to(device), mfcc.to(device), f0.to(device)

            outputs = model(spec, f0, mfcc)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            batch_scores = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_indices = batch_scores.argmax(axis=1)
            predicted_labels = [index_to_label[idx] for idx in predicted_indices]

            with open(save_path, 'a+') as fh:
                for idx, utt_id in enumerate(utt_ids):
                    class_scores_str = " ".join(map(str, batch_scores[idx]))
                    predicted_label = predicted_labels[idx]
                    fh.write(f"{utt_id} {predicted_label}\n")
                    # 디버깅용:
                    # fh.write(f"{utt_id} {predicted_label} {class_scores_str}\n")

    print(f"Scores saved to {save_path}")


def produce_embedding_file(dataset, model, device, embedding_save_path, batch_size):
    """
    평가(test) 데이터셋에 대해 모델의 임베딩 벡터를 추출하여 파일로 저장합니다.
    
    각 샘플에 대해:
    utt_id, 그리고 임베딩 벡터(공백으로 구분된 숫자들)를 저장합니다.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    # 임베딩 파일을 저장할 디렉토리 생성 (없으면 생성)
    os.makedirs(embedding_save_path, exist_ok=True)

    with torch.no_grad():
        for batch_x, utt_ids in tqdm(data_loader, desc="Extracting Embeddings", ncols=100):
            batch_x = batch_x.to(device)
            # 모델의 forward가 (logits, embeddings)를 반환한다고 가정
            _, embeddings = model(batch_x)
            embeddings_np = embeddings.cpu().numpy()

            # 배치 내 각 샘플에 대해 별도 파일로 저장
            for utt_id, emb in zip(utt_ids, embeddings_np):
                # 파일 경로에서 파일명만 추출 후 확장자 제거 (예: "file.wav" -> "file")
                base_name = os.path.splitext(os.path.basename(utt_id))[0]
                save_path = os.path.join(embedding_save_path, base_name + ".npy")
                np.save(save_path, emb)
    print(f"Embeddings saved in directory: {save_path}")

def apply_portion_sampling(file_list, label_dict, portion):
    """
    랜덤 샘플링을 적용하여 데이터셋 크기를 축소합니다.
    
    Args:
    - file_list: 전체 파일 리스트
    - label_dict: 파일에 대한 라벨 딕셔너리
    - portion: 샘플링 비율 (0 < portion <= 1)
    
    Returns:
    - 샘플링된 file_list와 label_dict
    """
    if portion <= 0 or portion > 1:
        raise ValueError("Portion must be in the range (0, 1].")

    idx = range(len(file_list))
    idx = np.random.choice(idx, int(len(file_list) * portion), replace=False)
    sampled_file_list = [file_list[i] for i in idx]
    
    if len(label_dict) > 0:
        sampled_label_dict = {k: label_dict[k] for k in sampled_file_list}
    else:
        sampled_label_dict = {}

    return sampled_file_list, sampled_label_dict

def train_finetune(model, train_loader, dev_loader, optimizer, criterion, writer, args):
    """
    저장된 모델(best.pth 등)과 n-best 상태를 불러와 이어서 학습하는 함수
    """
    n_mejores = 3
    best_save_path = args.save_path if args.save_path.endswith('/') else args.save_path + '/'
    os.makedirs(best_save_path, exist_ok=True)
    # 상태 파일 경로
    state_path = os.path.join(best_save_path, 'finetune_state.npz')

    # 기본값
    bests = np.ones(n_mejores, dtype=float) * float('inf')
    best_loss = float('inf')
    not_improving = 0
    epoch = 0
    patience = args.early_stop_patience

    # 상태 불러오기
    if os.path.exists(state_path):
        state = np.load(state_path)
        bests = state['bests']
        best_loss = state['best_loss']
        not_improving = int(state['not_improving'])
        epoch = int(state['epoch'])
        print(f"[Resume] epoch={epoch}, not_improving={not_improving}, best_loss={best_loss}, bests={bests}")
    else:
        print("[Start new finetune]")

    while not_improving < patience and epoch < args.num_epochs:
        print(f'######## Finetune Epoch {epoch} ########')
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, log_file="train_debug_log.txt")
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, args.device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.2f}%")

        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('val_loss', dev_loss, epoch + 1)
        writer.add_scalar('train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('val_accuracy', dev_acc, epoch + 1)

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(best_save_path, 'best.pth'))
            print('New best epoch')
            not_improving = 0
        else:
            not_improving += 1

        for i in range(n_mejores):
            if bests[i] > dev_loss:
                for t in range(n_mejores - 1, i, -1):
                    bests[t] = bests[t - 1]
                    src = os.path.join(best_save_path, f'best_{t-1}.pth')
                    dst = os.path.join(best_save_path, f'best_{t}.pth')
                    if os.path.exists(src):
                        os.system(f'cp {src} {dst}')
                bests[i] = dev_loss
                torch.save(model.state_dict(), os.path.join(best_save_path, f'best_{i}.pth'))
                break

        # 상태 저장
        np.savez(state_path, bests=bests, best_loss=best_loss, not_improving=not_improving, epoch=epoch+1)

        print(f'\n{epoch} - {dev_loss}')
        print('n-best loss:', bests)
        epoch += 1

    print('Total epochs (finetune):', epoch)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Noise Classifier")
    parser.add_argument("--protocol_file", type=str, required=True, help="Path to protocol file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of output classes")
    parser.add_argument('--input_height', type=int, default=128)
    parser.add_argument('--input_width', type=int, default=126)
    parser.add_argument('--f0_len', type=int, default=126)

    parser.add_argument("--is_train", action='store_true', default=False, help="Set True for training")
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval mode')
    parser.add_argument("--is_dev", type=bool, default=False, help="Set True for dev dataset evaluation")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--save_path", type=str, default="checkpoint.pth", help="Path to save the best model")
    parser.add_argument("--save_results", type=str, default="eval_result.txt", help="Path to save the evaluation results")
    parser.add_argument("--log_dir", type=str, default="runs", help="Path to save tensorboard logs")
    parser.add_argument("--database_path", type=str, default="/path/your/dataset", help="Path to Database")
    parser.add_argument('--model_path', type=str, default=None, help='Model checkpoint')
    parser.add_argument("--save_embeddings", action='store_true', default=False, help="Save embedding vectors for the test dataset")
    parser.add_argument("--embedding_save_path", type=str, default="/embeddings", help="Path to save embedding vectors")
    parser.add_argument('--finetune', action='store_true', default=False, help='Continue training from checkpoint and n-best state')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Train 또는 Fine-tune
    
    model = FusionNet(
        num_classes=args.num_classes, 
        branch_output_dim=1024,
        spec_shape=(1, args.input_height, args.input_width),
        mfcc_shape=(1, 13, args.input_width),
        f0_len=args.f0_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # model = nn.DataParallel(
    #     model,
    #     device_ids=[1,0],
    #     output_device=1
    # )
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded: {args.model_path}")
    
    if args.is_eval:
        eval_files = gen_list(args.protocol_file, is_eval=True)
        eval_dataset = MultiFeatureDataset(eval_files, labels=None, is_train=False, is_eval=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

        produce_evaluation_file(eval_dataset, model, device, args.save_results, args.batch_size)

        if args.save_embeddings:
            produce_embedding_file(eval_dataset, model, device, args.embedding_save_path, args.batch_size)

        sys.exit(0)
        
    if args.is_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        writer = SummaryWriter(log_dir=args.log_dir)
        early_stop = EarlyStop(patience=args.early_stop_patience, save_path=args.save_path)

        train_labels, train_files = gen_list(args.protocol_file, is_train=True)
        train_dataset = MultiFeatureDataset(train_files, train_labels, is_train=True, is_eval=False, enable_cache=True, train_random_start=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        dev_labels, dev_files = gen_list(args.protocol_file, is_dev=True)
        dev_dataset = MultiFeatureDataset(dev_files, dev_labels, is_train=False, is_eval=False, enable_cache=True, train_random_start=False)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

        best_loss = float('inf')
        for epoch in range(1, args.num_epochs + 1):
            print(f"\nEpoch {epoch}/{args.num_epochs}")
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, log_file="train_debug_log.txt")
            val_loss, val_acc = evaluate(model, dev_loader, criterion, device)

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)

            # early stopping 체크
            early_stop(val_loss, model)
            if early_stop.early_stop:
                print("Early stopping triggered.")
                break
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), args.save_path)
                print(f"New best model saved at epoch {epoch}")

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        writer.close()