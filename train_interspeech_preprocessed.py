import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from datautils.data_preprocessed import PreprocessedDatasetManager
from models.AlexNet_fusion import FusionNet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys

label_mapping = {
    "clean": 0,
    "background_noise": 1,
    "background_music": 2,
    "gaussian_noise": 3,
    "band_pass_filter": 4,
    "manipulation": 5,
    "auto_tune": 6,
    "echo": 7,
    "reverberation": 8
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

def train_epoch(model, train_loader, optimizer, criterion, device, log_file="train_debug_log.txt"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

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
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    model.eval()

    index_to_label = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for spec, mfcc, f0, utt_ids in tqdm(data_loader, ncols=100):
            spec, mfcc, f0 = spec.to(device), mfcc.to(device), f0.to(device)

            outputs = model(spec, mfcc, f0)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            batch_scores = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_indices = batch_scores.argmax(axis=1)
            predicted_labels = [index_to_label[idx] for idx in predicted_indices]

            with open(save_path, 'a+') as fh:
                for idx, utt_id in enumerate(utt_ids):
                    predicted_label = predicted_labels[idx]
                    fh.write(f"{utt_id} {predicted_label}\n")

    print(f"Scores saved to {save_path}")


def train_finetune(model, train_loader, dev_loader, optimizer, criterion, writer, args):
    n_mejores = 3
    best_save_path = args.save_path if args.save_path.endswith('/') else args.save_path + '/'
    os.makedirs(best_save_path, exist_ok=True)
    state_path = os.path.join(best_save_path, 'finetune_state.npz')

    bests = np.ones(n_mejores, dtype=float) * float('inf')
    best_loss = float('inf')
    not_improving = 0
    epoch = 0
    patience = args.early_stop_patience

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
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device, log_file="train_debug_log.txt")
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

        np.savez(state_path, bests=bests, best_loss=best_loss, not_improving=not_improving, epoch=epoch+1)

        print(f'\n{epoch} - {dev_loss}')
        print('n-best loss:', bests)
        epoch += 1

    print('Total epochs (finetune):', epoch)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Noise Classifier with Preprocessed Data")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Path to preprocessed data directory")
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
    parser.add_argument('--model_path', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--finetune', action='store_true', default=False, help='Continue training from checkpoint and n-best state')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Using device: {device}")
    
    # Initialize dataset manager
    dataset_manager = PreprocessedDatasetManager(args.preprocessed_dir)
    info = dataset_manager.get_info()
    
    print("Dataset Information:")
    print("=" * 50)
    for subset, stats in info['datasets'].items():
        if 'error' not in stats:
            print(f"{subset}: {stats['total_samples']} samples")
    
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
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded: {args.model_path}")
    
    if args.is_eval:
        eval_dataset = dataset_manager.get_dataset('eval', is_eval=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, pin_memory=True)
        
        produce_evaluation_file(eval_dataset, model, device, args.save_results, args.batch_size)
        sys.exit(0)
        
    if args.is_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        writer = SummaryWriter(log_dir=args.log_dir)
        early_stop = EarlyStop(patience=args.early_stop_patience, save_path=args.save_path)

        train_dataset = dataset_manager.get_dataset('train', is_eval=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=args.num_workers, pin_memory=True)

        dev_dataset = dataset_manager.get_dataset('dev', is_eval=False)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, 
                               num_workers=args.num_workers, pin_memory=True)

        print(f"\nTraining with:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Dev samples: {len(dev_dataset)}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Workers: {args.num_workers}")

        if args.finetune:
            train_finetune(model, train_loader, dev_loader, optimizer, criterion, writer, args)
        else:
            best_loss = float('inf')
            for epoch in range(1, args.num_epochs + 1):
                print(f"\nEpoch {epoch}/{args.num_epochs}")
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, log_file="train_debug_log.txt")
                val_loss, val_acc = evaluate(model, dev_loader, criterion, device)

                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Val', val_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Val', val_acc, epoch)

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