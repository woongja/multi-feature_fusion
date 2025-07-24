import argparse
import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import hashlib
from datautils.data_multi_fusion_interspeech import gen_list, fix_length


def extract_features(wav_np, sr=16000):
    """Extract all features from waveform"""
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    window = 'hamming'
    
    # Spectrogram
    stft = librosa.stft(wav_np, n_fft=n_fft, hop_length=hop_length, window=window)
    spec_db = librosa.amplitude_to_db(np.abs(stft))
    spec_db = torch.from_numpy(spec_db).float().unsqueeze(0)

    # MFCC
    frame_size = int(0.025 * sr)  # 25ms
    hop_size = int(0.010 * sr)    # 10ms
    mfcc = librosa.feature.mfcc(y=wav_np, sr=sr, n_mfcc=13,
                                n_fft=frame_size, hop_length=hop_size, n_mels=n_mels)
    mfcc = torch.from_numpy(mfcc).float().unsqueeze(0)

    # F0 Extraction
    f0 = librosa.yin(wav_np, fmin=50, fmax=600, sr=sr,
                    frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0)
    f0 = torch.from_numpy(f0).float().unsqueeze(0)
    
    return spec_db, mfcc, f0


def preprocess_dataset(protocol_file, output_dir, subset_name, sr=16000, target_len=64000):
    """Preprocess a dataset subset and save features as pt files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Generate file list based on subset
    if subset_name == "train":
        labels, file_list = gen_list(protocol_file, is_train=True)
    elif subset_name == "dev":
        labels, file_list = gen_list(protocol_file, is_dev=True)
    elif subset_name == "eval":
        file_list = gen_list(protocol_file, is_eval=True)
        labels = None
    else:
        raise ValueError(f"Unknown subset: {subset_name}")
    
    print(f"Processing {len(file_list)} files for {subset_name} set...")
    
    # Index to track processed files
    index_data = []
    failed_files = []
    
    for i, utt_id in enumerate(tqdm(file_list, desc=f"Processing {subset_name}")):
        try:
            # Load audio
            wav, fs = librosa.load(utt_id, sr=sr)
            
            # Fix length (no random start for preprocessing)
            wav = fix_length(torch.tensor(wav), target_len, random_start=False)
            wav_np = wav.numpy()
            
            # Extract features
            spec_db, mfcc, f0 = extract_features(wav_np, sr)
            
            # Generate unique filename based on original path
            path_hash = hashlib.md5(utt_id.encode()).hexdigest()
            feature_filename = f"{path_hash}.pt"
            feature_path = features_dir / feature_filename
            
            # Save features
            features = {
                'spec_db': spec_db,
                'mfcc': mfcc,
                'f0': f0,
                'original_path': utt_id
            }
            torch.save(features, feature_path)
            
            # Add to index
            index_entry = {
                'feature_file': str(feature_path),
                'original_path': utt_id,
                'subset': subset_name
            }
            
            if labels is not None:
                index_entry['label'] = labels[utt_id]
            
            index_data.append(index_entry)
            
        except Exception as e:
            print(f"Failed to process {utt_id}: {e}")
            failed_files.append(utt_id)
            continue
    
    # Save index file
    index_file = output_dir / f"{subset_name}_index.json"
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Save failed files log
    if failed_files:
        failed_file = output_dir / f"{subset_name}_failed.txt"
        with open(failed_file, 'w') as f:
            for failed in failed_files:
                f.write(f"{failed}\n")
        print(f"Failed to process {len(failed_files)} files. See {failed_file}")
    
    print(f"Preprocessing complete for {subset_name}:")
    print(f"  - Processed: {len(index_data)} files")
    print(f"  - Failed: {len(failed_files)} files")
    print(f"  - Index saved to: {index_file}")
    print(f"  - Features saved to: {features_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio features for training")
    parser.add_argument("--protocol_file", type=str, required=True, 
                       help="Path to protocol file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for preprocessed features")
    parser.add_argument("--subset", type=str, choices=["train", "dev", "eval", "all"],
                       default="all", help="Which subset to preprocess")
    parser.add_argument("--sr", type=int, default=16000, 
                       help="Sample rate")
    parser.add_argument("--target_len", type=int, default=64000,
                       help="Target length for audio segments")
    
    args = parser.parse_args()
    
    if args.subset == "all":
        subsets = ["train", "dev", "eval"]
    else:
        subsets = [args.subset]
    
    for subset in subsets:
        print(f"\n{'='*50}")
        print(f"Processing {subset.upper()} set")
        print(f"{'='*50}")
        preprocess_dataset(
            protocol_file=args.protocol_file,
            output_dir=args.output_dir,
            subset_name=subset,
            sr=args.sr,
            target_len=args.target_len
        )


if __name__ == "__main__":
    main()