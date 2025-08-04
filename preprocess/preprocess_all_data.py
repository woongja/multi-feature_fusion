#!/usr/bin/env python3
"""
전체 데이터셋 전처리 스크립트
모든 오디오 파일을 pt 파일로 변환하여 다른 서버로 이동 가능하게 만듭니다.
"""

import argparse
import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import hashlib
import multiprocessing as mp
from functools import partial
import shutil
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


def process_single_file(args_tuple):
    """단일 파일 처리 함수 (멀티프로세싱용)"""
    utt_id, output_dir, sr, target_len, subset_name, label = args_tuple
    
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
        feature_path = output_dir / "features" / feature_filename
        
        # Save features
        features = {
            'spec_db': spec_db,
            'mfcc': mfcc,
            'f0': f0,
            'original_path': utt_id,
            'subset': subset_name
        }
        torch.save(features, feature_path)
        
        # Return index entry
        index_entry = {
            'feature_file': str(feature_path),
            'original_path': utt_id,
            'subset': subset_name,
            'feature_filename': feature_filename,
            'shapes': {
                'spec_db': list(spec_db.shape),
                'mfcc': list(mfcc.shape),
                'f0': list(f0.shape)
            }
        }
        
        if label is not None:
            index_entry['label'] = label
            
        return index_entry, None
        
    except Exception as e:
        return None, f"Failed to process {utt_id}: {e}"


def preprocess_all_datasets(protocol_file, output_dir, sr=16000, target_len=64000, num_workers=8):
    """모든 데이터셋을 전처리"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Features directory: {features_dir}")
    
    # 모든 subset 처리
    all_files = []
    all_index_data = []
    failed_files = []
    
    subsets = ['train', 'dev', 'eval']
    
    for subset_name in subsets:
        print(f"\n{'='*60}")
        print(f"Processing {subset_name.upper()} set")
        print(f"{'='*60}")
        
        # Generate file list based on subset
        if subset_name == "train":
            labels, file_list = gen_list(protocol_file, is_train=True)
        elif subset_name == "dev":
            labels, file_list = gen_list(protocol_file, is_dev=True)
        elif subset_name == "eval":
            file_list = gen_list(protocol_file, is_eval=True)
            labels = {}
        
        print(f"Found {len(file_list)} files for {subset_name}")
        
        # Prepare arguments for multiprocessing
        args_list = []
        for utt_id in file_list:
            label = labels.get(utt_id) if labels else None
            args_list.append((utt_id, output_dir, sr, target_len, subset_name, label))
        
        # Process files in parallel
        print(f"Processing with {num_workers} workers...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc=f"Processing {subset_name}"
            ))
        
        # Collect results
        subset_index_data = []
        for index_entry, error in results:
            if index_entry:
                subset_index_data.append(index_entry)
                all_index_data.append(index_entry)
            if error:
                failed_files.append(error)
                
        # Save subset-specific index
        subset_index_file = output_dir / f"{subset_name}_index.json"
        with open(subset_index_file, 'w') as f:
            json.dump(subset_index_data, f, indent=2)
            
        print(f"✓ {subset_name}: {len(subset_index_data)} files processed")
        print(f"  Index saved to: {subset_index_file}")
    
    # Save combined index
    combined_index_file = output_dir / "all_data_index.json"
    with open(combined_index_file, 'w') as f:
        json.dump(all_index_data, f, indent=2)
    
    # Save preprocessing info
    info = {
        "total_files": len(all_index_data),
        "failed_files": len(failed_files),
        "preprocessing_config": {
            "sr": sr,
            "target_len": target_len,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 128,
            "n_mfcc": 13
        },
        "subsets": {subset: len([x for x in all_index_data if x['subset'] == subset]) 
                   for subset in subsets}
    }
    
    info_file = output_dir / "preprocessing_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save failed files log
    if failed_files:
        failed_file = output_dir / "failed_files.txt"
        with open(failed_file, 'w') as f:
            for failed in failed_files:
                f.write(f"{failed}\n")
        print(f"\n⚠️  Failed to process {len(failed_files)} files. See {failed_file}")
    
    print(f"\n🎉 전처리 완료!")
    print(f"📁 Total processed files: {len(all_index_data)}")
    print(f"💾 Features directory: {features_dir}")
    print(f"📋 Combined index: {combined_index_file}")
    print(f"📊 Info file: {info_file}")
    
    # 데이터 크기 정보
    total_size = sum(f.stat().st_size for f in features_dir.glob("*.pt"))
    print(f"💿 Total data size: {total_size / (1024**3):.2f} GB")
    
    return output_dir


def create_transfer_package(preprocessed_dir, target_dir=None):
    """다른 서버로 이동하기 위한 패키지 생성"""
    if target_dir is None:
        target_dir = Path(preprocessed_dir).parent / f"{Path(preprocessed_dir).name}_transfer"
    
    target_dir = Path(target_dir)
    
    print(f"\n📦 Creating transfer package...")
    print(f"Source: {preprocessed_dir}")
    print(f"Target: {target_dir}")
    
    # Copy entire directory
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(preprocessed_dir, target_dir)
    
    # Create README for transfer
    readme_content = f"""# Preprocessed Data Transfer Package

## 사용법
1. 이 디렉토리를 대상 서버로 복사
2. train_interspeech_preprocessed.py 사용하여 훈련

## 파일 구조
- features/: 전처리된 .pt 파일들
- *_index.json: 각 subset별 인덱스 파일
- all_data_index.json: 전체 데이터 인덱스
- preprocessing_info.json: 전처리 설정 정보

## 훈련 명령어 예시
```bash
python train_interspeech_preprocessed.py \\
    --preprocessed_dir {target_dir.name} \\
    --is_train \\
    --batch_size 32 \\
    --num_epochs 100
```
"""
    
    with open(target_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Transfer package created: {target_dir}")
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Preprocess all audio data for training")
    parser.add_argument("--protocol_file", type=str, required=True,
                       help="Path to protocol file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for preprocessed features")
    parser.add_argument("--sr", type=int, default=16000,
                       help="Sample rate")
    parser.add_argument("--target_len", type=int, default=64000,
                       help="Target length for audio segments")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--create_transfer_package", action='store_true',
                       help="Create a transfer package for moving to another server")
    parser.add_argument("--transfer_dir", type=str, default=None,
                       help="Directory for transfer package")
    
    args = parser.parse_args()
    
    # Preprocess all data
    preprocessed_dir = preprocess_all_datasets(
        protocol_file=args.protocol_file,
        output_dir=args.output_dir,
        sr=args.sr,
        target_len=args.target_len,
        num_workers=args.num_workers
    )
    
    # Create transfer package if requested
    if args.create_transfer_package:
        transfer_dir = create_transfer_package(preprocessed_dir, args.transfer_dir)
        print(f"\n🚀 Ready to transfer: {transfer_dir}")


if __name__ == "__main__":
    main()