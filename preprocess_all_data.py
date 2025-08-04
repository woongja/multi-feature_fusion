#!/usr/bin/env python3
"""
전체 데이터셋 전처리 스크립트 (두 개의 protocol 파일 사용)
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess all audio data")
    parser.add_argument("--protocol_train_dev", type=str, 
                       default="/home/woongjae/noise-tracing/multi-feature_fusion/Datasets/protocol_train_dev.txt",
                       help="Path to train/dev protocol file")
    parser.add_argument("--protocol_eval", type=str,
                       default="/home/woongjae/noise-tracing/multi-feature_fusion/Datasets/protocol_eval.txt", 
                       help="Path to eval protocol file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for preprocessed features")
    parser.add_argument("--sr", type=int, default=16000,
                       help="Sample rate")
    parser.add_argument("--target_len", type=int, default=64000,
                       help="Target length for audio segments")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Protocol train/dev: {args.protocol_train_dev}")
    print(f"Protocol eval: {args.protocol_eval}")
    
    all_index_data = []
    failed_files = []
    
    # Train/Dev 처리
    for subset_name in ['Train', 'Dev']:
        print(f"\n{'='*60}")
        print(f"Processing {subset_name.upper()} set")
        print(f"{'='*60}")
        
        if subset_name == "Train":
            labels, file_list = gen_list(args.protocol_train_dev, is_train=True)
        elif subset_name == "Dev":
            labels, file_list = gen_list(args.protocol_train_dev, is_dev=True)
        
        print(f"Found {len(file_list)} files for {subset_name}")
        
        # 멀티프로세싱용 arguments 준비
        args_list = []
        for utt_id in file_list:
            label = labels.get(utt_id)
            args_list.append((utt_id, output_dir, args.sr, args.target_len, subset_name, label))
        
        # 병렬 처리
        print(f"Processing with {args.num_workers} workers...")
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc=f"Processing {subset_name}"
            ))
        
        # 결과 수집
        subset_index_data = []
        for index_entry, error in results:
            if index_entry:
                subset_index_data.append(index_entry)
                all_index_data.append(index_entry)
            if error:
                failed_files.append(error)
                
        # subset별 인덱스 저장
        subset_index_file = output_dir / f"{subset_name}_index.json"
        with open(subset_index_file, 'w') as f:
            json.dump(subset_index_data, f, indent=2)
            
        print(f"✓ {subset_name}: {len(subset_index_data)} files processed")
    
    # Eval 처리
    print(f"\n{'='*60}")
    print(f"Processing EVAL set")
    print(f"{'='*60}")
    
    file_list = gen_list(args.protocol_eval, is_eval=True)
    print(f"Found {len(file_list)} files for eval")
    
    # 멀티프로세싱용 arguments 준비 (eval은 label이 없음)
    args_list = []
    for utt_id in file_list:
        args_list.append((utt_id, output_dir, args.sr, args.target_len, 'eval', None))
    
    # 병렬 처리
    print(f"Processing with {args.num_workers} workers...")
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc=f"Processing eval"
        ))
    
    # 결과 수집
    eval_index_data = []
    for index_entry, error in results:
        if index_entry:
            eval_index_data.append(index_entry)
            all_index_data.append(index_entry)
        if error:
            failed_files.append(error)
            
    # eval 인덱스 저장
    eval_index_file = output_dir / "eval_index.json"
    with open(eval_index_file, 'w') as f:
        json.dump(eval_index_data, f, indent=2)
        
    print(f"✓ eval: {len(eval_index_data)} files processed")
    
    # 전체 인덱스 저장
    combined_index_file = output_dir / "all_data_index.json"
    with open(combined_index_file, 'w') as f:
        json.dump(all_index_data, f, indent=2)
    
    # 실패한 파일 로그
    if failed_files:
        failed_file = output_dir / "failed_files.txt"
        with open(failed_file, 'w') as f:
            for failed in failed_files:
                f.write(f"{failed}\n")
        print(f"\n⚠️  Failed to process {len(failed_files)} files. See {failed_file}")
    
    # 최종 결과
    total_size = sum(f.stat().st_size for f in features_dir.glob("*.pt"))
    
    print(f"\n🎉 전처리 완료!")
    print(f"📁 Total processed files: {len(all_index_data)}")
    print(f"💾 Features directory: {features_dir}")
    print(f"💿 Total data size: {total_size / (1024**3):.2f} GB")
    print(f"📋 Combined index: {combined_index_file}")
    
    subset_counts = {}
    for entry in all_index_data:
        subset = entry['subset']
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    for subset, count in subset_counts.items():
        print(f"  - {subset}: {count} files")


if __name__ == "__main__":
    main()