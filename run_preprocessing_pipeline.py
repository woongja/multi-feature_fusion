#!/usr/bin/env python3
"""
Complete preprocessing and training pipeline for noise classification

This script demonstrates how to:
1. Preprocess audio features and save as .pt files
2. Train using preprocessed data for faster loading
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"✓ {description} completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Complete preprocessing and training pipeline")
    parser.add_argument("--protocol_file", type=str, required=True,
                       help="Path to protocol file")
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                       help="Directory to save preprocessed features")
    parser.add_argument("--skip_preprocessing", action='store_true',
                       help="Skip preprocessing step (use existing preprocessed data)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="model_checkpoint.pth")
    parser.add_argument("--log_dir", type=str, default="runs_preprocessed")
    parser.add_argument("--num_workers", type=int, default=8)
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    protocol_file = Path(args.protocol_file).absolute()
    preprocessed_dir = Path(args.preprocessed_dir).absolute()
    
    if not protocol_file.exists():
        print(f"Error: Protocol file not found: {protocol_file}")
        sys.exit(1)
    
    print("Preprocessing and Training Pipeline")
    print("=" * 60)
    print(f"Protocol file: {protocol_file}")
    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Skip preprocessing: {args.skip_preprocessing}")
    
    # Step 1: Preprocessing (if not skipped)
    if not args.skip_preprocessing:
        preprocess_cmd = [
            sys.executable, 
            str(script_dir / "preprocess_features.py"),
            "--protocol_file", str(protocol_file),
            "--output_dir", str(preprocessed_dir),
            "--subset", "all"
        ]
        run_command(preprocess_cmd, "Feature preprocessing")
    else:
        print("\n⏭️  Skipping preprocessing step")
    
    # Step 2: Verify preprocessed data
    verify_cmd = [
        sys.executable,
        str(script_dir / "datautils" / "data_preprocessed.py"),
        "--preprocessed_dir", str(preprocessed_dir)
    ]
    run_command(verify_cmd, "Data verification")
    
    # Step 3: Training
    train_cmd = [
        sys.executable,
        str(script_dir / "train_interspeech_preprocessed.py"),
        "--preprocessed_dir", str(preprocessed_dir),
        "--is_train",
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--save_path", args.save_path,
        "--log_dir", args.log_dir,
        "--num_workers", str(args.num_workers)
    ]
    run_command(train_cmd, "Model training")
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Preprocessed data: {preprocessed_dir}")
    print(f"💾 Model saved to: {args.save_path}")
    print(f"📊 Tensorboard logs: {args.log_dir}")
    
    print(f"\n📋 To run evaluation:")
    print(f"python train_interspeech_preprocessed.py \\")
    print(f"    --preprocessed_dir {preprocessed_dir} \\")
    print(f"    --is_eval \\")
    print(f"    --model_path {args.save_path} \\")
    print(f"    --save_results eval_results.txt")

if __name__ == "__main__":
    main()