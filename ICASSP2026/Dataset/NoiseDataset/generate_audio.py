#!/usr/bin/env python3
import os, re, argparse, shutil, glob, json
from pathlib import Path
import yaml, pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 너의 augmentation 모듈들 ---
from augmentation import (
    BackgroundNoiseAugmentorDeepen as BackgroundNoiseAugmentor,
    BackgroundMusicAugmentorDeepen as BackgroundMusicAugmentor,
    GaussianAugmentorV1,
    HighPassFilterAugmentor,
    LowPassFilterAugmentor,
    PitchAugmentor,
    TimeStretchAugmentor,
    EchoAugmentorDeepen as EchoAugmentor,
    ReverbAugmentor,
)

AUGMENTATION_CLASSES = {
    "background_noise": BackgroundNoiseAugmentor,
    "background_music": BackgroundMusicAugmentor,
    "gaussian_noise": GaussianAugmentorV1,
    "high_pass_filter": LowPassFilterAugmentor.__class__,  # placeholder (아래에서 덮어씀)
    "low_pass_filter": LowPassFilterAugmentor,
    "pitch_shift": PitchAugmentor,
    "time_stretch": TimeStretchAugmentor,
    "echo": EchoAugmentor,
    "reverberation": ReverbAugmentor,
}
AUGMENTATION_CLASSES["high_pass_filter"] = HighPassFilterAugmentor  # 명시 덮어쓰기
AUG_LIST_ALL = list(AUGMENTATION_CLASSES.keys())

SPLIT_PAT = re.compile(r"ASVspoof2019_LA_(train|dev|eval)", re.IGNORECASE)

def infer_subset_from_path(p: str) -> str:
    m = SPLIT_PAT.search(p)
    if m: return m.group(1).lower()
    lp = p.lower()
    for s in ("train", "dev", "eval"):
        if f"/{s}/" in lp: return s
    return "train"

def parse_protocol_line(line: str):
    s = line.strip()
    if not s or s.startswith("#"): return None
    parts = s.split()
    if len(parts) < 3: return None
    spk, fullpath, lab = parts[0], parts[1], parts[2]
    return spk, fullpath, lab

def ensure_dirs(root: Path):
    for subset in ("train","dev","eval"):
        (root/subset/"clean").mkdir(parents=True, exist_ok=True)
        (root/subset/"augmented").mkdir(parents=True, exist_ok=True)

def collect_protocol_files(protocols, protocol_dir, pattern):
    files = []
    if protocols:
        for p in protocols:
            files.append(Path(p))
    if protocol_dir:
        pat = pattern or "*.txt"
        files.extend([Path(x) for x in glob.glob(str(Path(protocol_dir)/pat))])
    if not files:
        raise ValueError("No protocol files found. Use --protocol or --protocol-dir.")
    return files

def process_single_line(args_tuple):
    """Process a single protocol line - designed for multiprocessing"""
    line, aug_list, AUG_CONFIG, AUGMENTATION_CLASSES, sr, out_root, copy_clean, force_subset = args_tuple
    
    parsed = parse_protocol_line(line)
    if not parsed:
        return []
    
    spk, clean_flac, lab = parsed
    subset = (force_subset or infer_subset_from_path(clean_flac))
    
    clean_stem = Path(clean_flac).stem
    out_clean = out_root / subset / "clean" / f"{clean_stem}.wav"
    
    records = []
    
    # clean meta
    records.append({
        "subset": subset,
        "speaker": spk,
        "src_path": clean_flac,
        "file_path": str(out_clean),
        "label2": lab,
        "label1": "clean",
        "mode": "-",
        "snr_db": "-",
        "params_json": "-",
        "samplerate": sr,
        "format": "wav",
    })
    
    if copy_clean:
        try:
            shutil.copy2(clean_flac, out_clean)
        except Exception:
            pass
    
    # augment
    for aug_name in aug_list:
        aug_class = AUGMENTATION_CLASSES[aug_name]
        cfg = dict(AUG_CONFIG.get(aug_name, {}))
        cfg.setdefault("target_sr", sr)
        out_aug = out_root / subset / "augmented" / f"{clean_stem}__{aug_name}.wav"
        out_aug.parent.mkdir(parents=True, exist_ok=True)
        cfg["output_path"] = str(out_aug)
        cfg["out_format"] = "wav"
        
        try:
            augmentor = aug_class(cfg)
            augmentor.load(clean_flac)
            augmentor.transform()
            augmentor.augmented_audio.export(str(out_aug), format="wav")
            
            params = getattr(augmentor, "params", None)
            records.append({
                "subset": subset,
                "speaker": spk,
                "src_path": clean_flac,
                "file_path": str(out_aug),
                "label2": lab,
                "label1": aug_name,
                "mode": getattr(augmentor, "mode", "-"),
                "snr_db": getattr(augmentor, "snr_db", "-"),
                "params_json": "-" if params is None else json.dumps(params, ensure_ascii=False),
                "samplerate": sr,
                "format": "wav",
            })
        except Exception as e:
            print(f"Error processing {clean_flac} with {aug_name}: {e}")
    
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", nargs="*", help="one or more protocol.txt files")
    ap.add_argument("--protocol-dir", help="directory containing protocol files")
    ap.add_argument("--protocol-pattern", help="glob pattern inside --protocol-dir (default: *.txt)")
    ap.add_argument("--out-root", required=True, help="output root dir")
    ap.add_argument("--aug-config", required=True, help="YAML config path")
    ap.add_argument("--sr", type=int, default=16000, help="export sample rate (default 16000)")
    ap.add_argument("--copy-clean", action="store_true", help="also copy clean into out-root/subset/clean")
    ap.add_argument("--exclude", nargs="*", default=[], help="augmentation names to exclude")
    ap.add_argument("--force-subset",dest="force_subset",choices=["train","dev","eval"],help="force subset for all lines from these protocol(s)")
    ap.add_argument("--meta-out", default="meta_noise.csv", help="metadata CSV filename (created under out-root)")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    with open(args.aug_config, "r") as f:
        AUG_CONFIG = yaml.safe_load(f)

    aug_list = [a for a in AUG_LIST_ALL if a not in set(args.exclude)]

    prot_files = collect_protocol_files(args.protocol, args.protocol_dir, args.protocol_pattern)

    meta_path = out_root / args.meta_out
    meta_exists = meta_path.exists()
    all_records = []
    
    # Collect all lines from all protocol files
    all_lines = []
    for pfile in prot_files:
        lines = [ln for ln in open(pfile, "r", encoding="utf-8") if ln.strip() and not ln.startswith("#")]
        all_lines.extend(lines)
    
    print(f"Total lines to process: {len(all_lines)}")
    
    # Set up multiprocessing
    num_workers = args.num_workers or cpu_count()
    print(f"Using {num_workers} workers for parallel processing")
    
    # Prepare arguments for each worker
    worker_args = [
        (line, aug_list, AUG_CONFIG, AUGMENTATION_CLASSES, args.sr, out_root, args.copy_clean, args.force_subset)
        for line in all_lines
    ]
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_line, worker_args),
            total=len(all_lines),
            desc="Processing audio files"
        ))
    
    # Flatten results
    for record_list in results:
        all_records.extend(record_list)

    # meta append
    df_new = pd.DataFrame(all_records)
    if meta_exists:
        df_old = pd.read_csv(meta_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(meta_path, index=False)
    else:
        df_new.to_csv(meta_path, index=False)

    print(f"[DONE] saved metadata: {meta_path}  (+{len(df_new)} rows)")
    print(f"out-root: {out_root}")

if __name__ == "__main__":
    main()
