#!/usr/bin/env python3
import os, re, argparse, shutil, glob, json
from pathlib import Path
import yaml, pandas as pd
from tqdm import tqdm

from augmentation import AutoTuneAugmentor  # 전용

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
        for p in protocols: files.append(Path(p))
    if protocol_dir:
        pat = pattern or "*.txt"
        files.extend([Path(x) for x in glob.glob(str(Path(protocol_dir)/pat))])
    if not files:
        raise ValueError("No protocol files found. Use --protocol or --protocol-dir.")
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", nargs="*", help="one or more protocol.txt files")
    ap.add_argument("--protocol-dir", help="directory containing protocol files")
    ap.add_argument("--protocol-pattern", help="glob pattern inside --protocol-dir (default: *.txt)")
    ap.add_argument("--out-root", required=True, help="output root dir")
    ap.add_argument("--aug-config", required=True, help="YAML config path (must include 'auto_tune')")
    ap.add_argument("--sr", type=int, default=16000, help="export sample rate (default 16000)")
    ap.add_argument("--copy-clean", action="store_true", help="also copy clean into out-root/subset/clean")
    ap.add_argument("--force-subset", choices=["train","dev","eval"], help="force subset for all lines")
    ap.add_argument("--meta-out", default="meta_noise_autotune.csv", help="metadata CSV filename (under out-root)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    with open(args.aug_config, "r") as f:
        cfg_all = yaml.safe_load(f)
    if "auto_tune" not in cfg_all:
        raise ValueError("YAML must contain 'auto_tune' section.")
    base_cfg = dict(cfg_all["auto_tune"])
    base_cfg.setdefault("target_sr", args.sr)

    meta_path = out_root / args.meta_out
    meta_exists = meta_path.exists()
    all_records = []

    prot_files = collect_protocol_files(args.protocol, args.protocol_dir, args.protocol_pattern)

    for pfile in prot_files:
        lines = [ln for ln in open(pfile, "r", encoding="utf-8") if ln.strip() and not ln.startswith("#")]
        for ln in tqdm(lines, desc=f"AutoTune {pfile.name}", total=len(lines)):
            parsed = parse_protocol_line(ln)
            if not parsed: continue
            spk, clean_flac, lab = parsed
            subset = args.force_subset or infer_subset_from_path(clean_flac)

            clean_stem = Path(clean_flac).stem
            out_clean = out_root / subset / "clean" / f"{clean_stem}.wav"
            out_aug  = out_root / subset / "augmented" / f"{clean_stem}__auto_tune.wav"
            out_aug.parent.mkdir(parents=True, exist_ok=True)

            # clean 메타 (복사 여부는 옵션)
            all_records.append({
                "subset": subset,
                "speaker": spk,
                "src_path": clean_flac,
                "file_path": str(out_clean),
                "label2": lab,
                "label1": "clean",
                "mode": "-",
                "snr_db": "-",
                "params_json": "-",
                "samplerate": args.sr,
                "format": "wav",
            })
            if args.copy_clean:
                try:
                    shutil.copy2(clean_flac, out_clean)  # flac->wav 변환은 별도 처리 필요 시 건너뜀
                except Exception:
                    pass

            # --- AutoTune 증강
            cfg = dict(base_cfg)
            cfg["output_path"] = str(out_aug)
            cfg["out_format"] = "wav"

            augmentor = AutoTuneAugmentor(cfg)
            augmentor.load(clean_flac)     # .flac 읽기 지원 가정
            augmentor.transform()
            augmentor.augmented_audio.export(str(out_aug), format="wav")

            params = getattr(augmentor, "params", None)
            all_records.append({
                "subset": subset,
                "speaker": spk,
                "src_path": clean_flac,
                "file_path": str(out_aug),
                "label2": lab,
                "label1": "auto_tune",
                "mode": getattr(augmentor, "mode", "-"),
                "snr_db": getattr(augmentor, "snr_db", "-"),
                "params_json": "-" if params is None else json.dumps(params, ensure_ascii=False),
                "samplerate": args.sr,
                "format": "wav",
            })

    df_new = pd.DataFrame(all_records)
    if meta_exists:
        df_old = pd.read_csv(meta_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(meta_path, index=False)
    else:
        df_new.to_csv(meta_path, index=False)

    print(f"[DONE] saved metadata: {meta_path} (+{len(df_new)} rows)")
    print(f"out-root: {out_root}")

if __name__ == "__main__":
    main()
