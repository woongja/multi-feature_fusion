#!/usr/bin/env python3
import argparse
from pathlib import Path

def parse_line(line: str):
    """
    허용 포맷:
      1) 5토큰: <spk> <utt> - - <label>
      2) 3토큰: <spk> <utt> <label>   (유연 처리)
    반환: (speaker, utt, label) 또는 None
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) >= 5:
        spk, utt, lab = parts[0], parts[1], parts[4]
        return spk, utt, lab
    elif len(parts) == 3:
        spk, utt, lab = parts
        return spk, utt, lab
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="원본 프로토콜 텍스트 경로")
    ap.add_argument("--base",  required=True, help="LA .flac 파일들이 존재하는 디렉토리 (예: .../ASVspoof2019_LA_train/flac)")
    ap.add_argument("--output", required=True, help="출력 파일 경로")
    ap.add_argument("--check", action="store_true", help="실제 파일 존재 여부 확인(없어도 진행은 가능)")
    args = ap.parse_args()

    in_path = Path(args.input)
    base = Path(args.base)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    n_missing = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for raw in f_in:
            n_total += 1
            parsed = parse_line(raw)
            if not parsed:
                continue
            spk, utt, lab = parsed

            flac_path = base / f"{utt}.flac"
            if args.check and not flac_path.is_file():
                n_missing += 1

            # 출력 라인: "<speaker> <full_flac_path> <label>"
            f_out.write(f"{spk} {str(flac_path)} {lab}\n")
            n_written += 1

    print(f"[DONE] total_lines={n_total}, written={n_written}, missing_files={n_missing}")
    print(f"[OUT ] {out_path}")

if __name__ == "__main__":
    main()
