#!/usr/bin/env python3
"""
Train an RVC voice model (Quran-safe, kid/adult tuned)

This script orchestrates:
1) Feature extraction
2) FAISS index building
3) RVC model training

It assumes you are using the official RVC repository as a submodule
or installed alongside this project.

Author: Quralingo
"""

import argparse
import subprocess
from pathlib import Path
import sys


def run(cmd, cwd=None):
    print("▶", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Train RVC model (kid vs adult tuned)"
    )

    parser.add_argument("--data", required=True, help="Processed WAV dataset directory")
    parser.add_argument("--name", required=True, help="Model name (e.g. female_kid)")
    parser.add_argument(
        "--voice_type",
        choices=["kid", "adult"],
        required=True,
        help="Voice type (affects pitch & training params)",
    )
    parser.add_argument(
        "--rvc_root",
        default="rvc",
        help="Path to RVC repo root",
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    rvc_root = Path(args.rvc_root).resolve()

    # Use isolated RVC virtual environment if available
    rvc_python = rvc_root / ".venv" / "bin" / "python"
    if not rvc_python.exists():
        print("❌ RVC virtual environment not found at rvc/.venv")
        print("Please create it with: python3.10 -m venv rvc/.venv")
        sys.exit(1)

    exp_dir = rvc_root / "logs" / args.name

    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Ensure dataset is copied into experiment folder
    # -------------------------
    wav_src = data_dir
    wav_dst = exp_dir / "1_16k_wavs"
    wav_dst.mkdir(parents=True, exist_ok=True)

    # Copy wav files if not already present
    for wav in wav_src.glob("*.wav"):
        target = wav_dst / wav.name
        if not target.exists():
            target.write_bytes(wav.read_bytes())

    # -------------------------
    # Voice-type specific tuning
    # -------------------------
    if args.voice_type == "kid":
        f0_min = 120
        f0_max = 600
        lr = 2e-4
    else:  # adult
        f0_min = 50
        f0_max = 350
        lr = 1e-4

    # -------------------------
    # Step 1: Feature extraction
    # -------------------------
    run(
        [
            str(rvc_python),
            "infer/modules/train/extract_feature_print.py",
            "cpu",
            "1",
            "0",
            str(exp_dir),
            "v1",
            "False",
        ],
        cwd=rvc_root,
    )

    # -------------------------
    # Step 2: Pitch extraction
    # -------------------------
    run(
        [
            str(rvc_python),
            "infer/modules/train/extract/extract_f0_print.py",
            str(exp_dir),
            "1",
            "rmvpe",
        ],
        cwd=rvc_root,
    )

    # -------------------------
    # Generate filelist.txt required by RVC
    # -------------------------
    feature_dir = exp_dir / "3_feature256"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"
    wav_dir = exp_dir / "1_16k_wavs"

    filelist_path = exp_dir / "filelist.txt"

    lines = []
    for wav_file in sorted(wav_dir.glob("*.wav")):
        name = wav_file.stem
        feat = feature_dir / f"{name}.npy"
        f0 = f0_dir / f"{name}.wav.npy"
        f0nsf = f0nsf_dir / f"{name}.wav.npy"

        if feat.exists() and f0.exists() and f0nsf.exists():
            lines.append(
                f"{wav_file}|{feat}|{f0}|{f0nsf}|0\n"
            )

    with open(filelist_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Generated filelist.txt with {len(lines)} entries")

    # -------------------------
    # Step 3: Train model
    # -------------------------
    run(
        [
            str(rvc_python),
            "infer/modules/train/train.py",
            "-e", args.name,
            "-sr", "40000",
            "-f0", "1",
            "-bs", str(args.batch_size),
            "-g", str(args.gpu if args.gpu >= 0 else 0),
            "-te", str(args.epochs),
            "-se", "10",
            "-v", "v1",
            "-l", "0",
            "-c", "0",
        ],
        cwd=rvc_root,
    )

    # -------------------------
    # Step 4: Build FAISS index
    # -------------------------
    index_script = rvc_root / "infer/modules/train" / "train_index.py"
    if index_script.exists():
        run(
            [
                str(rvc_python),
                str(index_script.relative_to(rvc_root)),
                "-e",
                args.name,
            ],
            cwd=rvc_root,
        )
    else:
        print("⚠ train_index.py not found, skipping index build")

    print("\n✅ Training complete")
    print(f"Model artifacts in: {exp_dir}")
    print("You can now run inference using this model.")


if __name__ == "__main__":
    main()