#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
DEFAULT_CHUNK_SECONDS = 5


def run_ffmpeg(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError("FFmpeg command failed")


def convert_to_16k_mono(input_path: Path, output_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def trim_edges(y: np.ndarray, top_db: int = 30):
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt


def main():
    parser = argparse.ArgumentParser(
        description="RVC audio preprocessing (16k mono, RVC-ready structure)"
    )

    parser.add_argument("--input", required=True, help="Input audio directory")
    parser.add_argument("--exp_name", required=True, help="RVC experiment name")
    parser.add_argument(
        "--chunk_seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Length of each chunk in seconds (default: 5)",
    )

    args = parser.parse_args()
    chunk_seconds = args.chunk_seconds

    # Resolve project structure
    project_root = Path(__file__).resolve().parent
    rvc_root = project_root / "rvc"
    exp_dir = rvc_root / "logs" / args.exp_name
    wav_output_dir = exp_dir / "1_16k_wavs"

    wav_output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input)

    audio_files = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"Found {len(audio_files)} audio files")
    print(f"Output directory: {wav_output_dir}")

    chunk_index = 0

    for audio in tqdm(audio_files, desc="Preprocessing"):
        try:
            tmp_wav = wav_output_dir / "__tmp.wav"
            convert_to_16k_mono(audio, tmp_wav)

            y, sr = librosa.load(tmp_wav, sr=16000, mono=True)
            y = trim_edges(y)

            # Split into fixed-length chunks
            samples_per_chunk = chunk_seconds * sr
            total_samples = len(y)

            for start in range(0, total_samples, samples_per_chunk):
                end = start + samples_per_chunk
                chunk = y[start:end]

                # Skip very short tail segments
                if len(chunk) < sr:  # shorter than 1 second
                    continue

                out_path = wav_output_dir / f"{chunk_index:06d}.wav"
                sf.write(out_path, chunk, sr)
                chunk_index += 1

            tmp_wav.unlink(missing_ok=True)

        except Exception as e:
            print(f"[WARN] Failed {audio.name}: {e}")

    print("Preprocessing complete. Files are ready inside rvc/logs/<exp_name>/1_16k_wavs/")


if __name__ == "__main__":
    main()