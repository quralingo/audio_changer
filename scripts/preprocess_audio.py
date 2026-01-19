#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


def run_ffmpeg(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError("FFmpeg command failed")


def convert_to_wav(input_path: Path, output_path: Path, sr: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-vn",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def normalize_audio(y: np.ndarray, target_db: float = -20.0):
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-6:
        return y
    gain = 10 ** (target_db / 20) / rms
    return np.clip(y * gain, -1.0, 1.0)


def trim_edges(y: np.ndarray, sr: int, top_db: int = 30):
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt


def split_on_long_silence(
    y: np.ndarray,
    sr: int,
    silence_db: int,
    min_silence_sec: float,
    min_chunk_sec: float,
    max_chunk_sec: float,
):
    intervals = librosa.effects.split(y, top_db=silence_db)

    chunks = []
    current = []

    for start, end in intervals:
        segment = y[start:end]
        current.append(segment)

        duration = sum(len(c) for c in current) / sr
        if duration >= max_chunk_sec:
            chunks.append(np.concatenate(current))
            current = []

    if current:
        combined = np.concatenate(current)
        duration = len(combined) / sr
        if duration >= min_chunk_sec:
            chunks.append(combined)

    return chunks


def process_file(
    input_path: Path,
    output_dir: Path,
    sr: int,
    silence_db: int,
    min_silence_sec: float,
    min_chunk_sec: float,
    max_chunk_sec: float,
):
    tmp_wav = output_dir / "__tmp.wav"
    convert_to_wav(input_path, tmp_wav, sr)

    y, _ = librosa.load(tmp_wav, sr=sr, mono=True)
    y = normalize_audio(y)
    y = trim_edges(y, sr)

    chunks = split_on_long_silence(
        y,
        sr,
        silence_db,
        min_silence_sec,
        min_chunk_sec,
        max_chunk_sec,
    )

    stem = input_path.stem
    for i, chunk in enumerate(chunks):
        out_path = output_dir / f"{stem}_{i:03d}.wav"
        sf.write(out_path, chunk, sr)

    tmp_wav.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Quran-safe audio preprocessing for RVC training"
    )

    parser.add_argument("--input", required=True, help="Input audio directory")
    parser.add_argument("--output", required=True, help="Output WAV directory")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate")
    parser.add_argument("--silence_db", type=int, default=32, help="Silence threshold dB")
    parser.add_argument(
        "--min_silence_sec",
        type=float,
        default=1.2,
        help="Minimum silence duration to consider splitting",
    )
    parser.add_argument(
        "--min_chunk_sec",
        type=float,
        default=10.0,
        help="Minimum chunk length in seconds",
    )
    parser.add_argument(
        "--max_chunk_sec",
        type=float,
        default=120.0,
        help="Maximum chunk length in seconds",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"Found {len(audio_files)} audio files")

    for audio in tqdm(audio_files, desc="Preprocessing"):
        try:
            process_file(
                audio,
                output_dir,
                args.sr,
                args.silence_db,
                args.min_silence_sec,
                args.min_chunk_sec,
                args.max_chunk_sec,
            )
        except Exception as e:
            print(f"[WARN] Failed {audio.name}: {e}")

    print("Preprocessing complete. Audio is ready for RVC training.")


if __name__ == "__main__":
    main()