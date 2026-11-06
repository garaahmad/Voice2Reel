"""Validate recordings and export deterministic, disjoint XTTS metadata."""

import argparse
import csv
import hashlib
import math
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voice2reel.common import read_jsonl, write_json
from voice2reel.scriptwriter.data import normalized


def load_audio(path, min_seconds=2.0, max_seconds=11.0):
    info = sf.info(path)
    if not min_seconds <= info.duration <= max_seconds:
        raise ValueError(f"{path}: duration {info.duration:.2f}s outside {min_seconds}–{max_seconds}s")
    if info.samplerate < 16000 or info.channels > 2:
        raise ValueError(f"{path}: use mono/stereo audio at >= 16000 Hz")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: non-finite audio")
    if np.mean(np.abs(audio) >= 0.999) > 0.001:
        raise ValueError(f"{path}: clipped recording; record again at lower gain")
    audio = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 0.003:
        raise ValueError(f"{path}: silent or excessively quiet recording")
    divisor = math.gcd(sr, 22050)
    audio = resample_poly(audio, 22050 // divisor, sr // divisor).astype(np.float32)
    # No silence trimming, denoising, or gain boost that could corrupt transcription alignment.
    return audio, {"duration": len(audio) / 22050, "rms": rms}


def prepare(manifest, audio_root, output, seed=42):
    root, output = Path(audio_root).resolve(), Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a fresh directory")
    rows = read_jsonl(manifest)
    if len(rows) < 12:
        raise ValueError(
            "At least 12 distinct clips required for train/validation/test; record many more for quality"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="voice-prepare-", dir=output.parent))
    ids, texts, hashes, speakers = set(), set(), set(), set()
    prepared = []
    try:
        (staging / "wavs").mkdir()
        for index, row in enumerate(rows):
            if set(row) != {"audio", "text", "speaker"}:
                raise ValueError("Each voice record needs exactly audio, text, speaker")
            if not all(isinstance(row[k], str) and row[k].strip() for k in row):
                raise ValueError("Voice fields must be nonempty strings")
            if any(c in row["text"] + row["speaker"] for c in "|\n\r"):
                raise ValueError("Pipes and newlines are not supported in metadata fields")
            if len(row["text"]) > 250:
                raise ValueError("Transcript too long; split recordings at sentence boundaries")
            path = (root / row["audio"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing recording or path outside audio root: {row['audio']}")
            text_key = normalized(row["text"])
            if path in ids or text_key in texts:
                raise ValueError("Duplicate recording path or transcript")
            ids.add(path)
            texts.add(text_key)
            speakers.add(row["speaker"])
            if len(speakers) != 1:
                raise ValueError(
                    "This fine-tuning recipe expects one speaker; prepare separate runs per voice"
                )
            audio, stats = load_audio(path)
            # Hash canonical PCM; catches copies with different filenames/containers/sample rates.
            pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
            digest = hashlib.sha256(pcm.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError("Duplicate audio content; split leakage would invalidate evaluation")
            hashes.add(digest)
            name = f"wavs/clip_{index:05d}.wav"
            sf.write(staging / name, audio, 22050, subtype="PCM_16")
            prepared.append(
                {
                    "audio_file": name,
                    "text": row["text"].strip(),
                    "speaker_name": row["speaker"],
                    "sha256": digest,
                    **stats,
                }
            )
        random.Random(seed).shuffle(prepared)
        count = max(2, round(len(prepared) * 0.1))
        splits = {
            "validation": prepared[:count],
            "test": prepared[count : 2 * count],
            "train": prepared[2 * count :],
        }
        for split, samples in splits.items():
            with (staging / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|"
                )
                writer.writeheader()
                for sample in samples:
                    writer.writerow({key: sample[key] for key in writer.fieldnames})
        report = {
            "seed": seed,
            "sample_rate": 22050,
            "speaker": next(iter(speakers)),
            "total_minutes": sum(x["duration"] for x in prepared) / 60,
            "counts": {key: len(value) for key, value in splits.items()},
            "splits": splits,
        }
        write_json(staging / "report.json", report)
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--output", default="data/voice")
    args = p.parse_args()
    report = prepare(args.manifest, args.audio_root, args.output)
    print(f"Prepared {report['counts']}; {report['total_minutes']:.1f} minutes. Report in {args.output}")


if __name__ == "__main__":
    main()
