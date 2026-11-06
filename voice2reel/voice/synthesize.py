import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from voice2reel.common import write_json
from voice2reel.scriptwriter.schema import Script
from voice2reel.voice.model import download_model, load_model
from voice2reel.voice.prepare import load_audio


def synthesize_segments(model, texts, reference, seed=42):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    conditioning, speaker = model.get_conditioning_latents(audio_path=[str(reference)])
    audio, timing, cursor = [], [], 0
    with torch.inference_mode():
        for text in texts:
            result = model.inference(
                text,
                "ar",
                conditioning,
                speaker,
                temperature=0.65,
                repetition_penalty=5.0,
                enable_text_splitting=True,
            )
            wav = np.asarray(result["wav"], dtype=np.float32)
            if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
                raise ValueError("Model returned invalid audio")
            timing.append({"text": text, "start": cursor / 24000, "end": (cursor + len(wav)) / 24000})
            audio.append(wav)
            cursor += len(wav)
    combined = np.concatenate(audio)
    peak = float(np.max(np.abs(combined)))
    if peak > 0.99:
        combined *= 0.99 / peak
    return combined, timing


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--script", help="Validated script JSON; synthesizes each beat separately")
    source.add_argument("--text", help="UTF-8 narration text file")
    p.add_argument("--reference", required=True)
    p.add_argument("--base", default="models/xtts-v2")
    p.add_argument("--checkpoint", help="Fine-tuned best_model.pth; omit for baseline")
    p.add_argument("--output", default="outputs/narration.wav")
    args = p.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime")
    load_audio(args.reference, min_seconds=3, max_seconds=30)
    if args.script:
        script = Script.model_validate_json(Path(args.script).read_text(encoding="utf-8"))
        texts = [b.narration for b in script.beats]
    else:
        texts = [Path(args.text).read_text(encoding="utf-8").strip()]
        if not texts[0] or len(texts[0]) > 5000:
            raise ValueError("Text file must contain 1–5000 characters")
    base = download_model(args.base)
    model = load_model(base, args.checkpoint)
    audio, timing = synthesize_segments(model, texts, args.reference)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, 24000, subtype="PCM_16")
    write_json(
        output.with_suffix(".json"),
        {
            "sample_rate": 24000,
            "duration": len(audio) / 24000,
            "segments": timing,
            "timing_method": "separately_synthesized_segments",
            "checkpoint": args.checkpoint or "base",
            "seed": 42,
        },
    )
    print(f"Wrote {output} and segment timings (not word-level alignment)")


if __name__ == "__main__":
    main()
