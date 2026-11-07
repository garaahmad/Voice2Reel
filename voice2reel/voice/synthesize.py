import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from voice2reel.common import write_json
from voice2reel.scriptwriter.schema import Script
from voice2reel.voice.model import DEFAULT_SPEAKER, available_speakers, download_model, load_model
from voice2reel.voice.prepare import load_audio


def synthesize_segments(model, texts, reference=None, seed=42, *, language="ar", speaker=None):
    import torch

    if language not in ("ar", "en"):
        raise ValueError("Supported languages: ar, en")
    if reference is not None and speaker is not None:
        raise ValueError("Choose a preset speaker or a reference recording, not both")
    if not texts or any(not isinstance(t, str) or not t.strip() or len(t) > 5000 for t in texts):
        raise ValueError("Provide nonempty text segments with at most 5000 characters each")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if reference is not None:
        conditioning, embedding = model.get_conditioning_latents(audio_path=[str(reference)])
    else:
        name = speaker or DEFAULT_SPEAKER
        manager = getattr(model, "speaker_manager", None)
        if manager is None or name not in manager.speakers:
            raise ValueError(f"Unknown preset voice {name!r}; use --list-speakers to see available names")
        preset = manager.speakers[name]
        conditioning, embedding = preset["gpt_cond_latent"], preset["speaker_embedding"]
    audio, timing, cursor = [], [], 0
    with torch.inference_mode():
        for text in texts:
            result = model.inference(
                text,
                language,
                conditioning,
                embedding,
                temperature=0.65,
                repetition_penalty=5.0,
                enable_text_splitting=True,
            )
            wav = np.asarray(result["wav"], dtype=np.float32)
            if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all() or not np.any(wav):
                raise ValueError("Model returned invalid audio")
            timing.append({"text": text, "start": cursor / 24000, "end": (cursor + len(wav)) / 24000})
            audio.append(wav)
            cursor += len(wav)
    combined = np.concatenate(audio)
    peak = float(np.max(np.abs(combined)))
    if peak > 0.99:
        combined *= 0.99 / peak
    return combined, timing


def read_source(*, script_path=None, text_path=None, content=None, language=None):
    """Resolve language before downloading weights; reject contradictory script metadata."""
    if sum(value is not None for value in (script_path, text_path, content)) != 1:
        raise ValueError("Provide exactly one script, text file, or text content")
    if script_path:
        script = Script.model_validate_json(Path(script_path).read_text(encoding="utf-8"))
        if language and language != script.language:
            raise ValueError("Requested language differs from script.language")
        return [b.narration for b in script.beats], script.language
    text = Path(text_path).read_text(encoding="utf-8") if text_path else content
    if not text or not text.strip() or len(text) > 5000:
        raise ValueError("Text must contain 1–5000 characters")
    language = language or "ar"
    if language not in ("ar", "en"):
        raise ValueError("Supported languages: ar, en")
    return [text.strip()], language


def main():
    p = argparse.ArgumentParser(description="Generate Arabic/English speech with a ready-made AI voice")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--script", help="Validated script JSON; synthesizes each beat separately")
    source.add_argument("--text", help="UTF-8 narration text file")
    source.add_argument("--content", help="Narration text directly on the command line")
    voice = p.add_mutually_exclusive_group()
    voice.add_argument("--reference", help="Optional voice-cloning reference WAV")
    voice.add_argument("--speaker", help=f"Preset voice name; default: {DEFAULT_SPEAKER}")
    p.add_argument("--list-speakers", action="store_true", help="List preset voices and exit")
    p.add_argument("--language", choices=["ar", "en"], help="Defaults to script language, or ar for text")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--base", default="models/xtts-v2")
    p.add_argument("--checkpoint", help="Fine-tuned best_model.pth; omit for baseline")
    p.add_argument("--output", default="outputs/narration.wav")
    args = p.parse_args()
    import torch

    if args.list_speakers:
        print("\n".join(available_speakers(args.base)))
        return
    texts, language = read_source(
        script_path=args.script, text_path=args.text, content=args.content, language=args.language
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.reference:
        load_audio(args.reference, min_seconds=3, max_seconds=30)
    base = download_model(args.base, training=False)
    model = load_model(base, args.checkpoint, device=device)
    audio, timing = synthesize_segments(model, texts, args.reference, language=language, speaker=args.speaker)
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
            "language": language,
            "voice_mode": "cloned" if args.reference else "preset",
            "speaker": None if args.reference else (args.speaker or DEFAULT_SPEAKER),
            "synthetic_audio": True,
        },
    )
    print(f"Wrote {output} and segment timings (not word-level alignment)")


if __name__ == "__main__":
    main()
