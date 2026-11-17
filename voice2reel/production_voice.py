"""Reviewable Arabic narration using multiple Edge neural voices.

Writes only MP3 and speech-script text; never alters scene or character assets.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from voice2reel.production_contract import ProductionPlan

VOICES = {"ar": "ar-JO-TaimNeural", "en": "en-US-GuyNeural"}
VOICE_OPTIONS = {
    "ar": ("ar-JO-TaimNeural", "ar-SA-HamedNeural", "ar-SA-ZariyahNeural",
           "ar-EG-ShakirNeural", "ar-EG-SalmaNeural"),
    "en": ("en-US-GuyNeural", "en-US-JennyNeural"),
}
RATES = ("-15%", "-10%", "-8%", "-5%", "-3%", "+0%", "+3%", "+5%")
PITCHES = ("-4Hz", "-2Hz", "+0Hz", "+2Hz", "+4Hz")
_WORD_SPACE = re.compile(r"\s+")

def prepare_text(text: str) -> str:
    """Fix spacing and sentence breaks; never silently rewrite or translate Arabic."""
    if not isinstance(text, str):
        raise TypeError("Narration text must be a string")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
    text = text.replace(chr(96), "").replace("*", "")
    text = _WORD_SPACE.sub(" ", text).strip()
    text = re.sub(r"\s+([،,.!?؟؛:])", r"\1", text)
    text = re.sub(r"([،؛])(?=\S)", r"\1 ", text)
    if not text:
        raise ValueError("Empty narration text")
    if text[-1] not in ".!?؟":
        text += "."
    return text

def read_override_file(path: str | Path | None) -> dict[int, str]:
    """Optional 1-based shot narration replacements separate from the visual plan."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("shots"), dict):
        raise ValueError("Voice overrides must be JSON: {'shots': {'1': '...'}}")
    output = {}
    for key, value in data["shots"].items():
        if not str(key).isdigit() or int(key) < 1:
            raise ValueError("Narration keys must be 1-based integers")
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 600:
            raise ValueError("Voice override must have 1..600 characters")
        output[int(key)] = value.strip()
    return output

def read_terms_file(path: str | Path | None) -> dict[str, str]:
    """Optional explicit technical-term pronunciations, no automatic translation."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) > 50:
        raise ValueError("Pronunciation terms must be an object with <=50 entries")
    for key, value in data.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 80:
            raise ValueError("Invalid pronunciation term")
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 120:
            raise ValueError("Invalid pronunciation substitution")
    return data

def replace_terms(text: str, terms: dict[str, str]) -> str:
    """Replace whole technical terms, not matching substrings in other words."""
    for key, replacement in sorted(terms.items(), key=lambda item: -len(item[0])):
        # Arabic clitics can attach directly to Latin technical tokens: وsetState.
        # Match Latin token boundaries independently of adjacent Arabic letters.
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])"
        else:
            pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
        text = re.sub(pattern, lambda _m: replacement, text, flags=re.IGNORECASE)
    return text

def script_for_plan(plan: ProductionPlan, overrides=None, terms=None) -> str:
    overrides = overrides or {}
    terms = terms or {}
    if any(i > len(plan.shots) for i in overrides):
        raise ValueError("Override refers to a nonexistent shot")
    clauses = []
    for index, shot in enumerate(plan.shots, 1):
        narration = overrides.get(index, shot.narration)
        clauses.append(prepare_text(replace_terms(narration, terms)))
    return " ".join(clauses)

async def synthesize(plan, output, language="ar", voice=None, rate="+0%",
                     pitch="+0Hz", script_json=None, terms_json=None,
                     preview_text=None):
    import edge_tts
    if language not in VOICES:
        raise ValueError("language must be ar or en")
    chosen_voice = voice or VOICES[language]
    if chosen_voice not in VOICE_OPTIONS[language]:
        raise ValueError(f"Unsupported voice for {language}: {chosen_voice}")
    if rate not in RATES or pitch not in PITCHES:
        raise ValueError(f"Unsupported controls: rate={rate}, pitch={pitch}")
    if preview_text is not None:
        if len(preview_text) > 700:
            raise ValueError("Preview text exceeds 700 characters")
        text = prepare_text(preview_text)
    else:
        plan = ProductionPlan.model_validate(plan)
        text = script_for_plan(
            plan, read_override_file(script_json), read_terms_file(terms_json)
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp3")
    temporary.unlink(missing_ok=True)
    try:
        await edge_tts.Communicate(
            text, chosen_voice, rate=rate, pitch=pitch, volume="+0%"
        ).save(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("TTS returned empty audio")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix(".speech.txt").write_text(text, encoding="utf-8")
    print(f"Voice: {chosen_voice} | rate: {rate} | pitch: {pitch}")
    print("Speech script:", output.with_suffix(".speech.txt"))
    return output

def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan")
    source.add_argument("--preview-text")
    p.add_argument("--output", required=True)
    p.add_argument("--language", choices=tuple(VOICES), default="ar")
    p.add_argument("--voice")
    p.add_argument("--rate", default="+0%")
    p.add_argument("--pitch", default="+0Hz")
    p.add_argument("--script-json")
    p.add_argument("--terms-json")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    result = asyncio.run(synthesize(
        plan, args.output, args.language, args.voice, args.rate, args.pitch,
        args.script_json, args.terms_json, args.preview_text
    ))
    print(result)

if __name__ == "__main__":
    main()
