"""Generate narration audio for a validated production plan using Edge TTS.

Online service; text leaves the local machine. This does not guarantee
per-shot synchronization; alignment is a separate production requirement.
"""
import argparse
import asyncio
import json
from pathlib import Path
from voice2reel.production_contract import ProductionPlan

VOICES = {"ar": "ar-SA-HamedNeural", "en": "en-US-GuyNeural"}
VOICE_OPTIONS = {"ar": ("ar-SA-HamedNeural", "ar-JO-TaimNeural", "ar-EG-SalmaNeural"),
                 "en": ("en-US-GuyNeural", "en-US-JennyNeural")}

async def synthesize(plan, output, language="ar", voice=None, rate="-8%"):
    import edge_tts
    plan = ProductionPlan.model_validate(plan)
    if language not in VOICES:
        raise ValueError("language must be ar or en")
    chosen_voice = voice or VOICES[language]
    if chosen_voice not in VOICE_OPTIONS[language]:
        raise ValueError(f"Unsupported voice for {language}: {chosen_voice}")
    if rate not in ("-15%", "-10%", "-8%", "-5%", "+0%", "+5%"):
        raise ValueError("Unsupported speech rate")
    text = " ".join(shot.narration.strip() for shot in plan.shots)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    await edge_tts.Communicate(text, chosen_voice, rate=rate).save(str(output))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("TTS did not produce audio")
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--language", choices=tuple(VOICES), default="ar")
    p.add_argument("--voice", default=None)
    p.add_argument("--rate", default="-8%")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    print(asyncio.run(synthesize(plan, args.output, args.language, args.voice, args.rate)))

if __name__ == "__main__":
    main()
