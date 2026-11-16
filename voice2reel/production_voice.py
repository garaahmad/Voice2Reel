"""Generate narration audio for a validated production plan using Edge TTS.

Online service; text leaves the local machine. This does not guarantee
per-shot synchronization; alignment is a separate production requirement.
"""
import argparse
import asyncio
import json
from pathlib import Path
from voice2reel.production_contract import ProductionPlan

VOICES = {"ar": "ar-JO-TaimNeural", "en": "en-US-GuyNeural"}

async def synthesize(plan, output, language="ar"):
    import edge_tts
    plan = ProductionPlan.model_validate(plan)
    if language not in VOICES:
        raise ValueError("language must be ar or en")
    text = " ".join(shot.narration for shot in plan.shots)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    await edge_tts.Communicate(text, VOICES[language]).save(str(output))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("TTS did not produce audio")
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--language", choices=tuple(VOICES), default="ar")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    print(asyncio.run(synthesize(plan, args.output, args.language)))

if __name__ == "__main__":
    main()
