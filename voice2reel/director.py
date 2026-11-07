"""Run the project's Qwen twice: director first, narration second."""

import argparse
import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import Field

from voice2reel.common import write_json
from voice2reel.scriptwriter.generate import ScriptGenerator
from voice2reel.scriptwriter.schema import Script, ShortText, StrictModel, Text
from voice2reel.scriptwriter.train import MODEL_ID, MODEL_REVISION
from voice2reel.storyboard import Brief, Plan, QUESTIONS


class NarrationBeat(StrictModel):
    narration: Text
    caption: ShortText


class Narration(StrictModel):
    title: ShortText
    beats: Annotated[list[NarrationBeat], Field(min_length=3, max_length=8)]


def validate_narration_language(narration, language):
    for beat in narration.beats:
        for field in ("narration", "caption"):
            text = getattr(beat, field)
            has_arabic = re.search(r"[\u0621-\u064a]", text) is not None
            if language == "ar" and not has_arabic:
                raise ValueError(f"{field} must be written in Arabic, including captions")
            if language == "en" and (has_arabic or not re.search(r"[A-Za-z]", text)):
                raise ValueError(f"{field} must be written in English")


def clean_response(raw):
    """Repair JSON syntax only; callers must still validate schema and content."""
    cleaned = raw.strip()
    for fence in ("```json\n", "```\n"):
        if cleaned.startswith(fence) and cleaned.endswith("```"):
            cleaned = cleaned[len(fence):-3].strip()
            break
    try:
        json.loads(cleaned)
    except json.JSONDecodeError:
        from json_repair import repair_json

        return repair_json(cleaned, ensure_ascii=False), True
    return cleaned, False


def structured_response(generator, messages, validate, trace_path, max_tokens=2200):
    import torch
    trace = []
    original_messages = list(messages)
    for attempt in range(3):
        print(f"Generating {Path(trace_path).stem}: attempt {attempt + 1}/3", flush=True)
        inputs = generator.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(generator.model.device)
        with torch.inference_mode():
            result = generator.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
        raw = generator.tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        trace.append({"attempt": attempt + 1, "raw_response": raw})
        try:
            cleaned, repaired = clean_response(raw)
            if repaired:
                trace[-1]["json_repaired"] = True
                trace[-1]["repaired_response"] = cleaned
            value = validate(cleaned)
        except ValueError as exc:
            trace[-1]["error"] = str(exc)
            write_json(trace_path, trace)
            print("Output failed validation; asking the model to repair it.", flush=True)
            messages = original_messages + [
                {"role": "user", "content": (
                    "The previous response failed validation: " + str(exc)[:500] +
                    "\nReturn the complete corrected JSON object only. No explanations or apologies."
                )},
            ]
        else:
            write_json(trace_path, trace)
            return value
    raise ValueError(f"Model failed validation after 3 attempts; inspect {trace_path}")


def create_project(generator, brief, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "generation_status.json", {"status": "planning"})
    write_json(output / "brief.json", brief.model_dump())

    def check_plan(raw):
        plan = Plan.model_validate_json(raw)
        if len(plan.shots) != brief.shot_count:
            raise ValueError(f"Exactly {brief.shot_count} shots required")
        return plan

    # A text-only model must not be told it has viewed the uploaded reference.
    context = brief.model_dump(exclude={"reference_video"})
    plan = structured_response(generator, [
        {"role": "system", "content": (
            "You direct realistic short films. Write JSON only. Plan visually achievable shots, "
            "each with one simple action and a camera movement. Be concise: one sentence per field. "
            "Keep the same person, clothing, "
            "location and lighting in continuity. Write ALL fields in English, including title and purpose. "
            "Do not ask the video model to draw "
            "readable code, letters or diagrams. Avoid invented facts. purpose states what the "
            "narrator should explain. Describe physical actions, not source code or technical claims. "
            "The factual explanation is a separate voice-over. Keep computer screens out of focus; "
            "show human reactions, gestures and the environment. Never quote code in any field. "
            "Return a JSON instance, not a schema. Required structure: "
            '{"title":"short title","continuity":"English description of consistent subject and setting",'
            '"shots":[{"purpose":"what the narrator explains","visual_prompt":"English physical scene and camera",'
            '"opening_action":"English beginning action","closing_action":"English ending action"}]} '
            f"Include exactly {brief.shot_count} shots."
        )},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False) +
         "\nWrite continuity, visual_prompt, opening_action and closing_action in ENGLISH. "
         "Use physical actions and camera movement, not readable code on screen."},
    ], check_plan, output / "director.trace.json", max_tokens=500 * brief.shot_count)
    write_json(output / "plan.json", plan.model_dump())
    write_json(output / "generation_status.json", {"status": "writing_script"})

    def check_narration(raw):
        narration = Narration.model_validate_json(raw)
        validate_narration_language(narration, brief.language)
        if len(narration.beats) != len(plan.shots):
            raise ValueError("One narration beat per director shot required")
        return Script.model_validate({
            "schema_version": 1, "title": narration.title, "language": brief.language,
            "target_duration_seconds": brief.target_duration_seconds,
            "beats": [{**beat.model_dump(), "pose": "IDLE", "camera": "medium"}
                      for beat in narration.beats],
        })

    script = structured_response(generator, [
        {"role": "system", "content": (
            "Write natural short narration following the director's shots in order. JSON only. "
            "Use Arabic for ar or English for en in BOTH narration and caption. One sentence per "
            "shot, about 7 to 11 words. Explain the supplied message accurately. No invented facts "
            "or generalizations; do not describe camera instructions aloud. Return a JSON instance, not a schema. "
            'Required structure: {"title":"short title in the narration language",'
            '"beats":[{"narration":"spoken sentence","caption":"short caption in the SAME language"}]} '
            f"Include exactly {len(plan.shots)} beats."
        )},
        {"role": "user", "content": json.dumps({"brief": context, "plan": plan.model_dump()}, ensure_ascii=False)},
    ], check_narration, output / "script.trace.json", max_tokens=1200)
    write_json(output / "script.json", script.model_dump())
    write_json(output / "provenance.json", {
        "director_model": MODEL_ID, "revision": MODEL_REVISION,
        "stages": ["director", "narrator"], "reference_analysis": "none",
        "factual_accuracy_verified": False,
    })
    write_json(output / "generation_status.json", {"status": "ready_for_speech"})
    return plan, script


def main():
    p = argparse.ArgumentParser(description="Idea + at most five questions → director plan → script")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--brief", help="Brief JSON; skips interactive questions")
    source.add_argument("--idea", help="Written idea; answer exactly five questions in the terminal")
    p.add_argument("--reference-video", help="Optional local video; its first frame conditions the first clip")
    p.add_argument("--output", default="outputs/realistic")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-dtype", choices=["float32", "bfloat16"], default="float32")
    args = p.parse_args()
    if args.brief:
        brief = Brief.model_validate_json(Path(args.brief).read_text(encoding="utf-8"))
    else:
        answers = {key: input(question + " ").strip() for key, question in QUESTIONS}
        answers["target_duration_seconds"] = int(answers["target_duration_seconds"])
        brief = Brief(idea=args.idea, **answers)
    if args.reference_video:
        brief.reference_video = args.reference_video
    if brief.reference_video and not Path(brief.reference_video).is_file():
        raise ValueError("Reference video does not exist")
    generator = ScriptGenerator(device=args.device, cpu_dtype=args.cpu_dtype)
    create_project(generator, brief, args.output)
    print(f"Saved director plan and script: {args.output}. Review facts before generating expensive video.")


if __name__ == "__main__":
    main()
