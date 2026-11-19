"""Generate validated JSON design instructions with trained Qwen, then render a rigged mascot.

The model cannot emit arbitrary SVG. Geometry and rig stay trusted/authored.
"""
from __future__ import annotations

import argparse
import json

from voice2reel.character_spec import SYSTEM_PROMPT, parse_spec, save_spec


def _extract_json(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        raise ValueError("Qwen produced no JSON object")
    payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(payload, dict):
        raise ValueError("Qwen output must be a JSON object")
    return json.dumps(payload, ensure_ascii=False)


def generate(prompt: str, adapter: str, attempts: int = 3,
             max_new_tokens: int = 320):
    if not prompt.strip() or len(prompt) > 2000:
        raise ValueError("Prompt must be 1..2000 characters")
    if not 1 <= attempts <= 5:
        raise ValueError("Attempts must be 1..5")
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from voice2reel.train_character_spec import MODEL, REVISION

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter).eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_error = "No valid JSON yet"
    for attempt in range(1, attempts + 1):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        encoded = {name: tensor.to(model.device)
                   for name, tensor in encoded.items()}
        with torch.inference_mode():
            result = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        raw = tokenizer.decode(
            result[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        try:
            spec = parse_spec(_extract_json(raw))
            print(f"✅ CharacterSpec validated (attempt {attempt}/{attempts})", flush=True)
            return spec
        except (ValueError, json.JSONDecodeError) as error:
            last_error = str(error)
            print(f"⚠️ Invalid CharacterSpec attempt {attempt}: {last_error[:350]}", flush=True)
            if attempt < attempts:
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Correct the previous invalid response. Output exactly one "
                        "JSON object with the six allowed design attributes plus "
                        "version=1, no SVG, no code, no additional keys. Error: "
                        + last_error[:800]
                    )},
                ])
    raise ValueError(
        f"Qwen failed to produce a valid CharacterSpec after {attempts} attempts: "
        + last_error[:600]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--spec-out", required=True)
    p.add_argument("--preview", required=True)
    p.add_argument("--attempts", type=int, default=3)
    args = p.parse_args()
    spec = generate(args.prompt, args.adapter, attempts=args.attempts)
    save_spec(spec, args.spec_out)
    from voice2reel.mascot_animator import MascotAnimator
    from pathlib import Path
    preview = Path(args.preview)
    preview.parent.mkdir(parents=True, exist_ok=True)
    MascotAnimator(spec=spec).frame_image("idle", 1.25, 3, width=512).save(preview)
    print("🎨 Validated JSON:", args.spec_out)
    print("🖼 Rig-safe preview PNG:", preview)


if __name__ == "__main__":
    main()
