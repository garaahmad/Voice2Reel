"""Generate a validated, safe scene DSL from a trained Qwen adapter.

Generated DSL is parsed as data, never executed.
"""
import argparse
from pathlib import Path

from voice2reel.python_scene_dsl import PYTHON_SYSTEM_PROMPT, parse_python_scene
from voice2reel.character_templates import load_lock


def generate(idea, lock_path, adapter, max_new_tokens=1024, max_attempts=3):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from voice2reel.train_python_scene import MODEL, REVISION

    if not idea or len(idea) > 2000:
        raise ValueError("Idea must be 1..2000 characters")
    if not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5")

    lock = load_lock(lock_path)
    prompt = (f"Locked character: {lock['template_id']}. Never alter its design. "
              f"Video idea: {idea}")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, adapter).eval()

    messages = [
        {"role": "system", "content": PYTHON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_error = None
    for attempt in range(1, max_attempts + 1):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        input_length = encoded["input_ids"].shape[-1]
        raw = tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
        try:
            plan = parse_python_scene(raw)
            return raw, plan
        except (ValueError, SyntaxError) as exc:
            last_error = str(exc)
            print(f"Scene validation failed (attempt {attempt}/{max_attempts}): {last_error}", flush=True)
            if attempt == max_attempts:
                break
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your previous scene DSL failed validation. Correct the ENTIRE script "
                    "and output ONLY the corrected scene() and shot() calls. "
                    "Do not add any explanations or unsupported fields. "
                    "Allowed gestures: idle, present, point, wave, think, explain, celebrate, step. "
                    "Validation error: " + last_error[:1200]
                )},
            ])
    raise ValueError(
        f"Qwen could not produce a valid scene after {max_attempts} attempts. "
        f"Last validation error: {last_error}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-attempts", type=int, default=3)
    args = p.parse_args()
    raw, plan = generate(args.idea, args.lock, args.adapter,
                         max_attempts=args.max_attempts)
    Path(args.output).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    Path(args.output + ".py.txt").write_text(raw, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
