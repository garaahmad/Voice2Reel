"""Generate a safe Python scene from Qwen adapter; do not execute generated code."""
import argparse
from pathlib import Path
from voice2reel.python_scene_dsl import PYTHON_SYSTEM_PROMPT, parse_python_scene
from voice2reel.character_templates import load_lock

def generate(idea, lock_path, adapter, max_new_tokens=1024):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from voice2reel.train_python_scene import MODEL, REVISION
    if not idea or len(idea) > 2000:
        raise ValueError("Idea must be 1..2000 characters")
    lock = load_lock(lock_path)
    prompt = (f"Locked character: {lock['template_id']}. Never alter its design. "
              f"Video idea: {idea}")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, adapter).eval()
    ids = tokenizer.apply_chat_template([
        {"role": "system", "content": PYTHON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True)
    plan = parse_python_scene(raw)
    return raw, plan

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    raw, plan = generate(args.idea, args.lock, args.adapter)
    Path(args.output).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    Path(args.output + ".py.txt").write_text(raw, encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
