"""QLoRA for strict Qwen text -> CharacterSpec JSON, never raw SVG."""
import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Empty dataset: " + str(path))
    from voice2reel.character_spec import parse_spec
    for row in rows:
        if len(row["messages"]) != 3:
            raise ValueError("Expected system, user, assistant")
        parse_spec(row["messages"][-1]["content"])
    return rows

def encode(row, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(row["messages"][:2], tokenize=True,
                                           add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                         add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Tokenizer prefix mismatch")
    if len(full) > max_length:
        raise ValueError("Sequence too long: " + row["id"])
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="datasets/character_spec/train.jsonl")
    p.add_argument("--validation", default="datasets/character_spec/validation.jsonl")
    p.add_argument("--output", default="outputs/qwen_character_spec")
    p.add_argument("--check-only", action="store_true", help="Validate SVG and tokenizer lengths without a GPU or training")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=3072)
    args = p.parse_args()
    from transformers import AutoTokenizer
    train = load(args.train)
    validation = load(args.validation)
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("ID split leakage")
    if {r["messages"][1]["content"] for r in train} & {r["messages"][1]["content"] for r in validation}:
        raise ValueError("Prompt split leakage")
    print(f"✅ CharacterSpec JSON dataset checked: train={len(train)}, validation={len(validation)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    print("🔎 Checking token lengths and assistant masks...", flush=True)
    encoded_train = [encode(r, tokenizer, args.max_length) for r in train]
    encoded_validation = [encode(r, tokenizer, args.max_length) for r in validation]
    print(f"✅ Tokenizer checks passed. Longest train record: "
          f"{max(len(r['input_ids']) for r in encoded_train)} tokens; "
          f"validation: {max(len(r['input_ids']) for r in encoded_validation)} tokens", flush=True)
    if args.check_only:
        print("✅ Preflight complete; no model loaded and no training started.", flush=True)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. In Google Colab select Runtime > Change runtime type > T4 GPU, "
            "then restart the setup and retry.")
    from transformers import (EarlyStoppingCallback, AutoModelForCausalLM, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from voice2reel.scriptwriter.data import AnswerCollator
    from voice2reel.common import write_json, digest_file
    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, device_map={"": 0},
        dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                        "up_proj", "down_proj"]))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh, empty CharacterSpec adapter directory")
    trainer = Trainer(
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        model=model, processing_class=tokenizer,
        train_dataset=encoded_train,
        eval_dataset=encoded_validation,
        data_collator=AnswerCollator(tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, learning_rate=1e-4,
            warmup_ratio=0.1, lr_scheduler_type="cosine",
            fp16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=2,
            logging_steps=10, report_to="none", seed=42))
    print(f"🏋️ Training CharacterSpec adapter at {output}", flush=True)
    trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    write_json(output / "metrics.json", trainer.evaluate())
    write_json(output / "provenance.json", {
        "model": MODEL, "revision": REVISION,
        "task": "text-to-CharacterSpec-json-v1",
        "train_sha256": digest_file(args.train),
        "validation_sha256": digest_file(args.validation),
        "train_rows": len(train), "validation_rows": len(validation),
        "gpu": torch.cuda.get_device_name(0), "args": vars(args),
    })

if __name__ == "__main__":
    main()
