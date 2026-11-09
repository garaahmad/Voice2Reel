"""QLoRA training for a T4. Run with python -m voice2reel.scriptwriter.train."""

import argparse
from pathlib import Path

from voice2reel.common import digest_file, write_json
from voice2reel.scriptwriter.data import AnswerCollator, encode_record, load_records, validate_splits

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="datasets/scripts/train.jsonl")
    p.add_argument("--validation", default="datasets/scripts/validation.jsonl")
    p.add_argument("--output", default="outputs/scriptwriter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--train-limit", type=int, default=0,
                   help="Use only the first N train rows (0 = all); documents data-volume cycles")
    p.add_argument("--resume", help="Full checkpoint directory, including optimizer state")
    args = p.parse_args()
    if args.epochs < 1 or args.max_length < 64 or args.max_steps == 0 or args.max_steps < -1:
        p.error("epochs >= 1, max-length >= 64, max-steps must be -1 or positive")
    if args.train_limit < 0:
        p.error("train-limit must be 0 or positive")
    train, validation = load_records(args.train), load_records(args.validation)
    train = train[: args.train_limit] if args.train_limit else train
    validate_splits(train, validation)
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab (T4 or better)")
    if Path(args.output).exists() and any(Path(args.output).iterdir()) and not args.resume:
        raise ValueError("Output is not empty; use --resume or a new output directory")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_data = [encode_record(r, tokenizer, args.max_length) for r in train]
    eval_data = [encode_record(r, tokenizer, args.max_length) for r in validation]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map={"": 0},
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    output = Path(args.output)
    metadata = {
        "base_model": MODEL_ID,
        "revision": MODEL_REVISION,
        "seed": 42,
        "train_sha256": digest_file(args.train),
        "train_rows": len(train),
        "validation_sha256": digest_file(args.validation),
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.resume:
        import json

        previous = json.loads((output / "run.json").read_text())
        for key in ("base_model", "revision", "train_sha256", "train_rows", "validation_sha256"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Cannot resume: {key} changed")
    write_json(output / "run.json", metadata)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=AnswerCollator(tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(output),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=1,
            report_to="none",
            seed=42,
            data_seed=42,
            dataloader_num_workers=0,
            max_grad_norm=1.0,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    write_json(output / "metrics.json", trainer.evaluate())
    write_json(output / "adapter" / "provenance.json", metadata)


if __name__ == "__main__":
    main()
