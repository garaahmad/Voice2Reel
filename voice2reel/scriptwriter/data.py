"""Validate records and mask the prompt so loss trains only assistant answers."""

import json
import unicodedata

from voice2reel.common import read_jsonl
from voice2reel.scriptwriter.schema import Script, messages_for


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def load_records(path):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if set(row) != {"id", "idea", "script"}:
            raise ValueError("Each record needs exactly id, idea, script")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Missing or duplicate record ID")
        seen.add(row["id"])
        script = Script.model_validate(row["script"])
        messages_for(row["idea"], script.target_duration_seconds, script.language)
    return rows


def validate_splits(train, validation):
    for key in ("id", "idea"):
        if {normalized(r[key]) for r in train} & {normalized(r[key]) for r in validation}:
            raise ValueError(f"Train/validation leakage in {key}")
    a = {normalized(Script.model_validate(r["script"]).narration()) for r in train}
    b = {normalized(Script.model_validate(r["script"]).narration()) for r in validation}
    if a & b:
        raise ValueError("Train/validation leakage in narration")


def encode_record(row, tokenizer, max_length):
    script = Script.model_validate(row["script"])
    messages = messages_for(row["idea"], script.target_duration_seconds, script.language)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": json.dumps(row["script"], ensure_ascii=False)}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if complete[: len(prompt)] != prompt:
        raise ValueError("Chat template does not preserve the assistant prefix")
    if len(complete) > max_length:
        raise ValueError(f"{row['id']}: {len(complete)} tokens > {max_length}; shorten record or raise limit")
    if len(complete) <= len(prompt):
        raise ValueError("No assistant tokens to train")
    return {
        "input_ids": complete,
        "attention_mask": [1] * len(complete),
        "labels": [-100] * len(prompt) + complete[len(prompt) :],
    }


class AnswerCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch

        length = max(len(x["input_ids"]) for x in examples)
        result = {}
        for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([x[key] + [pad] * (length - len(x[key])) for x in examples])
        return result
