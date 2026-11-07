import copy
import json
from pathlib import Path

import pytest

from voice2reel.scriptwriter.data import AnswerCollator, encode_record, load_records, validate_splits
from voice2reel.scriptwriter.schema import Script, messages_for, parse_script

ROOT = Path(__file__).resolve().parents[1]


def example():
    return load_records(ROOT / "datasets/scripts/train.jsonl")[0]


def test_all_splits_are_valid_and_disjoint():
    splits = [
        load_records(ROOT / f"datasets/scripts/{name}.jsonl") for name in ("train", "validation", "test")
    ]
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            validate_splits(a, b)


@pytest.mark.parametrize("mutation", ["pose", "duration", "extra", "empty", "too_long"])
def test_invalid_outputs_rejected(mutation):
    value = copy.deepcopy(example()["script"])
    if mutation == "pose":
        value["beats"][0]["pose"] = "INVENTED_POSE"
    elif mutation == "duration":
        value["target_duration_seconds"] = 600
    elif mutation == "extra":
        value["python_code"] = "print('invalid field')"
    elif mutation == "empty":
        value["beats"][0]["narration"] = " "
    else:
        value["beats"][0]["narration"] = "كلمة " * 100
    with pytest.raises(ValueError):
        Script.model_validate(value)


def test_json_and_requested_duration():
    with pytest.raises(ValueError):
        parse_script("```json\n{}\n```")
    with pytest.raises(ValueError):
        parse_script(json.dumps(example()["script"]), expected_duration=30)
    with pytest.raises(ValueError):
        messages_for(" ")
    assert Script.model_validate(example()["script"]).narration()
    raw = json.dumps(example()["script"])
    assert parse_script("```json\n" + raw + "\n```").title == example()["script"]["title"]


def test_leakage_rejected():
    record = example()
    with pytest.raises(ValueError):
        validate_splits([record], [record])
    other = copy.deepcopy(record)
    other.update(id="other", idea="فكرة أخرى")
    with pytest.raises(ValueError, match="narration"):
        validate_splits([record], [other])


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 3, 4, 5, 0]


def test_prompt_mask_padding_and_eos():
    encoded = encode_record(example(), FakeTokenizer(), 6)
    assert encoded["labels"] == [-100, -100, -100, 4, 5, 0]
    shorter = {key: value[:-2] for key, value in encoded.items()}
    batch = AnswerCollator(0)([encoded, shorter])
    assert batch["labels"].tolist()[1] == [-100, -100, -100, 4, -100, -100]
    assert batch["labels"][0, -1] == 0  # EOS remains a training target, unlike padding.
    with pytest.raises(ValueError, match="tokens"):
        encode_record(example(), FakeTokenizer(), 4)
