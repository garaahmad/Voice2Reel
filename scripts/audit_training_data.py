"""Independent leakage/label-consistency audit for Voice2Reel training data.

This is a fast, dependency-free preflight. It does not claim real-world quality.
"""
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def read_rows(relative):
    path = ROOT / relative
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def verify_unique(label, rows, keys):
    for key in keys:
        seen = {}
        for split, samples in rows.items():
            for record in samples:
                value = record.get(key)
                if value is None:
                    continue
                value = norm(value) if isinstance(value, str) else json.dumps(value, sort_keys=True)
                if value in seen:
                    raise ValueError(f"{label}: duplicate {key} in {seen[value]} and {split}")
                seen[value] = split


def audit():
    report = {}
    scripts = {split: read_rows(f"datasets/scripts/{split}.jsonl") for split in SPLITS}
    verify_unique("scripts", scripts, ("id", "idea"))
    report["scripts"] = {split: len(records) for split, records in scripts.items()}
    for section, name, keys in (
        ("arabic", "tashkeel", ("id", "plain")),
        ("arabic", "codeswitch", ("id", "display")),
        ("motion", "motion_plans", ("id",)),
        ("stickman", "drawing", ("id",)),
    ):
        rows = {split: read_rows(f"datasets/{section}/{name}_{split}.jsonl") for split in SPLITS}
        verify_unique(name, rows, keys)
        report[name] = {split: len(records) for split, records in rows.items()}
        if name == "drawing":
            labels = {}
            for split, records in rows.items():
                counts = Counter("professional" if r["defect"] is None else r["defect"] for r in records)
                if any(len(r["points"]) != 18 or len(r["features"]) != 14 for r in records):
                    raise ValueError(f"stickman invalid feature/point dimensions in {split}")
                if any(
                    not all(math.isfinite(float(n)) for point in r["points"] for n in point)
                    or not all(math.isfinite(float(n)) for n in r["features"])
                    for r in records
                ):
                    raise ValueError(f"stickman nonfinite numeric data in {split}")
                labels[split] = dict(sorted(counts.items()))
            report["stickman_labels"] = labels
    # Dataset size/uniqueness checks do not prove independence of related
    # templates, professional aesthetic quality, or natural audio.
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
