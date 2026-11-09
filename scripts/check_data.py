"""Validate all committed v2 splits without downloading model weights."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice2reel.quality_gate import dataset_integrity  # noqa: E402
from voice2reel.scriptwriter.data import load_records, validate_splits  # noqa: E402


def _boolean_failures(value, prefix=""):
    if isinstance(value, bool):
        return [] if value else [prefix or "check"]
    if isinstance(value, dict):
        return [
            item
            for key, nested in value.items()
            for item in _boolean_failures(nested, f"{prefix}.{key}" if prefix else key)
        ]
    return []


if __name__ == "__main__":
    splits = {
        name: load_records(ROOT / "datasets" / "scripts" / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        validate_splits(splits[a], splits[b])

    # Rebuild in memory and compare committed bytes; fails on any data drift.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_datasets.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)

    integrity = dataset_integrity()
    failures = _boolean_failures(integrity)
    if failures:
        raise SystemExit("Dataset integrity failed: " + ", ".join(failures))
    print(json.dumps({
        "script_counts": {name: len(rows) for name, rows in splits.items()},
        "integrity": integrity,
        "reproducible": True,
    }, ensure_ascii=False))
