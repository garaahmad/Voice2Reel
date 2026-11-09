"""Validate all committed v2 and stickman drawing splits without model weights."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice2reel.quality_gate import dataset_integrity  # noqa: E402
from voice2reel.scriptwriter.data import load_records, validate_splits  # noqa: E402
from voice2reel.stickman_artist import run_rounds  # noqa: E402


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


def _check_builder(script, label):
    """Rebuild in memory and compare committed bytes; fails on any data drift."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(f"{label} reproducibility failed")
    return result.stdout.strip()


if __name__ == "__main__":
    splits = {
        name: load_records(ROOT / "datasets" / "scripts" / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        validate_splits(splits[a], splits[b])

    _check_builder("build_datasets.py", "v2")
    _check_builder("build_stickman_drawing_data.py", "stickman")

    integrity = dataset_integrity()
    failures = _boolean_failures(integrity)

    # Recompute the measured CPU report and compare it with the committed one.
    report = run_rounds(max_rounds=3)
    stickman_checks = report["integrity"]
    failures += _boolean_failures(stickman_checks, "stickman")
    committed_path = ROOT / "examples" / "stickman_drawing" / "artist_rounds.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    report_matches = committed == report
    if not report_matches:
        failures.append("stickman.artist_rounds_committed")

    if failures:
        raise SystemExit("Dataset integrity failed: " + ", ".join(failures))
    print(json.dumps({
        "script_counts": {name: len(rows) for name, rows in splits.items()},
        "integrity": integrity,
        "stickman_integrity": stickman_checks,
        "stickman_rounds": [item["score"] for item in report["rounds"]],
        "stickman_final_test_score": report["final_test_on_best_round"]["score"],
        "stickman_report_matches": report_matches,
        "reproducible": True,
    }, ensure_ascii=False))
