"""Validate all starter splits without downloading model weights."""

from voice2reel.scriptwriter.data import load_records, validate_splits

if __name__ == "__main__":
    splits = {
        name: load_records(f"datasets/scripts/{name}.jsonl") for name in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        validate_splits(splits[a], splits[b])
    print({name: len(rows) for name, rows in splits.items()})
