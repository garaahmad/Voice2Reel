"""Evaluate scripts, or compare manually checked/ASR transcripts for speech errors."""

import argparse
import re
import unicodedata

from voice2reel.common import read_jsonl, write_json


def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("[\u064b-\u065f\u0670\u0640]", "", text)
    text = re.sub("[أإآ]", "ا", text)
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def speech_metrics(rows):
    word_errors = char_errors = words = chars = 0
    for row in rows:
        ref, hyp = normalize_arabic(row["reference"]), normalize_arabic(row["hypothesis"])
        if not ref:
            raise ValueError("Reference transcript must not be empty")
        word_errors += edit_distance(ref.split(), hyp.split())
        char_errors += edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        words += len(ref.split())
        chars += len(ref.replace(" ", ""))
    if not words:
        raise ValueError("No evaluation records")
    return {
        "wer": word_errors / words,
        "cer": char_errors / chars,
        "utterances": len(rows),
        "note": "Content accuracy only; ASR errors affect scores. Voice similarity needs separate listening.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["script", "speech"])
    p.add_argument("--input", required=True)
    p.add_argument("--adapter")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = read_jsonl(args.input)
    if args.mode == "speech":
        result = speech_metrics(rows)
    else:
        from voice2reel.scriptwriter.generate import ScriptGenerator

        generator = ScriptGenerator(args.adapter)
        results = []
        for row in rows:
            try:
                script, attempts = generator.generate(
                    row["idea"], row["script"]["target_duration_seconds"], language=row["script"]["language"]
                )
                results.append(
                    {"id": row["id"], "valid": True, "attempts": attempts, "output": script.model_dump()}
                )
            except ValueError as exc:
                results.append({"id": row["id"], "valid": False, "error": str(exc)})
        result = {
            "valid_rate": sum(r["valid"] for r in results) / len(results),
            "first_pass_rate": sum(r.get("attempts") == 1 for r in results) / len(results),
            "records": results,
            "note": "Schema validity does not establish factual or creative quality",
        }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
