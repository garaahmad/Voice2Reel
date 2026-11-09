"""Evaluate scripts, or compare manually checked/ASR transcripts for speech errors."""

import argparse
import re
import unicodedata
from collections import Counter

from voice2reel.arabic_text import tashkeel_accuracy
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


def content_f1(reference, hypothesis):
    """Word-overlap F1 on diacritic-stripped text (not factual correctness)."""
    ref = Counter(normalize_arabic(reference).split())
    hyp = Counter(normalize_arabic(hypothesis).split())
    overlap = sum((ref & hyp).values())
    precision = overlap / max(1, sum(hyp.values()))
    recall = overlap / max(1, sum(ref.values()))
    return 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)


def speech_metrics(rows):
    word_errors = char_errors = words = chars = mark_scores = 0.0
    for row in rows:
        ref, hyp = normalize_arabic(row["reference"]), normalize_arabic(row["hypothesis"])
        if not ref:
            raise ValueError("Reference transcript must not be empty")
        word_errors += edit_distance(ref.split(), hyp.split())
        char_errors += edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        words += len(ref.split())
        chars += len(ref.replace(" ", ""))
        # Exact per-letter diacritic agreement is reported separately, because
        # WER/CER deliberately strip tashkeel to measure content only.
        mark_scores += tashkeel_accuracy(row["reference"], row["hypothesis"])
    if not words:
        raise ValueError("No evaluation records")
    return {
        "wer": word_errors / words,
        "cer": char_errors / chars,
        "tashkeel_accuracy": mark_scores / len(rows),
        "utterances": len(rows),
        "note": "WER/CER measure content only; tashkeel_accuracy measures exact marks. ASR errors affect both; voice similarity needs separate listening.",
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
                reference = " ".join(b["narration"] for b in row["script"]["beats"])
                results.append(
                    {
                        "id": row["id"], "valid": True, "attempts": attempts,
                        "content_f1": content_f1(reference, script.narration()),
                        "output": script.model_dump(),
                    }
                )
            except ValueError as exc:
                results.append({"id": row["id"], "valid": False, "error": str(exc)})
        valid_rate = sum(r["valid"] for r in results) / len(results)
        first_pass_rate = sum(r.get("attempts") == 1 for r in results) / len(results)
        content_scores = [r["content_f1"] for r in results if r["valid"]]
        mean_content_f1 = sum(content_scores) / len(content_scores) if content_scores else 0.0
        result = {
            "valid_rate": valid_rate,
            "first_pass_rate": first_pass_rate,
            "mean_content_f1": round(mean_content_f1, 4),
            "score": round(0.5 * valid_rate + 0.3 * mean_content_f1 + 0.2 * first_pass_rate, 4),
            "records": results,
            "note": "Schema validity and word overlap do not establish factual or creative quality; human review is still required.",
        }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
