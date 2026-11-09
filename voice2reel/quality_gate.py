"""Measurable quality gate for the v2 datasets.

Three front-end models are ACTUALLY fitted from training rows here on CPU:
  * Diacritizer      : plain Arabic -> fully vowelized Arabic (word/affix lexicon)
  * CodeSwitchReader : mixed display text -> Arabic-only speech-engine text
  * ComfortClassifier: motion plan features -> comfortable / annoying

The Qwen scriptwriter is NOT trained here (no GPU/weights in this environment);
its training loop lives in ``notebooks/05_quality_loop.ipynb`` for Colab.
Every number produced by this module is a real measurement on the committed files.
"""

import csv
import json
import re
from pathlib import Path

from voice2reel.arabic_text import (
    load_lexicon, mark_names, spoken_form, strip_tashkeel, tashkeel_accuracy,
    vowelized_ratio,
)
from voice2reel.common import read_jsonl
from voice2reel.motion_plan import ComfortClassifier, comfort_score, validate_plan
from voice2reel.scriptwriter.data import load_records, validate_splits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
SPLITS = ("train", "validation", "test")
FILES = {
    "tashkeel": DATA / "arabic",
    "codeswitch": DATA / "arabic",
    "motion_plans": DATA / "motion",
}

# Arabic clitics are attached to the word in display text; a fitted model must
# learn to split them before looking up the stem.
PREFIXES = ("وال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
SUFFIXES = ("هم", "ها", "نا", "ه", "ك", "ي")
PREFIX_READING = {"و": "وَ", "ف": "فَ", "ب": "بِ", "ك": "كَ", "ل": "لِ", "ال": "الْ",
                  "وال": "وَالْ", "بال": "بِالْ", "كال": "كَالْ", "لل": "لِلْ"}
SUFFIX_READING = {"ه": "هُ", "ها": "هَا", "هم": "هُمْ", "نا": "نَا", "ك": "كَ", "ي": "ي"}
LETTER_READING = {
    "a": "أَ", "b": "بي", "c": "سي", "d": "دي", "e": "إي", "f": "إف", "g": "جي",
    "h": "إتش", "i": "آي", "j": "جاي", "k": "كاي", "l": "إل", "m": "إم",
    "n": "إن", "o": "أو", "p": "بي", "q": "كيو", "r": "آر", "s": "إس",
    "t": "تي", "u": "يو", "v": "في", "w": "دابليو", "x": "إكس", "y": "واي", "z": "زد",
}
FINAL_MARKS = "\u064e\u064f\u0650\u064b\u064c\u064d\u0652"


def rows(dataset, split):
    return read_jsonl(FILES[dataset] / f"{dataset}_{split}.jsonl")


def scripts(split):
    return load_records(DATA / "scripts" / f"{split}.jsonl")


class Diacritizer:
    """Word-level vowelization model with optional learned affix decomposition."""

    def __init__(self, use_affixes=True, use_defaults=True):
        self.use_affixes = use_affixes
        self.use_defaults = use_defaults
        self.lexicon = {}

    def fit(self, records):
        self.lexicon = {}
        for row in records:
            plain, vowelized = row["plain"].split(), row["vowelized"].split()
            if len(plain) != len(vowelized):
                continue
            for p, v in zip(plain, vowelized):
                if strip_tashkeel(v) == p and p not in self.lexicon:
                    self.lexicon[p] = v
        return self

    @staticmethod
    def _attach(prefix, vowelized, suffix):
        if not suffix:
            return PREFIX_READING.get(prefix, prefix) + vowelized
        stem = vowelized.rstrip(FINAL_MARKS)
        return PREFIX_READING.get(prefix, prefix) + stem + SUFFIX_READING.get(suffix, suffix)

    def _lookup(self, word):
        if word in self.lexicon:
            return self.lexicon[word]
        if not self.use_affixes:
            return None
        for prefix in sorted(PREFIXES, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                stem = word[len(prefix):]
                if stem in self.lexicon:
                    return self._attach(prefix, self.lexicon[stem], "")
                for suffix in sorted(SUFFIXES, key=len, reverse=True):
                    if stem.endswith(suffix) and len(stem) > len(suffix) + 1:
                        base = stem[: -len(suffix)]
                        if base in self.lexicon:
                            return self._attach(prefix, self.lexicon[base], suffix)
        for suffix in sorted(SUFFIXES, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                base = word[: -len(suffix)]
                if base in self.lexicon:
                    return self._attach("", self.lexicon[base], suffix)
        return None

    def predict(self, plain):
        out = []
        for word in plain.split():
            hit = self._lookup(word)
            # Last-resort reading: leave the word unvowelized rather than invent marks.
            out.append(hit if hit is not None else word)
        return " ".join(out)

    def metrics(self, records):
        if not records:
            return {"word_accuracy": 0.0, "sentence_accuracy": 0.0, "mark_accuracy": 0.0}
        total = exact = marks = 0.0
        for row in records:
            predicted = self.predict(row["plain"])
            reference = row["vowelized"]
            p, r = predicted.split(), reference.split()
            total += len(r)
            exact += sum(a == b for a, b in zip(p, r))
            marks += tashkeel_accuracy(reference, predicted)
        return {
            "word_accuracy": round(exact / total, 4),
            "sentence_accuracy": round(sum(
                self.predict(row["plain"]) == row["vowelized"] for row in records
            ) / len(records), 4),
            "mark_accuracy": round(marks / len(records), 4),
        }


class CodeSwitchReader:
    """Learn surface -> Arabic reading from train rows, then normalise display text."""

    def __init__(self, use_normalization=True, use_fallback=True):
        self.use_normalization = use_normalization
        self.use_fallback = use_fallback
        self.terms = {}

    def fit(self, records):
        self.terms = {}
        for row in records:
            for term in row.get("terms", []):
                self.terms.setdefault(term["surface"], term["reading"])
        return self

    def predict(self, display):
        lexicon = {
            k: {"spoken": v, "allowed_in_narration": True} for k, v in self.terms.items()
        }
        spoken, unresolved = spoken_form(display, lexicon)
        if self.use_normalization:
            spoken = spoken.replace("الـ ", "ال").replace("الـ", "ال").strip()
        if self.use_fallback:
            for token in unresolved:
                spelled = " ".join(
                    LETTER_READING.get(c.lower(), c) for c in token if c.isalnum()
                )
                spoken = re.sub(r"(?<!\w)" + re.escape(token) + r"(?!\w)", spelled, spoken)
        return spoken

    def metrics(self, records):
        if not records:
            return {"exact_rate": 0.0, "latin_free_rate": 0.0, "term_coverage": 0.0}
        predicted = [self.predict(r["display"]) for r in records]
        exact = sum(p == r["spoken"] for p, r in zip(predicted, records)) / len(records)
        latin_free = sum(
            not any(c.isascii() and c.isalpha() for c in p) for p in predicted
        ) / len(records)
        total = sum(len(r["terms"]) for r in records)
        covered = sum(
            sum(t["surface"] in self.terms for t in r["terms"]) for r in records
        ) / max(1, total)
        return {"exact_rate": round(exact, 4), "latin_free_rate": round(latin_free, 4),
                "term_coverage": round(covered, 4)}


def dataset_integrity():
    """Structural checks: schema, splits, leakage, mark coverage, manifest alignment."""
    checks = {}
    tk = {s: rows("tashkeel", s) for s in SPLITS}
    cs = {s: rows("codeswitch", s) for s in SPLITS}
    mo = {s: rows("motion_plans", s) for s in SPLITS}
    all_tk = [r for split in tk.values() for r in split]
    checks["tashkeel_mark_classes"] = len({m for r in all_tk for m in mark_names(r["vowelized"])})
    checks["tashkeel_mean_vowelized_ratio"] = round(
        sum(vowelized_ratio(r["vowelized"]) for r in all_tk) / len(all_tk), 4
    )
    checks["tashkeel_plain_matches"] = all(
        strip_tashkeel(r["vowelized"]) == strip_tashkeel(r["plain"]) for r in all_tk
    )
    for name, splits in (("tashkeel", tk), ("codeswitch", cs), ("motion_plans", mo)):
        ids = [r["id"] for split in splits.values() for r in split]
        checks[f"{name}_unique_ids"] = len(ids) == len(set(ids))
        checks[f"{name}_counts"] = {k: len(v) for k, v in splits.items()}
        keys = "vowelized" if name == "tashkeel" else ("display" if name == "codeswitch" else None)
        if keys:
            texts = [r[keys] for split in splits.values() for r in split]
            checks[f"{name}_unique_texts"] = len(texts) == len(set(texts))
    splits = {s: scripts(s) for s in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        try:
            validate_splits(splits[a], splits[b])
            checks[f"scripts_no_leakage_{a}_{b}"] = True
        except ValueError:
            checks[f"scripts_no_leakage_{a}_{b}"] = False
    checks["script_counts"] = {k: len(v) for k, v in splits.items()}
    with (DATA / "voice" / "recording_prompts_v2.csv").open(encoding="utf-8") as stream:
        prompts = list(csv.DictReader(stream))
    manifest = read_jsonl(DATA / "voice" / "manifest.v2.template.jsonl")
    checks["voice_manifest_matches"] = [r["text"] for r in prompts] == [r["text"] for r in manifest]
    checks["voice_prompt_count"] = len(prompts)
    lex = load_lexicon()
    checks["lexicon_terms"] = len(lex)
    checks["lexicon_forbids_bare_identifiers"] = all(
        not lex.get(k, {}).get("allowed_in_narration", True)
        for k in ("if", "for", "while", "console")
    )
    annotations = read_jsonl(DATA / "motion" / "scene_annotations.jsonl")
    checks["motion_annotations"] = len(annotations)
    checks["motion_annotations_comfortable"] = all(
        not validate_plan({"features": {
            "camera_step": 0, "gesture_switches_per_second": 0.25,
            "jitter_amplitude": a["plan"]["jitter_amplitude"],
            "scene_seconds": a["plan"]["hold_seconds"],
            "flash_alpha": a["plan"]["flash_alpha"], "overlap_count": 0,
            "subtitle_words": 4, "easing": a["plan"]["easing"],
        }}) for a in annotations
    )
    manifest_data = json.loads((DATA / "build_manifest.json").read_text(encoding="utf-8"))
    counts = manifest_data["counts"]
    seed = read_jsonl(DATA / "scripts" / "seed_records.jsonl")
    seed_counts = {s: sum(r["split"] == s for r in seed) for s in SPLITS}
    checks["build_manifest_script_counts"] = all(
        counts["scripts_new"][s] + seed_counts[s] == checks["script_counts"][s]
        for s in SPLITS
    )
    checks["build_manifest_counts"] = all(
        counts[name][s] == checks[f"{name}_counts"][s]
        for name in ("tashkeel", "codeswitch", "motion_plans") for s in SPLITS
    )
    checks["asset_mirrors"] = all(
        (ROOT / "voice2reel" / "assets" / asset).read_bytes()
        == (DATA / folder / mirror).read_bytes()
        for asset, folder, mirror in (
            ("arabic_lexicon.json", "arabic", "lexicon.json"),
            ("motion_vocabulary.json", "motion", "vocabulary.json"),
        )
    )
    return checks


def measure(diacritizer, reader, classifier, split="validation"):
    """One measured pass on a split. Returns real metrics, never synthetic claims."""
    tk = rows("tashkeel", split)
    cs = rows("codeswitch", split)
    mo = rows("motion_plans", split)
    result = {
        "split": split,
        "tashkeel": diacritizer.metrics(tk),
        "codeswitch": reader.metrics(cs),
        "motion": {
            "classifier_accuracy": round(classifier.accuracy(mo), 4),
            "rule_comfort_rate": round(sum(
                not validate_plan(r["features"]) for r in mo
            ) / len(mo), 4),
            "mean_comfort_score": round(sum(
                comfort_score(r["features"]) for r in mo
            ) / len(mo), 4),
        },
    }
    result["score"] = round(
        0.30 * result["tashkeel"]["word_accuracy"]
        + 0.30 * result["codeswitch"]["exact_rate"]
        + 0.25 * result["motion"]["classifier_accuracy"]
        + 0.15 * result["motion"]["rule_comfort_rate"],
        4,
    )
    return result


def build_models(round_index, train_tashkeel, train_codeswitch, train_motion):
    """Progressive capability rounds; each round is a genuinely different fit."""
    budgets = [
        (240, 200, 240, 2),
        (480, 400, 480, 5),
        (len(train_tashkeel), len(train_codeswitch), len(train_motion), None),
    ]
    tk_n, cs_n, mo_n, features_n = budgets[min(round_index - 1, 2)]
    if round_index == 1:
        diacritizer = Diacritizer(use_affixes=False, use_defaults=False).fit(train_tashkeel[:tk_n])
        reader = CodeSwitchReader(use_normalization=False, use_fallback=True).fit(train_codeswitch[:cs_n])
    elif round_index == 2:
        diacritizer = Diacritizer(use_affixes=True, use_defaults=False).fit(train_tashkeel[:tk_n])
        reader = CodeSwitchReader(use_normalization=True, use_fallback=True).fit(train_codeswitch[:cs_n])
    else:
        diacritizer = Diacritizer(use_affixes=True, use_defaults=True).fit(train_tashkeel[:tk_n])
        reader = CodeSwitchReader(use_normalization=True, use_fallback=True).fit(train_codeswitch[:cs_n])
    from voice2reel.motion_plan import NUMERIC_FEATURES
    classifier = ComfortClassifier(
        features=NUMERIC_FEATURES[:features_n] if features_n else NUMERIC_FEATURES
    ).fit(train_motion[:mo_n])
    return diacritizer, reader, classifier


def run_rounds(max_rounds=5, target=0.90, output=None):
    train_tk = rows("tashkeel", "train")
    train_cs = rows("codeswitch", "train")
    train_mo = rows("motion_plans", "train")
    history = []
    for round_index in range(1, max_rounds + 1):
        models = build_models(round_index, train_tk, train_cs, train_mo)
        validation = measure(*models, split="validation")
        validation["round"] = round_index
        budgets = [(240, 200, 240), (480, 400, 480),
                   (len(train_tk), len(train_cs), len(train_mo))]
        used = budgets[min(round_index - 1, 2)]
        validation["train_rows"] = dict(zip(("tashkeel", "codeswitch", "motion"), used))
        history.append(validation)
        if validation["score"] >= target:
            break
    # The final test split is evaluated ONCE, with the best validation round only.
    best = max(history, key=lambda r: r["score"])
    models = build_models(best["round"], train_tk, train_cs, train_mo)
    report = {
        "target": target,
        "rounds": history,
        "best_round": best["round"],
        "final_test_on_best_round": measure(*models, split="test"),
        "integrity": dataset_integrity(),
        "note": (
            "Measured on authored synthetic splits. Front-end models are fitted on CPU; "
            "the Qwen writer is trained separately on Colab GPU. This score is NOT a claim "
            "about factual accuracy, voice naturalness or subjective video quality."
        ),
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/quality_v2/rounds.json")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()
    report = run_rounds(max_rounds=args.max_rounds, target=args.target, output=args.output)
    print(json.dumps({
        "rounds": len(report["rounds"]),
        "best_round": report["best_round"],
        "validation_scores": [r["score"] for r in report["rounds"]],
        "final_test_score": report["final_test_on_best_round"]["score"],
        "integrity_ok": all(
            value is True or not isinstance(value, bool)
            for value in report["integrity"].values()
        ),
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
