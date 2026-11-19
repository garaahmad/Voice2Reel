"""Generate reproducible text -> validated CharacterSpec JSON instruction data.

These are synthetic style-control examples, not learned image-quality labels.
The renderer keeps anatomy, limb connections, outlines and motion fixed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from voice2reel.character_spec import (
    ACCENTS, BROWS, CHEEKS, EYES, EXPRESSIONS, PALETTE_IDS,
    SYSTEM_PROMPT, CharacterSpec,
)

PAL_DESC = {
    "charcoal": ("charcoal black", "dark neutral charcoal", "فحمي داكن"),
    "navy": ("navy blue", "deep ocean blue", "كحلي"),
    "plum": ("plum purple", "deep purple plum", "بنفسجي"),
    "forest": ("forest green", "deep green", "أخضر غامق"),
    "sunset": ("warm sunset brown", "sunset ink", "بني دافئ"),
}
EXP_DESC = {
    "happy": ("happy", "smiling", "سعيد"),
    "surprised": ("surprised", "astonished", "متفاجئ"),
    "curious": ("curious", "interested", "فضولي"),
    "thoughtful": ("thoughtful", "reflective", "مفكر"),
    "confident": ("confident", "self assured", "واثق"),
}
EYE_DESC = {
    "friendly": ("friendly eyes", "normal bright eyes", "عيون لطيفة"),
    "wide": ("wide open eyes", "big lively eyes", "عيون واسعة"),
    "relaxed": ("relaxed eyes", "gentle relaxed eyes", "عيون هادئة"),
}
BROW_DESC = {
    "soft": ("soft eyebrows", "gentle brows", "حواجب ناعمة"),
    "raised": ("raised eyebrows", "lifted brows", "حواجب مرفوعة"),
    "focused": ("focused eyebrows", "determined brows", "حواجب مركزة"),
}
CHEEK_DESC = {
    "none": ("no cheek blush", "clean cheeks", "بدون خدود ملونة"),
    "subtle": ("subtle warm cheek blush", "soft blush", "خدود خفيفة"),
}
ACCENT_DESC = {
    "coral": ("a colorful neck accent", "small warm color accent", "لمسة لون"),
    "simple": ("a muted neck accent", "subtle neutral detail", "لمسة بسيطة"),
    "none": ("no neck accent", "no extra color details", "بدون زينة"),
}

def example(spec: CharacterSpec, language_index: int, phrasing: int) -> dict:
    pick = lambda table, key: table[key][language_index]
    palette = pick(PAL_DESC, spec.palette)
    expression = pick(EXP_DESC, spec.expression)
    eyes = pick(EYE_DESC, spec.eyes)
    brows = pick(BROW_DESC, spec.brows)
    cheeks = pick(CHEEK_DESC, spec.cheeks)
    accent = pick(ACCENT_DESC, spec.accent)
    if language_index == 2:
        templates = [
            f"صمم شخصية ستيكمان {expression} بلون {palette}، مع {eyes}، {brows}، {cheeks} و{accent}. بدون قبعة.",
            f"بدي شخصية كرتونية {palette}، ملامحها {expression}، فيها {eyes} و{brows}، و{cheeks}، {accent}.",
        ]
    else:
        templates = [
            f"Design an original {palette} mascot with a {expression} expression, {eyes}, {brows}, {cheeks}, and {accent}; no hat.",
            f"A clean explainer stickman in {palette}, looking {expression}. Use {eyes}, {brows}, {cheeks}, and {accent}.",
        ]
    prompt = templates[phrasing % 2]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": spec.model_dump_json()},
        ]
    }

def create_rows(count: int, seed: int = 42) -> list[dict]:
    if not 100 <= count <= 15000:
        raise ValueError("count must be 100..15000")
    rng = random.Random(seed)
    combinations = list(itertools.product(
        PALETTE_IDS, EXPRESSIONS, EYES, BROWS, CHEEKS, ACCENTS
    ))
    rng.shuffle(combinations)
    all_rows = []
    seen = set()
    # 2 English + 1 Arabic style for each combination / synonym variant.
    for variant in range(4):
        for values in combinations:
            for language_index in (0, 1, 2):
                style = CharacterSpec(
                    version=1, palette=values[0], expression=values[1],
                    eyes=values[2], brows=values[3], cheeks=values[4],
                    accent=values[5],
                )
                row = example(style, language_index, variant)
                utterance = row["messages"][1]["content"]
                if utterance in seen:
                    continue
                seen.add(utterance)
                all_rows.append(row)
    rng.shuffle(all_rows)
    if count > len(all_rows):
        raise ValueError(f"Only {len(all_rows)} unique synthetic prompts available")
    selected = all_rows[:count]
    for index, row in enumerate(selected):
        row["id"] = f"character-spec-{seed}-{index:05d}"
    return selected

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rows = create_rows(args.count, args.seed)
    cutoff = round(len(rows) * .9)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, batch in (("train", rows[:cutoff]),
                         ("validation", rows[cutoff:])):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in batch:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ CharacterSpec examples: train={cutoff}, validation={len(rows)-cutoff}")
    print(f"✅ Output: {output}")

if __name__ == "__main__":
    main()
