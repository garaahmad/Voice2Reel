"""Arabic text front-end: diacritics, mixed-script narration and TTS-safe readings.

The engines in this repository receive plain text; they do not learn pronunciation
from a dataset at inference time. This module is the deterministic layer that turns
*display* text (possibly fully vowelized and mixed Arabic/English) into the text
that is actually sent to a speech engine, using an authored pronunciation lexicon.
"""

import json
import re
import unicodedata
from pathlib import Path

# Combining marks, not letters. Precomposed hamza letters (أ إ آ ؤ ئ ء) are letters
# and are deliberately preserved: removing them would corrupt the word.
TASHKEEL = (
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"  # tanween, harakat, shadda, sukun
    "\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f"
    "\u0670\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06dd\u06de\u06df"
    "\u06e0\u06e1\u06e2\u06e3\u06e4\u06e5\u06e6\u06e7\u06e8\u06e9\u06ea\u06eb\u06ec\u06ed"
)
TATWEEL = "\u0640"
MARK_NAMES = {
    "\u064b": "tanween_fath", "\u064c": "tanween_damm", "\u064d": "tanween_kasr",
    "\u064e": "fatha", "\u064f": "damma", "\u0650": "kasra",
    "\u0651": "shadda", "\u0652": "sukun", "\u0670": "superscript_alef",
}
ALL_MARKS = tuple(sorted(MARK_NAMES))
ARABIC_LETTER = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u066f\u0671-\u06d3\u06fa-\u06fc]")
LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/-]*")
# Bundled with the package so an installed wheel works without the repository tree.
LEXICON_PATH = Path(__file__).parent / "assets" / "arabic_lexicon.json"
# Latin terms are bounded by Latin characters/digits only; Arabic letters and the
# tatweel in "الـ" must NOT block a match.
_LATIN_EDGE = r"A-Za-z0-9_.+#/-"


def strip_tashkeel(text):
    """Remove combining marks and tatweel while keeping base letters intact."""
    return "".join(c for c in unicodedata.normalize("NFC", text) if c not in TASHKEEL + TATWEEL)


def mark_names(text):
    """Names of the distinct diacritic classes present in ``text``."""
    return sorted({MARK_NAMES[c] for c in text if c in MARK_NAMES})


def letter_mark_pairs(text):
    """Return (letter, following marks) pairs; marks with no letter are ignored."""
    pairs, pending = [], ""
    for char in unicodedata.normalize("NFC", text):
        if char in TASHKEEL:
            pending += char
        elif ARABIC_LETTER.match(char):
            pairs.append((char, pending))
            pending = ""
        elif char.isspace() or char in ".,;:!؟?،؛()-\"'" or char == TATWEEL:
            pending = ""
        else:
            # A non-Arabic token inside the string; keep it out of the alignment.
            if pending:
                pairs.append((char, pending))
                pending = ""
    return pairs


def vowelized_ratio(text):
    """Fraction of Arabic letters carrying at least one diacritic."""
    pairs = letter_mark_pairs(text)
    if not pairs:
        return 0.0
    return sum(bool(marks) for _, marks in pairs) / len(pairs)


def tashkeel_accuracy(reference, hypothesis):
    """Exact per-letter diacritic agreement; base letters must align exactly."""
    ref, hyp = letter_mark_pairs(reference), letter_mark_pairs(hypothesis)
    if not ref or [letter for letter, _ in ref] != [letter for letter, _ in hyp]:
        return 0.0
    return sum(a[1] == b[1] for a, b in zip(ref, hyp)) / len(ref)


def latin_spans(text):
    return [m.group() for m in LATIN_TOKEN.finditer(text)]


def load_lexicon(path=None):
    path = Path(path) if path else LEXICON_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    terms = data["terms"] if isinstance(data, dict) else data
    if not isinstance(terms, dict) or not terms:
        raise ValueError("Pronunciation lexicon must be a non-empty object")
    return terms


def lexicon_entries(lexicon):
    """Only entries allowed in spoken narration, longest surface form first."""
    allowed = {
        key: value["spoken"]
        for key, value in lexicon.items()
        if isinstance(value, dict) and value.get("allowed_in_narration", True) and value.get("spoken")
    }
    return sorted(allowed.items(), key=lambda item: (-len(item[0]), item[0]))


def spoken_form(text, lexicon=None):
    """Render technical Latin terms as Arabic readings for the speech engine.

    Returns (spoken_text, unresolved_terms). Bare code identifiers such as ``if``
    are marked ``allowed_in_narration: false`` in the lexicon and stay unresolved,
    so narration containing literal source code is still rejected.
    """
    lexicon = lexicon if lexicon is not None else load_lexicon()
    result = text
    for surface, reading in lexicon_entries(lexicon):
        pattern = re.compile(
            r"(?<![" + _LATIN_EDGE + r"])" + re.escape(surface) + r"(?![" + _LATIN_EDGE + r"])",
            re.IGNORECASE,
        )
        result = pattern.sub(reading, result)
    # "الـ for loop" -> "الحلقة فور": the Arabic definite article must stay attached
    # to the reading, otherwise the speech engine reads a detached tatweel.
    result = result.replace("الـ ", "ال").replace("الـ", "ال")
    unresolved = sorted(set(latin_spans(result)))
    return result, unresolved


def prepare_narration(text, lexicon=None):
    """Validate mixed narration and return the TTS-safe reading."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Narration must not be empty")
    spoken, unresolved = spoken_form(text, lexicon)
    if unresolved:
        raise ValueError("Unregistered Latin terms in narration: " + ", ".join(unresolved))
    return spoken


def _field(scene, name):
    if isinstance(scene, dict):
        return scene.get(name, "") or ""
    return getattr(scene, name, "") or ""


def speech_text(scene):
    """Text a speech engine must read for a scene (display text may differ)."""
    return _field(scene, "spoken") or _field(scene, "narration")


def display_text(scene):
    return _field(scene, "narration")


def resolve_scene_texts(value, lexicon=None):
    """Fill ``spoken`` for compact AI output; reject unregistered code identifiers."""
    lexicon = lexicon if lexicon is not None else load_lexicon()
    for scene in value.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        narration = scene.get("narration")
        if not isinstance(narration, str):
            continue
        if latin_spans(narration):
            scene["spoken"] = prepare_narration(narration, lexicon)
    return value


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", help="Display narration to convert into speech-engine text")
    p.add_argument("--lexicon")
    args = p.parse_args()
    lexicon = load_lexicon(args.lexicon)
    spoken, unresolved = spoken_form(args.text, lexicon)
    print(json.dumps(
        {"display": args.text, "spoken": spoken, "unresolved": unresolved,
         "vowelized_ratio": round(vowelized_ratio(args.text), 4),
         "marks": mark_names(args.text)},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
