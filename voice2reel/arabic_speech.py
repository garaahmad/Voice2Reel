"""Conservative, reviewable Arabic pronunciation hints for neural TTS.

This is NOT automatic full Arabic diacritization. Arabic grammar/context
cannot be guessed safely; only explicitly documented technical terms get
phonetic hints, and speaker overrides are always optional.
"""
from __future__ import annotations

import re

# Keep diacritics inside the word, not arbitrary case endings (i'raab).
# These are pronunciation *hints* and may vary by speaker/accent.
TECHNICAL_PRONUNCIATIONS = {
    "الراوتر": "الرَّاوْتَر",
    "راوتر": "رَاوْتَر",
    "الإنترنت": "الإِنْتَرْنِت",
    "الواي فاي": "الواي فاي",
    "الواي فايّ": "الواي فاي",
    "Flutter": "فْلاتَر",
    "Firebase": "فايَرْبَيْس",
    "setState": "سِتْ سْتَيْت",
    "Widget": "وِدْجِت",
    "Widgets": "وِدْجِتْس",
    "API": "إيه بي آي",
    "Qwen": "كْوِين",
    "Wi-Fi": "واي فاي",
}


def replace_whole_terms(text: str, terms: dict[str, str]) -> str:
    """Match standalone Arabic/Latin terms, including Arabic conjunction 'و'."""
    for key, replacement in sorted(terms.items(), key=lambda item: -len(item[0])):
        if not key or not replacement:
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            # Arabic clitics like وFlutter may attach directly to an English term.
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])"
            text = re.sub(pattern, lambda m: replacement, text, flags=re.IGNORECASE)
        else:
            # Handle و/ب/ف before a complete Arabic term without editing
            # the interior of other words.
            pattern = r"(?<!\w)([وفب]?)(?:" + re.escape(key) + r")(?!\w)"
            text = re.sub(
                pattern, lambda m: m.group(1) + replacement,
                text, flags=re.IGNORECASE
            )
    return text


def pronounce(text: str, mode: str = "technical",
              overrides: dict[str, str] | None = None) -> str:
    """Return speech-only text with optional, controlled phonetic hints."""
    if mode not in ("off", "technical"):
        raise ValueError("Arabic pronunciation mode must be off or technical")
    overrides = overrides or {}
    # User overrides take precedence over the built-in glossary.
    result = replace_whole_terms(text, overrides)
    if mode == "technical":
        mapping = {key: value for key, value in TECHNICAL_PRONUNCIATIONS.items()
                   if key.casefold() not in {k.casefold() for k in overrides}}
        result = replace_whole_terms(result, mapping)
    return result
