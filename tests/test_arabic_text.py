import pytest

from voice2reel.arabic_text import (
    display_text, latin_spans, letter_mark_pairs, load_lexicon, mark_names,
    prepare_narration, resolve_scene_texts, spoken_form, speech_text,
    strip_tashkeel, tashkeel_accuracy, vowelized_ratio,
)


def test_full_tashkeel_metrics_and_normalization():
    text = "مَرْحَبًا بِالْعَالَمِ"
    assert "fatha" in mark_names(text)
    assert "sukun" in mark_names(text)
    assert "shadda" in mark_names("الشَّمْسُ")
    assert vowelized_ratio(text) > 0.5
    assert strip_tashkeel(text) == "مرحبا بالعالم"
    assert tashkeel_accuracy(text, text) == 1.0
    assert tashkeel_accuracy(text, strip_tashkeel(text)) < 0.5
    assert letter_mark_pairs("كَتَبَ")[0][0] == "ك"


def test_registered_technical_terms_become_arabic_speech():
    lex = load_lexicon()
    display = "مَرْحَبًا اليوم راح نشرح الـ for loop"
    spoken, unresolved = spoken_form(display, lex)
    assert unresolved == []
    assert "for" not in spoken and "loop" not in spoken
    assert "حلقة فور" in spoken
    assert spoken.startswith("مَرْحَبًا")
    assert prepare_narration(display, lex) == spoken
    assert "C++" in latin_spans("نستعمل C++ هنا")


def test_bare_code_identifiers_are_still_rejected():
    with pytest.raises(ValueError, match="Unregistered Latin terms"):
        prepare_narration("راجع شرط if قبل التشغيل")


def test_scene_text_resolution_and_speech_selection():
    value = {"scenes": [
        {"narration": "نشرح الـ for loop اليوم", "caption": "حلقة"},
        {"narration": "نص عربي بلا مصطلحات", "caption": "عادي"},
    ]}
    resolved = resolve_scene_texts(value)
    assert "for" not in resolved["scenes"][0]["spoken"]
    assert "حلقة فور" in resolved["scenes"][0]["spoken"]
    assert "spoken" not in resolved["scenes"][1]
    assert speech_text(resolved["scenes"][0]) == resolved["scenes"][0]["spoken"]
    assert speech_text(resolved["scenes"][1]) == resolved["scenes"][1]["narration"]
    assert display_text(resolved["scenes"][0]) == value["scenes"][0]["narration"]
