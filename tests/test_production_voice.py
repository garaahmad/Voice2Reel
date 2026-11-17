"""Offline tests for the Arabic speech-only pipeline."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice2reel.production_contract import ProductionPlan
from voice2reel.production_voice import (
    VOICE_OPTIONS, prepare_text, read_override_file, read_terms_file,
    replace_terms, script_for_plan, synthesize,
)


def sample_plan():
    return ProductionPlan.model_validate({
        "version": 1, "title": "الإنترنت", "fps": 30,
        "width": 1080, "height": 1920,
        "shots": [
            {"caption": "مقدمة", "narration": "أهلا فيكم", "seconds": 3.0,
             "gesture": "idle", "camera": "medium", "transition": "fade"},
            {"caption": "تطبيق", "narration": "جرب Flutter وsetState", "seconds": 5.0,
             "gesture": "explain", "camera": "medium", "transition": "cut"},
        ]
    })


def test_arabic_preparation_is_conservative():
    assert prepare_text("  أهلًا   وسهلًا ، كيف الحال ؟ ") == "أهلًا وسهلًا، كيف الحال؟"
    assert prepare_text("Flutter") == "Flutter."
    with pytest.raises(ValueError):
        prepare_text("   ")


def test_pronunciation_substitutions_only_match_full_words():
    text = replace_terms("Flutter وsetState وليس FlutterExtra", {
        "Flutter": "فلاتر", "setState": "سِت ستيت",
    })
    assert text == "فلاتر وسِت ستيت وليس FlutterExtra"


def test_overrides_do_not_mutate_visual_plan(tmp_path):
    plan = sample_plan()
    original = plan.model_dump_json()
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps({"shots": {"2": "نستخدم سِت ستيت هنا"}},
                   ensure_ascii=False), encoding="utf-8"
    )
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"Flutter": "فلاتر"}, ensure_ascii=False),
                     encoding="utf-8")
    script = script_for_plan(
        plan, read_override_file(overrides), read_terms_file(words)
    )
    assert script == "أهلا فيكم. نستخدم سِت ستيت هنا."
    assert plan.model_dump_json() == original


def test_override_rejects_unknown_shot(tmp_path):
    file = tmp_path / "voice.json"
    file.write_text('{"shots":{"3":"wrong index"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonexistent"):
        script_for_plan(sample_plan(), read_override_file(file))


def test_voice_audition_list():
    assert "ar-JO-TaimNeural" in VOICE_OPTIONS["ar"]
    assert "ar-EG-ShakirNeural" in VOICE_OPTIONS["ar"]
    assert "ar-SA-ZariyahNeural" in VOICE_OPTIONS["ar"]


def test_synthesis_writes_mp3_and_script_without_network(tmp_path, monkeypatch):
    captured = {}

    class FakeCommunicate:
        def __init__(self, text, voice, **options):
            captured.update(text=text, voice=voice, options=options)

        async def save(self, filename):
            Path(filename).write_bytes(b"fake mp3")

    monkeypatch.setitem(sys.modules, "edge_tts",
                        SimpleNamespace(Communicate=FakeCommunicate))
    path = tmp_path / "voice.mp3"
    plan = sample_plan()
    original = plan.model_dump_json()
    result = asyncio.run(synthesize(plan, path, language="ar",
                                    voice="ar-JO-TaimNeural",
                                    rate="+0%", pitch="+0Hz"))
    assert result == path
    assert path.read_bytes() == b"fake mp3"
    assert path.with_suffix(".speech.txt").read_text(encoding="utf-8") == captured["text"]
    assert captured["options"]["pitch"] == "+0Hz"
    assert captured["voice"] == "ar-JO-TaimNeural"
    assert plan.model_dump_json() == original
