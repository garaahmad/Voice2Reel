import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from voice2reel.common import digest_file, write_json
from voice2reel.presenter import generate_script, run
from voice2reel.presenter_schema import PresenterScript
from voice2reel.presenter_render import PresenterRenderer, validate_audio


def sample():
    return PresenterScript(
        title="راجع الشرط",
        language="ar",
        scenes=[
            {
                "narration": "راجع الشرط قبل تشغيل البرنامج",
                "caption": f"الخطوة {i + 1}",
                "gesture": ["point", "think", "wave"][i],
                "camera": ["wide", "medium", "close"][i],
                "visual": {"kind": "flow", "items": ["تعلّم", "جرّب"], "code": ""},
            }
            for i in range(3)
        ],
    )


def test_compact_ai_output_compiles_into_typed_animation(tmp_path):
    class Writer:
        calls = 0

        def complete(self, messages, max_tokens):
            self.calls += 1
            return json.dumps(
                {
                    "title": "تعلم البرمجة",
                    "scenes": [
                        {
                            "narration": "ابدأ بمثال صغير ثم راجع النتيجة",
                            "caption": "خطوة مفيدة",
                            "labels": ["تعلّم", "طبّق"],
                        }
                        for _ in range(3)
                    ],
                }
            )

    writer = Writer()
    script = generate_script(writer, {"language": "ar", "idea": "التعلم"}, tmp_path / "trace.json")
    assert writer.calls == 1
    assert script.scenes[1].gesture == "point"
    assert script.scenes[1].visual.items == ["تعلّم", "طبّق"]
    assert script.scenes[1].visual.kind == "flow"


def test_failed_model_output_has_no_silent_fake_fallback(tmp_path):
    class Writer:
        def complete(self, *args, **kwargs):
            return '{"title":"Hello","scenes":[]}'

    with pytest.raises(ValueError, match="failed validation"):
        generate_script(Writer(), {"language": "ar"}, tmp_path / "trace.json")
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert len(trace) == 1 and "error" in trace[0]


def test_long_reel_or_external_render_instructions_rejected():
    data = sample().model_dump()
    data["scenes"][0]["narration"] = "كلمة " * 200
    with pytest.raises(ValueError):
        PresenterScript.model_validate(data)
    data = sample().model_dump()
    data["scenes"][0]["visual"]["shell"] = "rm something"
    with pytest.raises(ValueError):
        PresenterScript.model_validate(data)


@pytest.mark.parametrize("change", ["gap", "text", "language", "nan"])
def test_renderer_rejects_stale_or_invalid_speech(tmp_path, change):
    script = sample()
    path = tmp_path / "voice.wav"
    sf.write(path, np.ones(24000) * 0.02, 24000)
    meta = {
        "language": "ar",
        "duration": 1.0,
        "segments": [
            {"start": i / 3, "end": (i + 1) / 3, "text": s.narration} for i, s in enumerate(script.scenes)
        ],
    }
    if change == "gap":
        meta["segments"][1]["start"] += 0.1
    elif change == "text":
        meta["segments"][0]["text"] = "نص قديم"
    elif change == "language":
        meta["language"] = "en"
    else:
        meta["segments"][0]["end"] = float("nan")
    with pytest.raises(ValueError):
        validate_audio(script, meta, path)


@pytest.fixture
def project(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    import voice2reel.fast_voice as voice
    import voice2reel.neural_voice as neural
    import voice2reel.presenter_render as renderer

    monkeypatch.setitem(renderer.PROFILES, "fast", (270, 6, "ultrafast"))

    class FakeVoice:
        calls = 0

        def __init__(self, *args):
            pass

        def synthesize(self, script, output):
            FakeVoice.calls += 1
            rate = 24000
            duration = 1.5
            sf.write(output, np.sin(np.arange(36000) * 0.03) * 0.05, rate)
            metadata = {
                "language": script.language,
                "duration": duration,
                "segments": [
                    {"start": i * 0.5, "end": (i + 1) * 0.5, "text": s.narration}
                    for i, s in enumerate(script.scenes)
                ],
            }
            write_json(output.with_suffix(".json"), metadata)
            return metadata

    monkeypatch.setattr(voice, "FastVoice", FakeVoice)
    monkeypatch.setattr(neural, "NeuralVoice", FakeVoice)
    source = tmp_path / "input.json"
    write_json(source, sample().model_dump())
    return tmp_path / "project", source, FakeVoice


def test_voice_and_video_cached_and_visual_edits_do_not_repeat_speech(project):
    output, source, voice = project
    video, cold = run(output, script_path=source, device="cpu")
    assert not cold["cache_hits"]["voice"] and voice.calls == 1
    original = digest_file(video)
    _, warm = run(output, script_path=source, device="cpu")
    assert warm["cache_hits"]["voice"] and warm["cache_hits"]["video"]
    assert voice.calls == 1
    changed = json.loads(source.read_text())
    changed["scenes"][0]["caption"] = "عنوان جديد"
    write_json(source, changed)
    video, edited = run(output, script_path=source, device="cpu")
    assert edited["cache_hits"]["voice"] and not edited["cache_hits"]["video"]
    assert voice.calls == 1 and digest_file(video) != original
    changed["scenes"][0]["narration"] = "النص الجديد يحتاج صوتًا جديدًا"
    write_json(source, changed)
    _, edited = run(output, script_path=source, device="cpu")
    assert not edited["cache_hits"]["voice"] and voice.calls == 2


def test_corrupt_audio_is_regenerated_not_silently_reused(project):
    output, source, voice = project
    run(output, script_path=source, device="cpu")
    (output / "narration.wav").write_bytes(b"corrupt")
    _, metrics = run(output, script_path=source, device="cpu")
    assert not metrics["cache_hits"]["voice"] and voice.calls == 2


def test_failed_render_preserves_previous_video(project, monkeypatch):
    output, source, _ = project
    video, _ = run(output, script_path=source, device="cpu")
    before = digest_file(video)
    changed = json.loads(source.read_text())
    changed["scenes"][0]["caption"] = "تغيير"
    write_json(source, changed)

    def fail(*args):
        raise RuntimeError("test render failure")

    monkeypatch.setattr(PresenterRenderer, "frame", fail)
    with pytest.raises(RuntimeError, match="test render failure"):
        run(output, script_path=source, device="cpu")
    assert digest_file(video) == before
    assert json.loads((output / "progress.json").read_text())["stage"] == "failed"


def test_duplicate_job_rejected_without_waiting(project):
    fcntl = pytest.importorskip("fcntl")
    output, source, _ = project
    output.mkdir()
    with (output / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another job"):
            run(output, script_path=source)


def test_invalid_request_fails_before_model_loading(tmp_path, monkeypatch):
    import voice2reel.presenter as pipeline

    def fail(*args, **kwargs):
        pytest.fail("invalid input must not load weights")

    monkeypatch.setattr(pipeline, "load_writer", fail)
    with pytest.raises(ValueError):
        run(tmp_path, request={"idea": "   ", "duration": 999})
    assert json.loads((tmp_path / "progress.json").read_text())["stage"] == "failed"


def test_fast_voice_timings_follow_real_samples_and_reject_nan(tmp_path):
    from types import SimpleNamespace
    from voice2reel.fast_voice import FastVoice

    voice = FastVoice.__new__(FastVoice)
    voice.language = "ar"

    class Model:
        config = SimpleNamespace(sample_rate=16000)

        def synthesize(self, text):
            yield SimpleNamespace(audio_float_array=np.ones(1600, dtype=np.float32) * 0.02)

    voice.voice = Model()
    metadata = voice.synthesize(sample(), tmp_path / "voice.wav")
    assert metadata["duration"] == pytest.approx(0.3)
    assert metadata["segments"][1]["start"] == pytest.approx(0.1)
    voice.voice.synthesize = lambda text: iter([SimpleNamespace(audio_float_array=np.array([np.nan]))])
    with pytest.raises(ValueError, match="invalid audio"):
        voice.synthesize(sample(), tmp_path / "broken.wav")


def test_identical_idea_cache_skips_model_loading(project, monkeypatch):
    import voice2reel.presenter as pipeline

    output, _, _ = project
    calls = []

    class Writer:
        model_id, revision, backend = "test-model", "test-revision", "test"

        def complete(self, *args, **kwargs):
            return json.dumps(
                {
                    "title": "تعلّم البرمجة",
                    "scenes": [
                        {
                            "narration": "ابدأ بخطوة صغيرة ثم راجع ما تعلمته",
                            "caption": "خطوة واضحة",
                            "labels": ["تعلّم", "راجع"],
                        }
                        for _ in range(3)
                    ],
                }
            )

    def load(*args, **kwargs):
        calls.append(True)
        return Writer()

    monkeypatch.setattr(pipeline, "load_writer", load)
    request = {"idea": "تعلم البرمجة", "facts": "ابدأ بخطوة صغيرة", "language": "ar"}
    run(output, request=request, device="cpu")
    _, metrics = run(output, request=request, device="cpu")
    assert len(calls) == 1
    assert metrics["cache_hits"] == {"script": True, "voice": True, "mix": True, "video": True}


def test_arabic_code_identifiers_trigger_repair_with_previous_response(tmp_path):
    class Writer:
        calls = 0
        previous = None

        def complete(self, messages, max_tokens):
            self.calls += 1
            if self.calls == 2:
                assert messages[-2]["role"] == "assistant"
                assert messages[-2]["content"] == self.previous
            data = {
                "title": "راجع الشرط",
                "scenes": [
                    {
                        "narration": "راجع الشرط قبل تشغيل هذا البرنامج",
                        "caption": "راجع الشرط",
                        "labels": ["الشرط", "النتيجة"],
                    }
                    for _ in range(3)
                ],
            }
            if self.calls == 1:
                data["scenes"][0]["narration"] = "راجع شرط if قبل تشغيل هذا البرنامج"
            self.previous = json.dumps(data, ensure_ascii=False)
            return self.previous

    writer = Writer()
    script = generate_script(writer, {"language": "ar"}, tmp_path / "trace.json", attempts=2)
    assert writer.calls == 2
    assert "if" not in script.scenes[0].narration
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert "error" in trace[0] and "error" not in trace[1]
