import json
from types import SimpleNamespace

import numpy as np
import pytest

from voice2reel.voice.model import DEFAULT_SPEAKER
from voice2reel.voice.synthesize import read_source, synthesize_segments
from voice2reel.scriptwriter.schema import messages_for, parse_script


class FakeVoiceModel:
    """Check dispatch/timing without downloading model weights in CI."""

    def __init__(self):
        self.speaker_manager = SimpleNamespace(
            speakers={DEFAULT_SPEAKER: {"gpt_cond_latent": "preset_cond", "speaker_embedding": "preset_emb"}}
        )
        self.calls = []
        self.references = []

    def get_conditioning_latents(self, audio_path):
        self.references.append(audio_path)
        return "clone_cond", "clone_emb"

    def inference(self, text, language, conditioning, embedding, **kwargs):
        self.calls.append((text, language, conditioning, embedding))
        return {"wav": np.full(2400, 0.1, dtype=np.float32)}


@pytest.mark.parametrize("language,text", [("ar", "مرحبا بالعالم"), ("en", "Hello world")])
def test_preset_voice_needs_no_recording_and_forwards_language(language, text):
    model = FakeVoiceModel()
    audio, timing = synthesize_segments(model, [text, text], language=language)
    assert not model.references
    assert model.calls[0] == (text, language, "preset_cond", "preset_emb")
    assert len(audio) == 4800
    assert timing[0]["start"] == 0 and timing[0]["end"] == 0.1
    assert timing[1]["start"] == 0.1 and timing[1]["end"] == 0.2


def test_clone_mode_remains_available():
    model = FakeVoiceModel()
    synthesize_segments(model, ["Hello"], "reference.wav", language="en")
    assert model.references == [["reference.wav"]]
    assert model.calls[0][2:] == ("clone_cond", "clone_emb")


@pytest.mark.parametrize(
    "kwargs",
    [{"language": "fr"}, {"speaker": "missing"}, {"reference": "sample.wav", "speaker": DEFAULT_SPEAKER}],
)
def test_bad_voice_options_fail_before_inference(kwargs):
    model = FakeVoiceModel()
    with pytest.raises(ValueError):
        synthesize_segments(model, ["Hello"], **kwargs)
    assert not model.calls


@pytest.mark.parametrize("texts", [[], [" "], ["x" * 5001]])
def test_bad_text_is_rejected(texts):
    with pytest.raises(ValueError):
        synthesize_segments(FakeVoiceModel(), texts)


def english_script():
    return {
        "schema_version": 1,
        "title": "Start small",
        "language": "en",
        "target_duration_seconds": 20,
        "beats": [
            {
                "narration": "Choose one useful task to finish today before adding another.",
                "caption": "Choose a task",
                "pose": "IDLE",
                "camera": "close",
            },
            {
                "narration": "Break it into smaller steps that you can complete and check.",
                "caption": "Small steps",
                "pose": "TYPE_A",
                "camera": "medium",
            },
            {
                "narration": "Review the result and use what you learned to plan tomorrow.",
                "caption": "Review",
                "pose": "CELEB",
                "camera": "far",
            },
        ],
    }


def test_script_language_inferred_and_mismatch_rejected(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps(english_script()))
    texts, language = read_source(script_path=path)
    assert len(texts) == 3 and language == "en"
    with pytest.raises(ValueError, match="differs"):
        read_source(script_path=path, language="ar")
    assert read_source(content="Hello", language="en") == (["Hello"], "en")
    assert read_source(content="مرحبا")[1] == "ar"


def test_english_generation_prompt_and_language_validation():
    assert "in English" in messages_for("Study", language="en")[0]["content"]
    with pytest.raises(ValueError, match="language"):
        parse_script(json.dumps(english_script()), expected_language="ar")
    assert parse_script(json.dumps(english_script()), expected_language="en").language == "en"


def test_download_profiles(monkeypatch, tmp_path):
    import huggingface_hub
    from voice2reel.voice.model import download_model

    calls = []

    def fake_download(*args, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    download_model(tmp_path, training=False)
    assert "speakers_xtts.pth" in calls[-1]["allow_patterns"]
    assert "dvae.pth" not in calls[-1]["allow_patterns"]
    download_model(tmp_path)
    assert "dvae.pth" in calls[-1]["allow_patterns"]


def test_model_loader_handles_upstream_eval_returning_none(monkeypatch, tmp_path):
    from TTS.tts.configs.xtts_config import XttsConfig
    import TTS.tts.models.xtts as upstream
    from voice2reel.voice.model import load_model

    class Model:
        def __init__(self, config):
            self.config = config

        def load_checkpoint(self, *args, **kwargs):
            self.loaded = True

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return None

    monkeypatch.setattr(upstream, "Xtts", Model)
    monkeypatch.setattr(XttsConfig, "load_json", lambda self, path: None)
    model = load_model(tmp_path, device="cpu")
    assert model.loaded and model.evaluated and model.device == "cpu"
