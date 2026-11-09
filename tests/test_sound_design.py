import json

import numpy as np
import pytest
import soundfile as sf

from voice2reel.audio_mix import effects, mix_audio, score
from voice2reel.common import digest_file
from voice2reel.neural_voice import NeuralVoice, validate_settings
from test_presenter import project, run, sample  # noqa: F401, F811


def test_online_voice_preserves_provider_timings_and_does_not_replace_audio_on_failure(tmp_path, monkeypatch):
    voice = NeuralVoice("ar")

    async def scenes(script, work):
        return [
            (np.ones(2400, dtype=np.float32) * 0.1, [{"text": "راجع", "start": 0.01, "end": 0.08}])
            for _ in script.scenes
        ]

    monkeypatch.setattr(voice, "_scenes", scenes)
    output = tmp_path / "narration.wav"
    meta = voice.synthesize(sample(), output)
    assert meta["segments"][1]["start"] == pytest.approx(0.28)
    assert meta["segments"][1]["words"][0]["start"] == pytest.approx(0.29)
    assert meta["duration"] == pytest.approx(0.72)
    before = digest_file(output)

    async def fail(*args):
        raise ConnectionError("service unavailable")

    monkeypatch.setattr(voice, "_scenes", fail)
    with pytest.raises(RuntimeError, match="No voice was substituted"):
        voice.synthesize(sample(), output)
    assert digest_file(output) == before


@pytest.mark.parametrize("voice,rate", [("en-US-GuyNeural", 0), ("ar-invalid", 0), (None, 80)])
def test_voice_configuration_cannot_mismatch_language(voice, rate):
    with pytest.raises(ValueError):
        validate_settings("ar", voice, rate)


def test_original_music_and_effects_have_actual_energy_and_repeat_deterministically():
    bed = score(3)
    fx = effects(3, [{"start": 0}, {"start": 1}, {"start": 2}])
    assert np.max(np.abs(bed)) > 0.03
    assert np.max(np.abs(fx)) > 0.01
    assert np.isfinite(bed).all() and np.isfinite(fx).all()
    np.testing.assert_array_equal(bed, score(3))
    assert np.max(np.abs(fx[23000:27000])) > 0.01


def test_mix_tracks_duck_beneath_speech_and_can_be_disabled(tmp_path):
    rate = 24000
    voice = np.sin(np.arange(rate * 3) * (2 * np.pi * 160 / rate)) * 0.15
    path = tmp_path / "voice.wav"
    sf.write(path, voice, rate)
    metadata = {"duration": 3.0, "segments": [{"start": 0.0}, {"start": 1.0}, {"start": 2.0}]}
    output = tmp_path / "soundtrack.wav"
    report = mix_audio(path, metadata, output)
    music, _ = sf.read(tmp_path / "music.wav")
    fx, _ = sf.read(tmp_path / "effects.wav")
    mixed, _ = sf.read(output)
    assert report["music_ducking"]
    assert 0 < report["music_rms"] < 0.014
    assert np.any(music) and np.any(fx)
    assert len(mixed) == rate * 3 and np.max(np.abs(mixed)) <= 0.941
    mix_audio(path, metadata, output, music=False, sfx=False)
    music, _ = sf.read(tmp_path / "music.wav")
    fx, _ = sf.read(tmp_path / "effects.wav")
    assert not np.any(music) and not np.any(fx)


def test_changing_music_reuses_the_narration_and_rebuilds_video(project):  # noqa: F811
    output, source, voice = project
    _, cold = run(output, script_path=source, device="cpu")
    _, edited = run(output, script_path=source, device="cpu", music=False)
    assert voice.calls == 1
    assert edited["cache_hits"]["voice"]
    assert not edited["cache_hits"]["mix"] and not edited["cache_hits"]["video"]
    assert json.loads((output / "soundtrack.json").read_text())["music_enabled"] is False


def test_missing_custom_music_fails_before_voice_generation(tmp_path):
    with pytest.raises(ValueError, match="existing file"):
        run(tmp_path / "project", script_path=tmp_path / "script.json", music_path=tmp_path / "missing.mp3")
