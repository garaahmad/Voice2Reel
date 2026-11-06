import json

import numpy as np
import pytest
import soundfile as sf

from voice2reel.voice.prepare import load_audio, prepare


def fixture_manifest(root, count=16):
    rows = []
    for i in range(count):
        rate = 24000
        audio = 0.15 * np.sin(2 * np.pi * (160 + 27 * i) * np.arange(rate * 3) / rate)
        filename = f"{i}.wav"
        sf.write(root / filename, audio, rate)
        rows.append({"audio": filename, "text": f"تسجيل للاختبار رقم {i}", "speaker": "test"})
    path = root / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path, rows


def test_preparation_and_split_reproducibility(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    a = prepare(manifest, tmp_path, tmp_path / "prepared_a")
    b = prepare(manifest, tmp_path, tmp_path / "prepared_b")
    assert a == b
    assert a["counts"] == {"train": 12, "validation": 2, "test": 2}
    hashes = [r["sha256"] for split in a["splits"].values() for r in split]
    assert len(hashes) == len(set(hashes))
    for split in a["splits"].values():
        for row in split:
            info = sf.info(tmp_path / "prepared_a" / row["audio_file"])
            assert info.samplerate == 22050 and info.channels == 1
    with pytest.raises(ValueError, match="exists"):
        prepare(manifest, tmp_path, tmp_path / "prepared_a")


@pytest.mark.parametrize("bad", ["duplicate_audio", "duplicate_text", "traversal", "speaker"])
def test_invalid_dataset_leaves_no_partial_output(tmp_path, bad):
    manifest, rows = fixture_manifest(tmp_path)
    if bad == "duplicate_audio":
        (tmp_path / "1.wav").write_bytes((tmp_path / "0.wav").read_bytes())
    elif bad == "duplicate_text":
        rows[1]["text"] = rows[0]["text"]
    elif bad == "speaker":
        rows[1]["speaker"] = "different"
    else:
        rows[0]["audio"] = "../outside.wav"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        prepare(manifest, tmp_path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("audio", [np.zeros(48000), np.ones(48000), np.zeros(200)])
def test_bad_audio_rejected(tmp_path, audio):
    path = tmp_path / "bad.wav"
    sf.write(path, audio, 16000)
    with pytest.raises(ValueError):
        load_audio(path)


def test_upstream_config_contract(tmp_path):
    pytest.importorskip("TTS")
    from voice2reel.voice.train import build_config
    from trainer import TrainerArgs

    config = build_config(tmp_path, tmp_path, tmp_path)
    config.check_values()
    assert config.model_args.gpt_num_audio_tokens == 1026
    assert config.audio.output_sample_rate == 24000
    assert TrainerArgs(continue_path="run", grad_accum_steps=16).continue_path == "run"
