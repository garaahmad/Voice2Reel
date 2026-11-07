import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice2reel.common import read_jsonl
from voice2reel.render import render, validate_timeline
from voice2reel.scriptwriter.schema import Script


def fixture():
    root = Path(__file__).resolve().parents[1]
    script = Script.model_validate(read_jsonl(root / "datasets/scripts/train.jsonl")[0]["script"])
    metadata = {
        "language": "ar",
        "duration": 1.5,
        "segments": [
            {"start": i * 0.5, "end": (i + 1) * 0.5, "text": b.narration} for i, b in enumerate(script.beats)
        ],
    }
    return script, metadata


@pytest.mark.parametrize("mutation", ["gap", "text", "language", "duration", "nan"])
def test_mismatched_narration_rejected(mutation):
    script, metadata = fixture()
    bad = copy.deepcopy(metadata)
    if mutation == "gap":
        bad["segments"][1]["start"] += 0.1
    elif mutation == "text":
        bad["segments"][0]["text"] = "unrelated"
    elif mutation == "nan":
        bad["segments"][0]["end"] = float("nan")
    else:
        bad[mutation] = "en" if mutation == "language" else 99
    with pytest.raises(ValueError):
        validate_timeline(script, bad)


def test_real_video_encode_and_audio_mux(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg not installed")
    script, metadata = fixture()
    source = tmp_path / "script.json"
    source.write_text(script.model_dump_json())
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps(metadata))
    wav = tmp_path / "voice.wav"
    sf.write(wav, 0.05 * np.sin(np.arange(36000) * 0.03), 24000)
    output = render(source, wav, timing, tmp_path / "video.mp4", width=360, fps=4)
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)]
        )
    )
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"], video["codec_name"]) == (360, 640, "h264")
    assert abs(float(info["format"]["duration"]) - 1.5) < 0.1
