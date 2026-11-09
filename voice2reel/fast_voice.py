"""Local Piper preset speech; the slower XTTS cloning path remains available."""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voice2reel.arabic_text import speech_text
from voice2reel.common import write_json

REPOSITORY = "rhasspy/piper-voices"
REVISION = "e7229f7d33a10e145e109b886d661984201ac3c4"
VOICES = {
    "ar": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
}


class FastVoice:
    def __init__(self, language, directory="models/piper", threads=2):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import PiperConfig

        if language not in VOICES or not 1 <= threads <= 32:
            raise ValueError("Choose ar/en and 1–32 CPU threads")
        stem = VOICES[language]
        files = [
            hf_hub_download(REPOSITORY, stem + extension, revision=REVISION, local_dir=directory)
            for extension in [".onnx", ".onnx.json"]
        ]
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.voice = PiperVoice(
            session=ort.InferenceSession(files[0], sess_options=options, providers=["CPUExecutionProvider"]),
            config=PiperConfig.from_dict(json.loads(Path(files[1]).read_text())),
        )
        self.language = language

    def synthesize(self, script, output):
        started = time.perf_counter()
        arrays, segments, cursor = [], [], 0
        rate = self.voice.config.sample_rate
        for scene in script.scenes:
            text = speech_text(scene)
            chunks = [c.audio_float_array for c in self.voice.synthesize(text)]
            if not chunks:
                raise ValueError("Speech model returned no audio")
            audio = np.concatenate(chunks).astype(np.float32)
            if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError("Speech model returned invalid audio")
            segments.append(
                {"text": text, "start": cursor / rate, "end": (cursor + len(audio)) / rate}
            )
            arrays.append(audio)
            cursor += len(audio)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined = np.concatenate(arrays)
        peak = np.max(np.abs(combined))
        if peak > 0.99:
            combined *= 0.99 / peak
        temporary = output.with_name(output.stem + ".partial.wav")
        sf.write(temporary, combined, rate, subtype="PCM_16")
        temporary.replace(output)
        metadata = {
            "language": self.language,
            "sample_rate": rate,
            "duration": cursor / rate,
            "segments": segments,
            "synthetic_audio": True,
            "engine": "piper",
            "voice": VOICES[self.language],
            "revision": REVISION,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output.with_suffix(".json"), metadata)
        return metadata
