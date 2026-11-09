"""Optional online neural speech via Edge; no silent substitution with another voice."""

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voice2reel.common import write_json

VERSION = "edge-tts-7.3.1-v2-arabic-terms"
VOICES = {"ar": "ar-JO-TaimNeural", "en": "en-US-GuyNeural"}


def validate_settings(language, voice, rate):
    if language not in VOICES:
        raise ValueError("Choose Arabic or English")
    voice = voice or VOICES[language]
    if not voice.startswith(language + "-") or not voice.endswith("Neural") or len(voice) > 100:
        raise ValueError("Voice locale must match the script language")
    if not isinstance(rate, int) or not -20 <= rate <= 25:
        raise ValueError("Speech rate must be between -20 and +25 percent")
    return voice


class NeuralVoice:
    def __init__(self, language, voice=None, rate=0):
        self.voice = validate_settings(language, voice, rate)
        self.language, self.rate = language, rate

    async def _scenes(self, script, work):
        import edge_tts
        import edge_tts.communicate as network

        # Honor an explicitly supplied corporate/session CA without disabling verification.
        if os.getenv("SSL_CERT_FILE"):
            network._SSL_CTX.load_verify_locations(cafile=os.environ["SSL_CERT_FILE"])
        semaphore = asyncio.Semaphore(2)

        async def scene(index, text):
            async with semaphore:
                speech = edge_tts.Communicate(
                    # Separate the technical name for the Arabic speech engine;
                    # keep the reviewed source text unchanged in metadata.
                    re.sub(r"(?<!\w)جافا\s*سكري?بت(?!\w)", "جافا سكريبت", text)
                    if self.language == "ar" else text,
                    self.voice,
                    rate=f"{self.rate:+d}%",
                    boundary="WordBoundary",
                    proxy=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
                    connect_timeout=10,
                    receive_timeout=25,
                )
                words, encoded = [], bytearray()
                async for event in speech.stream():
                    if event["type"] == "audio":
                        encoded.extend(event["data"])
                    elif event["type"] == "WordBoundary":
                        words.append(
                            {
                                "text": event["text"],
                                "start": event["offset"] / 1e7,
                                "end": (event["offset"] + event["duration"]) / 1e7,
                            }
                        )
                if not encoded:
                    raise ValueError("Online voice returned no audio")
                mp3 = work / f"{index}.mp3"
                wav = work / f"{index}.wav"
                mp3.write_bytes(encoded)
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", "24000", "-ac", "1", str(wav)],
                    check=True,
                    capture_output=True,
                )
                samples, rate = sf.read(wav, dtype="float32")
                if rate != 24000 or not len(samples) or not np.isfinite(samples).all() or not np.any(samples):
                    raise ValueError("Online voice returned invalid decoded speech")
                audible = np.flatnonzero(np.abs(samples) > 0.004)
                if len(audible):
                    keep = min(len(samples), int(audible[-1]) + 2880)
                    samples = samples[:keep]
                return samples, words

        tasks = [scene(i, s.narration) for i, s in enumerate(script.scenes)]
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=90)

    def synthesize(self, script, output):
        if script.language != self.language:
            raise ValueError("Voice language differs from script")
        started = time.perf_counter()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            try:
                chunks = asyncio.run(self._scenes(script, Path(directory)))
            except Exception as exc:
                raise RuntimeError(
                    "Online neural speech failed. Check connectivity or explicitly choose "
                    "--voice-engine piper for offline speech. No voice was substituted."
                ) from exc
            arrays, segments, cursor = [], [], 0
            for i, ((samples, words), scene) in enumerate(zip(chunks, script.scenes)):
                duration = len(samples) / 24000
                words = [
                    dict(
                        w,
                        start=max(0, min(duration, w["start"])) + cursor / 24000,
                        end=max(0, min(duration, w["end"])) + cursor / 24000,
                    )
                    for w in words
                    if np.isfinite(w["start"] + w["end"]) and w["end"] > w["start"]
                ]
                # Small, explicit inter-scene breath. Timings include it, so scenes remain contiguous.
                gap = np.zeros(4320 if i < len(chunks) - 1 else 1440, dtype=np.float32)
                arrays.extend([samples, gap])
                end = cursor + len(samples) + len(gap)
                segments.append(
                    {"text": scene.narration, "start": cursor / 24000, "end": end / 24000, "words": words}
                )
                cursor = end
            samples = np.concatenate(arrays)
            samples *= min(1.0, 0.97 / max(0.001, float(np.max(np.abs(samples)))))
            temporary = Path(directory) / "narration.wav"
            sf.write(temporary, samples, 24000, subtype="PCM_16")
            metadata = {
                "language": script.language,
                "duration": cursor / 24000,
                "sample_rate": 24000,
                "segments": segments,
                "engine": "edge",
                "voice": self.voice,
                "rate_percent": self.rate,
                "synthetic_audio": True,
                "online_service": "Microsoft Edge speech",
                "word_timing": "provider boundaries",
                "version": VERSION,
                "wall_seconds": time.perf_counter() - started,
            }
            temporary.replace(output)
            write_json(output.with_suffix(".json"), metadata)
            return metadata
