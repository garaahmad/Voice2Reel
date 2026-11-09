"""Original synthesized music and transition effects, mixed beneath speech on CPU."""

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from voice2reel.common import write_json

VERSION = "original-score-1"
RATE = 24000


def add(track, sound, at):
    start = max(0, round(at * RATE))
    count = min(len(sound), len(track) - start)
    if count > 0:
        track[start : start + count] += sound[:count]


def tone(note, duration, kind="pluck"):
    t = np.arange(round(duration * RATE)) / RATE
    hz = 440 * 2 ** ((note - 69) / 12)
    if kind == "bass":
        wave = np.sin(2 * math.pi * hz * t) + 0.15 * np.sin(4 * math.pi * hz * t)
        envelope = np.minimum(t / 0.02, 1) * np.exp(-t * 4)
    else:
        wave = (
            np.sin(2 * math.pi * hz * t)
            + 0.25 * np.sin(4 * math.pi * hz * t)
            + 0.08 * np.sin(6 * math.pi * hz * t)
        )
        envelope = np.minimum(t / 0.008, 1) * np.exp(-t * 6)
    return (wave * envelope).astype(np.float32)


def score(duration):
    """Self-authored restrained Am–F–C–G instrumental, with soft percussion."""
    n = math.ceil(duration * RATE)
    out = np.zeros(n, dtype=np.float32)
    beat = 60 / 96
    chords = [(57, 60, 64, 69), (53, 57, 60, 65), (48, 55, 60, 64), (55, 59, 62, 67)]
    rng = np.random.default_rng(2409)
    for bar in range(math.ceil(duration / (4 * beat))):
        chord = chords[bar % 4]
        start = bar * 4 * beat
        for step, note in enumerate([chord[0] + 12, chord[2] + 12, chord[1] + 12, chord[3]]):
            add(out, tone(note, 0.8) * 0.15, start + (step + 0.5) * beat)
        for step in (0, 2):
            add(out, tone(chord[0] - 12, 0.55, "bass") * 0.22, start + step * beat)
            t = np.arange(3600) / RATE
            kick = np.sin(2 * math.pi * (48 * t + 55 * (1 - np.exp(-t * 25)) / 25)) * np.exp(-t * 30)
            add(out, kick * 0.16, start + step * beat)
        for step in range(8):
            noise = rng.standard_normal(1440)
            hat = signal.lfilter([1, -1], [1], noise) * np.exp(-np.arange(1440) / 180)
            add(out, hat.astype(np.float32) * 0.006, start + (step + 0.5) * beat / 2)
    t = np.arange(n) / RATE
    out *= np.minimum(t / 0.6, 1) * np.minimum((duration - t) / 0.8, 1).clip(0, 1)
    return out


def effects(duration, segments):
    out = np.zeros(math.ceil(duration * RATE), dtype=np.float32)
    rng = np.random.default_rng(802)
    for i, segment in enumerate(segments):
        at = segment["start"]
        if i:
            t = np.arange(round(RATE * 0.24)) / RATE
            noise = signal.sosfilt(
                signal.butter(2, [450, 3800], fs=RATE, btype="bandpass", output="sos"),
                rng.standard_normal(len(t)),
            )
            envelope = np.sin(math.pi * np.arange(len(t)) / len(t)) ** 2
            swoosh = (
                noise * envelope * 0.035 + np.sin(2 * math.pi * (650 * t - 800 * t * t)) * envelope * 0.012
            )
            add(out, swoosh.astype(np.float32), max(0, at - 0.10))
        add(out, tone(76 if i == len(segments) - 1 else 69, 0.18) * 0.055, at + 0.25)
        if i == len(segments) - 1:
            add(out, tone(81, 0.22) * 0.045, at + 0.39)
    return out


def decode(path, output, *, normalize=False):
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE)]
    if normalize:
        args += ["-af", "highpass=f=70,loudnorm=I=-17:TP=-2:LRA=9"]
    subprocess.run(args + [str(output)], check=True, capture_output=True)
    samples, rate = sf.read(output, dtype="float32")
    if rate != RATE or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Invalid audio for mixing")
    return samples


def mix_audio(narration, metadata, output, *, music=True, sfx=True, music_path=None):
    if music_path and not music:
        raise ValueError("Choose custom music or --no-music, not both")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = metadata["duration"]
    n = round(duration * RATE)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        voice = decode(narration, Path(work) / "voice.wav", normalize=True)
        voice = np.pad(voice[:n], (0, max(0, n - len(voice))))
        bed = np.zeros(n, dtype=np.float32)
        if music:
            if music_path:
                bed = decode(music_path, Path(work) / "music.wav")
                # Smooth the seam of each repetition; never execute uploaded audio as code.
                fade = min(round(RATE * 0.12), len(bed) // 4)
                if fade:
                    bed[:fade] *= np.linspace(0, 1, fade)
                    bed[-fade:] *= np.linspace(1, 0, fade)
                bed = np.tile(bed, math.ceil(n / len(bed)))[:n]
            else:
                bed = score(duration)[:n]
            rms = float(np.sqrt(np.mean(bed**2)))
            bed *= 0.028 / max(0.001, rms)
            # Envelope following reduces music during speech, recovers gently in pauses.
            energy = np.sqrt(uniform_filter1d(voice * voice, size=2400).clip(0))
            duck = 1 - 0.72 * (energy / 0.025).clip(0, 1)
            bed *= duck.astype(np.float32)
        fx = effects(duration, metadata["segments"])[:n] if sfx else np.zeros(n, dtype=np.float32)
        mixed = voice + bed + fx
        peak = float(np.max(np.abs(mixed)))
        gain = min(1.0, 0.94 / max(0.001, peak))
        for name, data in [("music", bed), ("effects", fx)]:
            sf.write(output.parent / (name + ".wav"), data, RATE, subtype="PCM_16")
        temporary = Path(work) / "soundtrack.wav"
        sf.write(temporary, mixed * gain, RATE, subtype="PCM_16")
        temporary.replace(output)
        report = {
            "version": VERSION,
            "duration": duration,
            "music_enabled": music,
            "effects_enabled": sfx,
            "music_source": str(music_path)
            if music_path
            else "original synthesized score"
            if music
            else None,
            "music_ducking": True if music else False,
            "speech_target_lufs": -17,
            "output_peak": float(np.max(np.abs(mixed * gain))),
            "music_rms": float(np.sqrt(np.mean(bed**2))),
            "effects_rms": float(np.sqrt(np.mean(fx**2))),
            "master_gain": gain,
        }
        write_json(output.with_suffix(".json"), report)
        return report
