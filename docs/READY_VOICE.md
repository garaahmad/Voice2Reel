# Ready-made Arabic and English AI narration

Open [00_ready_voice_ar_en.ipynb](../notebooks/00_ready_voice_ar_en.ipynb) for the simplest path. Choose the language, a preset speaker, and text. Generate, listen, and download WAV. No voice sample, model training, or API key is needed. The first run downloads the pinned XTTS model; subsequent runs use its local cache.

Use Python 3.12 with torch/torchaudio 2.8.0 and `requirements/voice.txt`. The notebook handles installation. GPU is preferred for speed; `--device auto` falls back to CPU.

```bash
# Arabic text, default preset voice
python -m voice2reel.voice.synthesize --content "مرحبًا بكم في هذا المثال." --language ar --output outputs/ar.wav

# English text, a different preset
python -m voice2reel.voice.synthesize --content "Welcome to this example." --language en --speaker "Andrew Chipper" --output outputs/en.wav

# List available preset voices (small speaker asset download only)
python -m voice2reel.voice.synthesize --list-speakers

# Existing script: use its language field and preserve beat boundaries
python -m voice2reel.voice.synthesize --script outputs/script.json --output outputs/narration.wav

# Optional voice cloning still works
python -m voice2reel.voice.synthesize --text narration.txt --language ar --reference my_voice.wav
```

`--text` reads a UTF-8 file. `--content` accepts direct text. `--script` reads validated JSON. Supply exactly one. For plain text, the default language is Arabic; explicitly use `--language en` for English. For script JSON, language is inferred; an explicit conflicting language is rejected. Choose one preset with `--speaker` or one reference with `--reference`.

Outputs: mono 24 kHz PCM WAV and adjacent JSON with language, voice mode, preset name, `synthetic_audio: true`, and actual segment timing. A segment boundary is not word-level alignment. A generated voice can still mispronounce words; compare voices and listen before using the result. Arabic/English code-switching within one sentence has not been evaluated.

English script generation is available with `python -m voice2reel.scriptwriter.generate --idea "Explain how to plan a project" --language en --duration 20`. The bundled fine-tuning dataset remains Arabic-only; English adaptation quality has not been measured. Script generation also supports CPU inference; training still requires CUDA. See VIDEO_DEMO.md for resource notes.

Implementation uses upstream [XTTS preset-speaker support](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html#coqui-speakers). Its pinned speaker file contains 58 voices. The [CPML model/output license](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) allows non-commercial use; preset voices do not change these terms.
