# Voice2Reel

Turn a written idea or reviewed script into a voiced **2D cartoon-presenter reel**, with expressive faces, gestures, explanatory graphics, Arabic/English narration, background music and transition effects.

**Local prototype:** code generates the full MP4. There is no hosted web app yet. The previous presenter design, Arabic writing and Piper voice were rejected by the user; [presenter v2](docs/PRESENTER_V2.md) redraws the character, changes the default voice, adds sound design and strengthens writing checks. The new design still needs the user's evaluation. AI writing needs editorial review.

## Try the new version

Open **[04_fast_stickman.ipynb](notebooks/04_fast_stickman.ipynb)** on Colab, or follow the [Arabic instructions](docs/FAST_PRESENTER.md).

```bash
# Python 3.12 on Linux; ffmpeg and DejaVu Sans Mono must be installed.
python -m pip install -r requirements/presenter.txt
python -m voice2reel.presenter --script examples/presenter_v2/semicolon_ar.json --output outputs/presenter
```

This example uses a **manually reviewed Arabic script**. It is not presented as raw model output. It needs no GPU, Torch or text-model download. The bundled Cairo font includes its SIL Open Font License.

The default `edge` voice uses **Microsoft Edge's online neural speech service** through `edge-tts==7.3.1`: Arabic `ar-JO-TaimNeural`, English `en-US-GuyNeural`. Text is sent to that external service; these are not locally trained project weights. Connectivity and service availability are required. No silent voice substitution occurs on failure. For an entirely local voice, select `--voice-engine piper`. Optional XTTS cloning remains available.

```bash
# Original instrumental music and timed transition effects are enabled by default.
python -m voice2reel.presenter --script script.json --no-music --no-sfx --output outputs/quiet
python -m voice2reel.presenter --script script.json --music permitted-track.mp3 --output outputs/custom
python -m voice2reel.presenter --script script.json --speech-rate 5 --output outputs/faster-speech
# Offline preset speech
python -m voice2reel.presenter --script script.json --voice-engine piper --output outputs/offline
```

Outputs include `script.json`, raw narration and segment/word timings, separate music/effects stems, `soundtrack.wav`, `video.mp4`, progress and performance reports. Speech drives the mouth independently of music. Music ducks during speech; the mix normalizes speech and controls peak levels.

## Start from a written idea

```bash
CC=gcc CXX=g++ CMAKE_ARGS="-DGGML_CUDA=OFF -DLLAMA_CURL=OFF" CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install --no-binary llama-cpp-python -r requirements/director-cpu.txt
python -m voice2reel.presenter --idea "Explain a programming idea" --interactive --device cpu --output outputs/my-reel
```

Five questions cover audience, duration, language, facts and visual focus. The pinned Qwen3 4B CPU backend downloads about 2.5 GB once. CUDA uses a suitable Torch build plus `requirements/text.txt`; the Colab notebook configures CPU/GPU dependencies in idea mode. GPU/Colab runtime execution has not been measured here.

Arabic writing targets simple, spoken Modern Standard Arabic, concrete explanation and practical steps. Few-shot examples guide style. Validation rejects Latin identifiers in Arabic narration. A second correction call is allowed only after validation failure; `--text-attempts 1` disables it. These checks do not establish factual or editorial quality. A rejected actual model attempt is retained in [the v2 examples](examples/presenter_v2).

## Animation and speed

Original vector character artwork includes a detailed face/beret, hands/feet, rounded curved limbs, separate shoulders and six gestures. Supersampled character layers, cached faces/text, cropped transforms and independent script/voice/mix/video caches keep rendering on CPU. Visual or music edits preserve speech when its content/settings are unchanged.

`fast` outputs 540×960 at 24 FPS; `quality` outputs 720×1280 at 30 FPS. The mouth responds to audio energy, not phoneme alignment. Edge subtitle phrases use provider word boundaries; offline voices retain estimated phrase pacing. Gestures are a finite vocabulary, not arbitrary generated actions. `visual.focus` and `visual.status` allow explicit code emphasis; code is displayed, never executed.

The [earlier ~10-second CPU benchmark](examples/fast_presenter/benchmark.local.json) applies to the previous simpler renderer and Piper voice. It is not a v2 speed claim. New observations are recorded separately in [v2 examples](examples/presenter_v2). AI writing and correction take additional time; first installs/downloads and online service latency vary. A cache hit reuses a video rather than generating a new story.

## Existing training and cloning tools

| Tool | Entry point |
|---|---|
| Current presenter, voice and music | [04_fast_stickman.ipynb](notebooks/04_fast_stickman.ipynb) |
| Original preset XTTS voice | [00_ready_voice_ar_en.ipynb](notebooks/00_ready_voice_ar_en.ipynb) |
| Original 1.5B QLoRA scriptwriter training | [01_scriptwriter_qlora.ipynb](notebooks/01_scriptwriter_qlora.ipynb) |
| XTTS cloning/fine-tuning | [02_arabic_voice_xtts.ipynb](notebooks/02_arabic_voice_xtts.ipynb) |

For cloning, install `requirements/voice.txt` and select `--voice-engine xtts --reference voice.wav`. It is slower. Older 1.5B adapters are incompatible with the Qwen3 4B writer; no new trained weights are shipped.

See [research and license notes](docs/RESEARCH.md). Piper runtime/voice data, XTTS and online speech service terms are separate. No Instagram video assets, voice weights or third-party music recordings are redistributed. Realistic-video notebook 03 and earlier motion studies remain historical experiments; reference-video analysis is not an automatic feature of the current pipeline.
