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

Arabic writing targets simple, spoken Modern Standard Arabic, concrete explanation and practical steps. Few-shot examples guide style. Registered technical phrases such as `for loop` may appear in display narration and are converted to an Arabic-only reading for the speech engine; bare code identifiers such as `if` or `console` are still rejected. A second correction call is allowed only after validation failure; `--text-attempts 1` disables it. These checks do not establish factual or editorial quality. A rejected actual model attempt is retained in [the v2 examples](examples/presenter_v2).

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
| Measured v2 quality loop (CPU gate + GPU QLoRA cycles) | [05_quality_loop.ipynb](notebooks/05_quality_loop.ipynb) |
| Large Stickman drawing corpus + CPU measurement loop | [06_stickman_artist.ipynb](notebooks/06_stickman_artist.ipynb) |

## v2 Arabic, code-switching and motion data

`datasets/` now contains authored synthetic splits for full tashkeel (600 rows), mixed Arabic/English technical terms (760 rows), bounded motion plans (1200 rows), 930/95/95 scriptwriter records, 3240 scene annotations and 140 voice-recording prompts. A 95-term pronunciation lexicon and a motion vocabulary are mirrored into `voice2reel/assets/`. All v2 material is assistant-authored, CC0-1.0, deterministic (`python scripts/build_datasets.py`), and **not expert-reviewed**.

```bash
python scripts/build_datasets.py            # regenerate byte-identical files
python scripts/build_datasets.py --check    # verify reproducibility
python scripts/check_data.py                # validate all splits, leakage, mirrors, manifest
python -m voice2reel.quality_gate --output examples/quality_v2/rounds.json
```

`voice2reel/quality_gate.py` fits three small front-end models on CPU (diacritizer, code-switch reader, comfort classifier) and measures them on held-out splits; the Qwen scriptwriter is trained separately on the user's Colab GPU in notebook 05. The measured CPU front-end score is documented in [quality_v2/rounds.json](examples/quality_v2/rounds.json) and [QUALITY_V2.md](docs/QUALITY_V2.md). It is **not** a 90% claim about factual accuracy, voice naturalness, subjective video quality, or the production Qwen3-4B writer.

## Large Stickman drawing corpus

`datasets/stickman/` adds **60,000** deterministic vector drawing plans (50,000/5,000/5,000). Each row stores 18 named 2D joints, 14 numeric features, an actor box, camera/gesture/emotion/intent metadata, and either a professional plan or exactly one of 13 named defects. The corpus teaches two-bone IK, proportions, stroke style, layer order, safe zones, easing and defect detection to the **existing deterministic vector renderer** — it is not a raster/diffusion image dataset. All rows are assistant-authored synthetic CC0-1.0 material with no expert review.

```bash
python scripts/build_stickman_drawing_data.py --check
python -m voice2reel.stickman_artist --output examples/stickman_drawing/artist_rounds.json
python -m voice2reel.stickman_artist --render-samples examples/stickman_drawing/samples
```

`voice2reel/stickman_artist.py` fits a small CPU ridge model for joint prediction and a learned max-margin defect detector. Three documented rounds reach `0.9304` on validation and `0.9299` on the held-out test split; see [artist_rounds.json](examples/stickman_drawing/artist_rounds.json), [sample frames](examples/stickman_drawing/samples) and [QUALITY_V2.md](docs/QUALITY_V2.md). This is a measured synthetic CPU benchmark, **not** a claim about subjective professional aesthetics, production video quality, or improved `Qwen` weights. Notebook 06 reproduces it on CPU; notebook 05 still needs the user's T4 GPU for the writer.

For cloning, install `requirements/voice.txt` and select `--voice-engine xtts --reference voice.wav`. It is slower. Older 1.5B adapters are incompatible with the Qwen3 4B writer; no new trained weights are shipped.

See [research and license notes](docs/RESEARCH.md). Piper runtime/voice data, XTTS and online speech service terms are separate. No Instagram video assets, voice weights or third-party music recordings are redistributed. Realistic-video notebook 03 and earlier motion studies remain historical experiments; reference-video analysis is not an automatic feature of the current pipeline.
