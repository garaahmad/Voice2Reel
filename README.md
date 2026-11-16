<p align="center">
  <img src="docs/assets/hero.png" alt="Voice2Reel — Arabic and English animated explainers, with actual presenter frames" width="100%">
</p>

<p align="center">
  <a href="#quick-start"><strong>Get started</strong></a> ·
  <a href="#see-it-in-motion"><strong>See it in motion</strong></a> ·
  <a href="docs/FAST_PRESENTER.md"><strong>الدليل بالعربية</strong></a> ·
  <a href="#development"><strong>Development</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Language-Arabic%20%2B%20English-23856D" alt="Arabic and English">
  <img src="https://img.shields.io/badge/Output-Vertical%20MP4-E34D3E" alt="Vertical MP4 output">
  <img src="https://img.shields.io/badge/Status-Prototype-D9A441" alt="Prototype status">
</p>

# Voice2Reel

**Create narrated, vertical videos with an animated 2D presenter — in Arabic or English.**

Voice2Reel combines a structured script, speech synthesis, character animation, explanatory graphics, music, and sound effects into an MP4. Start with a reviewed script for direct control, or use the optional AI writer to turn an idea into a draft.

The current release is a **local Python prototype** with a command-line workflow and Colab notebooks. It does not include a hosted web application. AI-written scripts need human review before publication.

## See it in motion

<p align="center">
  <img src="docs/assets/presenter-demo.gif" alt="Animated Arabic presenter moving through thinking, pointing, and thumbs-up scenes with code panels and captions" width="360">
</p>

<p align="center"><sub>Actual project renderer · 9-second visual preview · silent GIF</sub></p>

The preview shows the renderer's gestures, expressions, captions, and code highlighting. It uses synthetic audio energy and demonstration timing to animate the mouth; it does not demonstrate speech synthesis or word alignment.

![Three scenes rendered by Voice2Reel: the question, explanation, and takeaway](docs/assets/scene-gallery.png)

## What it does

- **Arabic and English narration:** online neural speech by default, with optional Piper speech and XTTS voice cloning.
- **Animated presenter:** expressive faces, six gestures, camera framing, and audio-driven mouth movement.
- **Explanatory visuals:** captions, diagrams, labels, and code panels with explicit highlighting.
- **Sound design:** generated instrumental background music, timed effects, and music ducking during speech.
- **Editable output:** separate narration, music, effects, timing metadata, and the final video.
- **Incremental rendering:** independent caches for scripts, speech, mixing, and video reduce repeated work when you edit a project.

| Output profile | Resolution | Frame rate |
| --- | --- | --- |
| `fast` (default) | 540 × 960 | 24 FPS |
| `quality` | 720 × 1280 | 30 FPS |

## Quick start

### Run locally

Use **Python 3.12 on Linux**, with FFmpeg and DejaVu Sans Mono installed. The repository also bundles the Cairo font and its SIL Open Font License.

On Ubuntu/Debian, install the system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-dejavu-core
```

Then install the presenter in a virtual environment:

```bash
git clone https://github.com/garaahmad/Voice2Reel.git
cd Voice2Reel
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/presenter.txt

python -m voice2reel.presenter \
  --script examples/presenter_v2/semicolon_ar.json \
  --output outputs/presenter
```

The example uses a **manually reviewed Arabic script**. It requires no GPU, Torch, text-model download, or reference recording. Internet access is required for the default speech service.

### Run in Colab

Open [04_fast_stickman.ipynb](notebooks/04_fast_stickman.ipynb) in Colab and follow its setup cells. See the [Arabic walkthrough](docs/FAST_PRESENTER.md) for the reviewed-example and idea-generation workflows. Execution on an actual Colab allocation has not been benchmarked here.

## How the pipeline works

```text
Reviewed script ──────────────────────┐
                                    ├─→ Speech → Animation + audio mix → MP4
Idea → Optional AI writer → Review ──┘
```

The character is drawn by the Python renderer. The optional Qwen3 4B writer generates script text; it does not generate the character artwork.

A script defines narration, captions, gestures, camera framing, and explanatory visuals for each scene. See the [Arabic example](examples/presenter_v2/semicolon_ar.json) for the structure and the [presenter guide](docs/PRESENTER_V2.md) for visual fields and sound settings.

The output directory includes:

| File | Contents |
| --- | --- |
| `script.json` | The script used for the video |
| `narration.wav` | Raw speech used to drive mouth movement |
| `music.wav` / `effects.wav` | Separate background music and effects |
| `soundtrack.wav` / `soundtrack.json` | Final audio mix and mix metadata |
| `video.mp4` | Rendered vertical video |
| `progress.json` | Pipeline status, including failures |

Timing metadata and performance reports are also saved. Visual edits can reuse unchanged speech; music edits rebuild the mix and video. An identical request can reuse the cached video. Avoid running two jobs against the same output directory.

## Choose a voice

| Engine | Use | Requirements |
| --- | --- | --- |
| `edge` (default) | Online neural narration | Internet access to Microsoft's speech service |
| `piper` | Local preset narration | Initial voice-model download; local inference afterward |
| `xtts` | Optional reference-voice cloning | Additional dependencies, model downloads, and a reference recording |

The default engine uses `edge-tts==7.3.1` with `ar-JO-TaimNeural` for Arabic and `en-US-GuyNeural` for English. **Narration text is sent to Microsoft's external service.** These voices are not project-trained weights. The client is unofficial, service availability can vary, and the pipeline reports failure without automatically switching engines.

```bash
# Local Piper speech after the initial model download
python -m voice2reel.presenter \
  --script examples/presenter_v2/semicolon_ar.json \
  --voice-engine piper --output outputs/piper

# Adjust online speech speed
python -m voice2reel.presenter \
  --script examples/presenter_v2/semicolon_ar.json \
  --speech-rate 5 --output outputs/faster
```

For XTTS, install `requirements/voice.txt` in addition to the presenter dependencies and select `--voice-engine xtts --reference voice.wav`. This path has additional compute and model requirements.

## Customize the video

```bash
# Higher-resolution output
python -m voice2reel.presenter \
  --script examples/presenter_v2/semicolon_ar.json \
  --profile quality --output outputs/quality

# Disable music and sound effects
python -m voice2reel.presenter \
  --script examples/presenter_v2/semicolon_ar.json \
  --no-music --no-sfx --output outputs/quiet

# Supply a music file you have permission to use
python -m voice2reel.presenter \
  --script examples/presenter_v2/semicolon_ar.json \
  --music music.mp3 --output outputs/custom
```

Run `python -m voice2reel.presenter --help` for all options.

## Generate a script from an idea

Idea mode adds a text model to the presenter workflow. For the CPU backend, install a C/C++ build toolchain and CMake, then run:

```bash
CC=gcc CXX=g++ \
CMAKE_ARGS="-DGGML_CUDA=OFF -DLLAMA_CURL=OFF" \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
python -m pip install --no-binary llama-cpp-python -r requirements/director-cpu.txt

python -m voice2reel.presenter \
  --idea "Explain a programming idea" \
  --interactive --device cpu --output outputs/my-reel
```

Five questions capture the audience, duration, language, facts, and visual focus. The pinned Qwen3 4B CPU backend downloads approximately **2.5 GB** on first use. CUDA requires a compatible Torch build and `requirements/text.txt`; the Colab notebook handles the choice of CPU/GPU dependencies in idea mode.

The writer validates structure and language, and may make one correction attempt after validation fails. Use `--text-attempts 1` to disable that retry. These checks do not verify factual accuracy or editorial quality. Review the resulting script before sharing it; the [writing review examples](examples/presenter_v2/writing_review/README.md) document observed limitations.

Arabic writing targets simple, spoken Modern Standard Arabic, concrete explanation and practical steps. Few-shot examples guide style. Registered technical phrases such as `for loop` may appear in display narration and are converted to an Arabic-only reading for the speech engine; bare code identifiers such as `if` or `console` are still rejected. A second correction call is allowed only after validation failure; `--text-attempts 1` disables it. These checks do not establish factual or editorial quality. A rejected actual model attempt is retained in [the v2 examples](examples/presenter_v2).

## Development

Install the development tools alongside the presenter dependencies:

```bash
python -m pip install -r requirements/dev.txt
ruff check .
python scripts/check_data.py

# Presenter, audio, rendering, evaluation, and notebook checks
pytest -q tests/test_presenter.py tests/test_sound_design.py \
  tests/test_render.py tests/test_motion_preview.py \
  tests/test_evaluation.py tests/test_notebooks.py
```

Rendering tests use synthetic narration and exercise FFmpeg without calling an online voice service. The full `pytest -q` suite also covers training, XTTS, and experimental video tools; it needs additional dependencies. See the [CI workflow](.github/workflows/tests.yml) for that setup.

## Additional tools

The repository retains training and voice experiments alongside the current presenter:

| Notebook | Purpose |
| --- | --- |
| [04_fast_stickman](notebooks/04_fast_stickman.ipynb) | Current presenter, narration, music, and effects |
| [00_ready_voice_ar_en](notebooks/00_ready_voice_ar_en.ipynb) | Original preset XTTS narration |
| [01_scriptwriter_qlora](notebooks/01_scriptwriter_qlora.ipynb) | Original 1.5B QLoRA scriptwriter training |
| [02_arabic_voice_xtts](notebooks/02_arabic_voice_xtts.ipynb) | XTTS cloning and fine-tuning |
| [05_quality_loop](notebooks/05_quality_loop.ipynb) | Measured v2 quality loop: CPU gate + GPU QLoRA cycles |
| [06_stickman_generator](notebooks/06_stickman_generator.ipynb) | Stickman drawing corpus + CPU/MLP demo pipeline |
| [07_stickman_full_production](notebooks/07_stickman_full_production.ipynb) | Full Stickman production studio: train once → Qwen content → silent video → voice → mux |
| [08_character_lab](notebooks/08_character_lab.ipynb) | 12 character choices → PNG preview → explicit lock → CPU compositional motion training → silent MP4; optional Qwen before/after evaluation ([measured results and limits](docs/CHARACTER_LAB.md)) |

Older 1.5B adapters are incompatible with the current Qwen3 4B writer. No newly trained weights are shipped. Realistic-video notebook 03 and earlier motion studies are historical experiments, separate from the current presenter workflow.

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

`voice2reel/stickman_artist.py` fits a small CPU ridge model for joint prediction and a learned max-margin defect detector. Three documented rounds reach `0.9304` on validation and `0.9299` on the held-out test split; see [artist_rounds.json](examples/stickman_drawing/artist_rounds.json), [sample frames](examples/stickman_drawing/samples) and [QUALITY_V2.md](docs/QUALITY_V2.md). This is a measured synthetic CPU benchmark, **not** a claim about subjective professional aesthetics, production video quality, or improved `Qwen` weights.

For the drawing-data/demo path, `notebooks/06_stickman_generator.ipynb` regenerates/validates the 60k corpus, fits the CPU gate, trains a small PyTorch MLP and assembles a deterministic MP4 demo. For the **full production studio**, run `notebooks/07_stickman_full_production.ipynb`: it trains `Qwen2.5-1.5B` LoRA + the Stickman MLP **once**, saves `model_training_results.json` / `نتائج_تدريب_الموديل.json`, then reuses that file to turn a title + idea + source text into a Qwen-generated script, a silent video, scene audio, and a synced final MP4. It requires T4 for the one-off training; later runs reuse the saved file and do **not** retrain. Notebook 05 still measures the production writer and writes the Qwen `final_report.json`.

## Current limitations

- Mouth animation follows audio energy; it is not phoneme-level lip sync.
- Gestures come from a fixed vocabulary. The renderer does not generate arbitrary character movements.
- Code panels display script-provided content and highlights; they do not execute or verify code.
- AI-generated Arabic can need substantial editing. The reviewed example is not raw model output.
- Runtime depends on video settings, speech-service latency, model downloads, and cache state. The [earlier CPU benchmark](examples/fast_presenter/benchmark.local.json) measures the previous renderer and Piper voice, not the current version. See [v2 observations](examples/presenter_v2/measurements.json) for separate measurements.
- GPU execution has not been benchmarked here. Reference-video analysis is not an automatic pipeline feature.

## Documentation and third-party terms

- [Presenter design, audio, and output guide — Arabic](docs/PRESENTER_V2.md)
- [Colab walkthrough — Arabic](docs/FAST_PRESENTER.md)
- [Current examples and review notes](examples/presenter_v2/README.md)
- [Research, sources, and license notes](docs/RESEARCH.md)

Piper, its voice data, XTTS, and the online speech service have separate terms. The bundled Cairo font includes its license. The repository does not redistribute Instagram video assets, voice weights, or third-party music recordings.
