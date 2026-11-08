# Voice2Reel

Create a **2D cartoon-presenter reel** from a written idea or reviewed script, with expressive faces, gestures, animated explanatory panels, captions, and Arabic/English AI speech.

**Working local prototype:** the new fast pipeline runs end to end. It uses one compact text-model pass, local Piper speech, and CPU rendering. Character artwork is drawn in code and inspired by the [user's reference](docs/STICKMAN_REFERENCE.md); no assets were copied from the reel. Generated writing still needs editorial review. There is no web app or hosted job service yet.

## Start here

Use **[04_fast_stickman.ipynb](notebooks/04_fast_stickman.ipynb)** on Colab, or read the **[Arabic fast-workflow guide](docs/FAST_PRESENTER.md)**.

```bash
# Python 3.12 on Linux/Colab; install ffmpeg and an Arabic-capable font.
python -m pip install -r requirements/presenter.txt
python -m voice2reel.presenter --script examples/fast_presenter/semicolon_ar.json --output outputs/presenter
```

This reviewed example needs **no GPU, Torch or text-model download**. It creates `script.json`, `narration.wav`, segment timings, `video.mp4`, progress and performance reports. For English, use `semicolon_en.json`.

For a written idea on CPU, also install the optional GGUF backend:

```bash
python -m pip install --no-binary llama-cpp-python -r requirements/director-cpu.txt
python -m voice2reel.presenter --idea "Explain a programming idea" --interactive --device cpu --output outputs/my-reel
```

`--interactive` asks five questions: audience, target duration, language, message/facts, and visual focus. Supply flags instead for automation. The CPU backend downloads about 2.5 GB of Qwen3 weights once. On CUDA, install a suitable Torch build and `requirements/text.txt`, then use `--device cuda`. The Colab notebook handles CPU/GPU setup in idea mode; GPU is optional. GPU execution of this new flow has not been measured here.

## What makes it fast?

- AI writes only narration, headings and diagram labels in one pass. The engine compiles animation controls and computes frames directly.
- Piper supplies fast local preset speech: Kareem for Arabic, Lessac for English.
- Cached artwork, background/text layers and cropped character transforms reduce per-frame work.
- Script, speech and video have separate content-verified caches. Visual edits preserve narration; changed speech invalidates the audio cache.
- Default `fast`: 540×960 at 24 FPS. `quality`: 720×1280 output at 30 FPS.
- One text attempt by default; `--text-attempts 2` explicitly permits another format-repair attempt.

### Actual CPU measurements

Measured on a two-core CPU quota, with model files already downloaded:

| Run | Observed time |
|---|---|
| Reviewed Arabic script → about 31 s of voiced video | about 10 s, including voice and rendering |
| New written idea → complete AI-written/voiced video | about 54 s for a ~14 s sample; text generation took most of the time |
| Identical reviewed-script job, existing outputs valid | about 0.2 s including CLI startup; reuses the existing video |

These are observations, not service guarantees. Installation and first model downloads are excluded. AI writing on CPU still takes time; a cache hit is not generation of a new story. See [recorded measurements](examples/fast_presenter/benchmark.local.json) and the [actual AI run](examples/fast_presenter/actual_ai_run/README.md).

## Character and animation

The new renderer includes a beret-wearing character with separate eyes, eyebrows and mouth; visible hands and feet; six presenter gestures; expressions; pose blending and recovery; camera framing; code/list/flow/comparison panels; and Arabic/English text.

The mouth opening responds to audio energy. Subtitles use estimated phrase pacing. Neither is phoneme-level or word-level forced alignment. The renderer supports a finite presenter vocabulary, not arbitrary generated character actions or multiple-character scenes. Target duration guides writing; output length follows actual narration.

Advanced reviewed scripts can select `gesture`, `emotion`, `camera`, and `visual` per scene. Existing legacy scripts with three to six beats can be adapted. Code snippets are drawn as text and never executed. See the sample JSON files and [guide](docs/FAST_PRESENTER.md).

## Voice cloning and existing training tools

Use `--voice-engine xtts --reference voice.wav` for optional cloning, after installing `requirements/voice.txt`. This is a slower mode. Preset XTTS speakers are also available through `--speaker`.

| Tool | Entry point |
|---|---|
| Fast presenter video | [04_fast_stickman.ipynb](notebooks/04_fast_stickman.ipynb) |
| Original ready XTTS voice | [00_ready_voice_ar_en.ipynb](notebooks/00_ready_voice_ar_en.ipynb) |
| Original 1.5B scriptwriter QLoRA training | [01_scriptwriter_qlora.ipynb](notebooks/01_scriptwriter_qlora.ipynb) |
| XTTS cloning/fine-tuning | [02_arabic_voice_xtts.ipynb](notebooks/02_arabic_voice_xtts.ipynb) |

Fast text generation uses pinned Qwen3-4B-Instruct-2507 weights. The older 1.5B training adapters are not compatible with it. No newly trained weights are shipped. Piper's runtime is GPL-3.0; individual voice/data terms must also be reviewed. XTTS weights/outputs use non-commercial CPML. See [research and licenses](docs/RESEARCH.md).

The previous stickman study and realistic-video experiment remain in the repository as historical work. Notebook 03 and `realistic.py` are not the default workflow. Reference videos currently guide design; the fast pipeline does not analyze arbitrary uploaded videos or clone their animation style automatically.
