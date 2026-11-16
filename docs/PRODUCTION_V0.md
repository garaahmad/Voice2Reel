# Production v0: Qwen → restricted Python scene → locked character → MP4

This branch adds a **new experimental end-to-end path**, not a claim of production-ready video quality.

## Why restricted Python?

Qwen outputs `scene(...)` and `shot(...)` Python calls. `python_scene_dsl.parse_python_scene` uses AST to extract literal arguments and validates them against a Pydantic schema. It **never executes Qwen-generated Python**. The actual renderer uses only the selected Character Lab template, so Qwen cannot replace its artwork. Arbitrary Python programs from the older dataset are **not** accepted by this engine.

## Workflow

Install the project's Character Lab requirements and text-training requirements in appropriate environments. For online TTS, install `edge-tts==7.3.1`.

1. Select and confirm one of the 12 characters in `notebooks/08_character_lab.ipynb`, saving its lock JSON. The lock is verified before every render.
2. Generate training data: `python scripts/build_python_scene_dataset.py`. This creates **240 train / 48 validation / 48 test** examples. Topics are held out, but syntax/templates are shared; this is synthetic starter data and **not** an independently artist-reviewed corpus.
3. On CUDA: `python -m voice2reel.train_python_scene --output outputs/qwen_python_scene`. This is a real QLoRA training entry point; training **has not been run** in this delivery.
4. Generate and validate a scene from the resulting adapter:

```bash
python -m voice2reel.generate_python_scene \
  --idea "Explain Git to beginners" \
  --lock outputs/character_lab/character_lock.json \
  --adapter outputs/qwen_python_scene/adapter \
  --output outputs/scene.json
```

5. Optional online TTS (sends narration text to a third-party service):

```bash
python -m voice2reel.production_voice \
  --plan outputs/scene.json --output outputs/voice.mp3 --language en
```

6. Render the character, captions, and mix your **licensed** music:

```bash
python -m voice2reel.production_pipeline \
  --plan outputs/scene.json \
  --lock outputs/character_lab/character_lock.json \
  --voice outputs/voice.mp3 --music assets/licensed_music.mp3 \
  --output outputs/final.mp4
```

For silent video, omit `--voice` and `--music`. For narration-only, omit `--music`. Music is user-provided, never scraped or generated without a licensed source.

## Known gaps (important)

- **No GPU Qwen training was performed**; no adapter checkpoint is included.
- TTS is functional code but requires internet access and has **not** been run here.
- Narration is synthesized as one audio stream; **word/shot-level alignment and duration matching are not implemented**. For professional production, synthesize per-shot, measure durations, derive scene timing, then mux.
- Gesture transitions blend toward idle but have not been visually reviewed for anatomy or flicker. Camera modes and transition values are validated but **not yet rendered** as distinct effects.
- Caption overflow, Arabic layout and mobile safe zones need actual frame-by-frame quality gates.
- Output is 540×960 for the current renderer, not native 1080×1920.
- No full local render, full test suite, or subjective video quality evaluation has been completed in this delivery.
- Existing `scripts/build_python_motion_data.py` generates free-form Python examples and is **not compatible** with the restricted scene DSL trainer. Keep the datasets separate.
- The locked character is fixed by the user. Never use `exec` on model outputs.

## Next quality gates

Run syntax/AST tests, a 2-second silent render for all 12 characters, a voice+music mux smoke test, inspect frames, measure A/V durations, evaluate held-out topics with base-vs-adapter validity, and collect human ratings for visual quality. Reject or regenerate failed scenes rather than silently shipping them.
