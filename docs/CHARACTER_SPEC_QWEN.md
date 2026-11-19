# Voice2Reel — Qwen CharacterSpec (safe JSON instead of free-form SVG)

## Why this exists

The old text-to-SVG adapter occasionally generated disconnected circles and
abstract shapes. It passed XML safety validation, but had **no knowledge of
character anatomy**. Increasing epochs cannot make that failure impossible.

This pipeline trains a *different* Qwen adapter to emit **strict JSON design
instructions**. A trusted, hand-authored SVG puppet supplies the shape and
exactly 14 rigged parts. The generated JSON changes only palette, expression,
eyes, brows, cheek highlights and accent style. **It cannot modify arm joints,
add hats, draw arbitrary SVG, or execute code.**

Pipeline:

\`\`\`text
English/Arabic prompt → Qwen + CharacterSpec adapter → validated JSON
                                                       ↓
                 trusted assets/mascot_v1/base.svg + 14-joint rig
                                                       ↓
                   styled SVG → frame-by-frame rig motion → Reels MP4
\`\`\`

### JSON Contract

\`\`\`json
{
  "version": 1,
  "palette": "navy",
  "expression": "curious",
  "eyes": "wide",
  "brows": "raised",
  "cheeks": "subtle",
  "accent": "coral"
}
\`\`\`

The allowed values live in \`voice2reel/character_spec.py\`. Unsupported
fields, arbitrary RGB strings, executable assets or joint coordinates are
rejected. The renderer is \`character_spec_renderer.py\` and
\`mascot_animator.py\`.

## Open Colab

\`notebooks/10_ai_story_director_all_in_one.ipynb\` on the
\`test-new-data\` branch. The notebook is ordered as follows:

1. Settings: keep \`CHARACTER_ENGINE = "rig"\`; keep scene
   \`RUN_TRAINING = False\` if you already have a trained scene adapter.
2. Setup and Mascot V1 pose preview.
3. **CharacterSpec training** (one time): set
   \`RUN_CHARACTER_SPEC_TRAINING = True\`, count \`1500\`, epochs \`2\`,
   and run on a T4 GPU. It writes a **separate** adapter to
   \`/content/drive/MyDrive/Voice2Reel_09/qwen_character_spec_.../adapter\`
   and remembers it in \`latest_character_spec_adapter.txt\`.
   This is not the old scene adapter and NOT the old text-to-SVG adapter.
4. **CharacterSpec generation**: switch \`CHARACTER_SPEC_MODE = "qwen"\`
   and write your style prompt. It validates the model output and saves
   \`character_spec.json\`, preview PNG, and a 4-pose preview sheet.
5. Proceed with your *existing* Qwen scene/story, voice and video stages.
   The rig-mode Reels renderer automatically uses the validated
   \`--character-spec\` JSON file.

**If you only want to test visual styling without GPU**, use
\`CHARACTER_SPEC_MODE = "preset"\`. It uses the known default JSON; the
model is not invoked, and this is labeled as a preset rather than claiming
to be a trained Qwen image.

### Local CPU preview

\`\`\`bash
python -m pip install -r requirements/character-lab.txt
PYTHONPATH=. python -m scripts.build_character_spec_dataset --count 1500 \
    --output datasets/character_spec
PYTHONPATH=. python -m scripts.preview_character_spec
PYTHONPATH=. python -m pytest -q tests/test_character_spec.py
\`\`\`

### Training (Colab/T4 GPU)

\`\`\`bash
python -m pip install -r requirements/text.txt
PYTHONPATH=. python -m voice2reel.train_character_spec \
    --check-only --train datasets/character_spec/train.jsonl \
    --validation datasets/character_spec/validation.jsonl

PYTHONPATH=. python -u -m voice2reel.train_character_spec \
    --train datasets/character_spec/train.jsonl \
    --validation datasets/character_spec/validation.jsonl \
    --output /content/drive/MyDrive/Voice2Reel_09/qwen_character_spec_YYYYMMDD_HHMMSS \
    --epochs 2
\`\`\`

Use a new empty output directory for training. \`--check-only\` verifies the
JSON, split uniqueness, tokenizer lengths, masks and chat templates before
loading the GPU model. Train only after those checks pass.

### Inference with a trained adapter

\`\`\`bash
PYTHONPATH=. python -m voice2reel.generate_character_spec \
    --prompt "A curious navy mascot, wide eyes, raised eyebrows, soft blush" \
    --adapter /path/to/qwen_character_spec_.../adapter \
    --spec-out /tmp/character_spec.json \
    --preview /tmp/character_preview.png
\`\`\`

It retries invalid generations with schema feedback and otherwise fails
clearly. It does **not** fall back silently to a fabricated or stale output.

### Video with Qwen's spec + your existing scenes

\`\`\`bash
PYTHONPATH=. python -m voice2reel.production_pipeline \
    --plan /path/to/scene.json \
    --lock /path/to/character_lock.json \
    --output /path/to/video.mp4 \
    --character-engine rig \
    --character-spec /path/to/character_spec.json
\`\`\`

The rendered video remains **1080×1920, 9:16**; the validated character
spec cannot change camera, animation rig, shot timings, text or narration.

### Honest limitations

- The synthetic dataset teaches JSON field mapping and design choices, **not
  aesthetic quality from real images**. The attractive geometry is fixed,
  authored vector art.
- This is not a diffusion model and does not learn to invent entirely
  new anatomy or arbitrary outfits. That constraint prevents malformed
  body parts.
- No actual GPU training is performed by committing these files. The user
  must run Colab training for the new CharacterSpec adapter.
- Model output quality must be inspected after training. The CPU tests
  validate syntax, style constraints, pose geometry, dataset uniqueness
  and rendering, not subjective art quality.
- The prior \`qwen_mascot_svg_...\` adapters are incompatible and should
  **not** be selected for CharacterSpec generation.
- Scene Qwen adapter and all existing voice settings remain separate.
