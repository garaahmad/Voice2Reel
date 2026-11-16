# Character Lab / مختبر الشخصية

New entry point: [`notebooks/08_character_lab.ipynb`](../notebooks/08_character_lab.ipynb).
Old notebooks are unchanged. Run from a repository checkout with Python 3.11:

```bash
python -m pip install -r requirements/character-lab.txt
# Open notebooks/08_character_lab.ipynb in Jupyter or Colab.
```

## Choose → preview → confirm

12 IDs: `classic`, `bold`, `neon`, `robot`, `sketch`, `ribbon`, `minimal`,
`professor`, `astro`, `shadow`, `pixel`, `ink`. Bodies, heads, accessories,
palettes and joint styles vary. All share the same 18-joint interface.

Use the dropdown and **معاينة**, then **موافق: تثبيت الشكل**. Alternatively set
`SELECTED`, run its preview cell, then type that ID in `APPROVE_ID`.
Confirmation is intentionally not automatic. Subsequent cells require a lock.
The lock stores the ID, catalog digest and preview digest; a changed catalog or
preview requires renewed approval. This is a reproducibility guard, not a
security signature. There is **no user-selected character yet** in this delivery.

Catalog: [`examples/character_lab/gallery.png`](../examples/character_lab/gallery.png).
Pixel hashes distinguish all 12 complete preview cards; this includes labels and
backgrounds and is **not** a perceptual diversity or aesthetics score.

## Actual CPU measurement (2026-11-16)

Full measurements: [`reports/character_motion_cpu.json`](reports/character_motion_cpu.json).
Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 python -m voice2reel.motion_curriculum --epochs 60
```

720 train plans / 80 validation / 80 test / 80 extrapolation, 24 frames each.
Train families: wave, point, present, explain, step, breathe. Validation: nod.
Test: celebrate. Extrapolation: celebrate at larger knob values (not a third
independent held-out family). Validation selects the checkpoint; test does not.

| Split | Learned MAE px | Nearest training plan MAE px |
|---|---:|---:|
| Validation | 0.321 | 1.845 |
| Test | 0.562 | 3.351 |
| Extrapolation | 0.859 | 6.442 |

Test error is **5.96× lower** than nearest-plan lookup. Maximum coordinate error
on test is 9.292 px; mean error alone hides outliers. This is synthetic
compositional regression trained with Adam, **not a human-motion model**.
Teacher and learner share a harmonic inductive bias; held-out families do not
establish generalization to arbitrary choreography. No bone-length constraints,
foot-plant physics, independent artist ratings or speech synchronization are
claimed. Save weights and scores in `outputs/character_lab` from the notebook.

An intentionally difficult defect experiment reports train accuracy 1.0,
validation 0.9944, but **test 0.3444** on unseen flicker/abrupt defects. This is a
failed generalization result, retained rather than hidden. Do not use that
learned gate as a production quality certificate. More negative examples and
independent evaluation are required.

## Clip production

The trained learner drives the locked character through PIL → raw RGB → ffmpeg
H.264, vertical 540×960, yuv420p, with JSON sidecar. No generated Python is
executed. Joint safe-zone and maximum-step checks reject obvious bad tracks;
they do not inspect accessory bounds or certify anatomy/aesthetics. A 2-second,
48-frame encoding smoke test succeeded locally (44,413 bytes). Its temporary
`minimal` lock was only a test fixture, not a user choice. Preview outputs are
ignored under `outputs/`; no model weights/videos are committed.

This is a silent motion-preview pipeline, not a finished narrated reel.
Narration timing, captions, camera choreography and audio mux remain separate.

## Qwen / script work: implemented workflow, not measured improvement

The notebook calls the existing pinned Qwen2.5-1.5B QLoRA trainer with the existing
script corpus and answer-only loss. It adds a base-vs-adapter evaluation workflow
on identical test IDs, preserving raw outputs and validation errors. First-pass
validity is reported separately from correction attempts. Structured feedback
can repair schema errors; this is not proof of improved hidden reasoning.
Editorial checks include repeated/long captions, pose variety and speaking rate;
factuality, hooks, coherence and usefulness still need blind human ratings.

GPU training and inference were **not run here**: torch, GPU and model weights
are unavailable. Install `requirements/text.txt` in a compatible GPU runtime
with Hugging Face access, enable training explicitly, then evaluation. Do not
claim Qwen improvement until before/after results and human review exist.
The separate Python-code dataset is not mixed into narration training.

## Validation

- New character/motion/notebook tests: **4 passed**.
- `ruff check .`: passed; `scripts/check_data.py`: passed.
- Broader suite excluding training-smoke: 113 passed, 10 skipped, 19 failed
  in this environment (missing torch/TTS/huggingface_hub, system ffmpeg and
  other optional dependencies). The full suite is **not green**.
- The notebook is schema/AST validated with no saved cell outputs. Its CPU
  components were run separately; the interactive approval flow and GPU cells
  were not end-to-end browser/GPU tested.
