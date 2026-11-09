# Dataset provenance

**All generated v2 examples are assistant-authored synthetic material (CC0-1.0), composed from explicit banks in `scripts/build_datasets.py`.** They were not scraped, not labelled by a measured teacher model, not recorded by voice actors, and have not received independent expert review. Bigger data here means broader controlled coverage; it does not by itself establish real-world accuracy, natural speech, or subjective video quality.

## Starter v1 records (immutable seed)

- `scripts/seed_records.jsonl`: byte-preserved snapshot of the original 30/5/5 records.
- Those examples use Modern Standard Arabic, 20-second targets and three narration beats; they are a small end-to-end experiment, not the full supported schema.
- `voice/recording_prompts.csv`, `voice/manifest.template.jsonl`, `voice/evaluation_texts.txt` remain from v1. **No WAV files are included.**

## Generated v2 splits (run `python scripts/build_datasets.py`)

| Area | Train | Validation | Test | File prefix |
|---|---:|---:|---:|---|
| Full tashkeel | 480 | 60 | 60 | `arabic/tashkeel_*.jsonl` |
| Arabic + technical Latin terms | 620 | 70 | 70 | `arabic/codeswitch_*.jsonl` |
| Motion comfort plans | 960 | 120 | 120 | `motion/motion_plans_*.jsonl` |
| Scriptwriter records | 930 | 95 | 95 | `scripts/{train,validation,test}.jsonl` |
| Stickman vector drawing plans | 50,000 | 5,000 | 5,000 | `stickman/drawing_*.jsonl` |

Additional generated assets:

- `arabic/lexicon.json` and mirrored `voice2reel/assets/arabic_lexicon.json`: 95 registered technical surfaces with Arabic speech readings, glosses, categories and `allowed_in_narration` flags. Bare identifiers (`if`, `for`, `console`, …) are forbidden in narration.
- `motion/vocabulary.json` and mirrored `voice2reel/assets/motion_vocabulary.json`: bounded gestures/cameras/easings, comfort limits and safe zones.
- `motion/scene_annotations.jsonl`: 3240 structured scene plans.
- `scripts/v2_records.jsonl`: the authored append-only script rows; splits are rebuilt from seed + v2 rows.
- `voice/recording_prompts_v2.csv`, `voice/manifest.v2.template.jsonl`, `voice/evaluation_texts_v2.txt`: 140 recording prompts, still **without audio**.
- `build_manifest.json`: seed, counts, provenance and license.
- `stickman/vocabulary.json` and mirrored `voice2reel/assets/stickman_drawing_vocabulary.json`: 6 gestures, 3 cameras with zoom factors, 5 emotions, 3 intents, 5 visuals, 20 Arabic captions, 13 named defects, 18 keypoints, 14 features, style/limits/safe zones/render order.
- `stickman/build_manifest.json`: counts, seed, provenance and `CC0-1.0` license for the 60,000 drawing plans.

Design decisions:

- Tashkeel examples cover fatha, damma, kasra, sukun, shadda, tanween and hamza/tatweel cases; `voice2reel/arabic_text.py` measures exact per-letter mark agreement.
- Code-switch examples keep the Latin surface in **display** text but store the Arabic-only `spoken` reading for TTS. Unregistered Latin terms are rejected, so literal code identifiers never reach the speech engine.
- Motion examples are balanced 50/50 comfortable/annoying; each negative row violates exactly one named rule. The fitted `ComfortClassifier` learns thresholds from train rows, not hard-coded labels.
- Splits are disjoint by ID and text; `validate_splits` still checks idea/narration leakage for scriptwriter rows. Extra vocabulary is interleaved across all splits so final test is as hard as validation.
- Voice prompts are recording scripts, not a recorded corpus. Any real-audio dataset must document source, speaker composition, consent, license and held-out policy before being mixed in.
- Stickman rows are **structured vector plans**, not raster artwork: 18 2D joints from two-bone IK, per-row proportions/stroke/camera scale, actor safe box, 14 numeric features, and one named defect on negative rows. Professional rows keep the head, neck, torso, arms and feet connected and stay inside authored safe zones. The 13 defects cover camera jumps, gesture flicker, jitter, cuts, flashes, overlap, dense subtitles, abrupt easing, disconnected joints, bad proportions, offscreen actors, cropped heads and lifted feet. This teaches the deterministic renderer and a small CPU defect detector; it is **not** a neural image-generator dataset and has **no expert aesthetic review**.

## Verification

```bash
python scripts/build_datasets.py --check
python scripts/build_stickman_drawing_data.py --check
python scripts/check_data.py
python -m voice2reel.quality_gate --output examples/quality_v2/rounds.json
python -m voice2reel.stickman_artist --output examples/stickman_drawing/artist_rounds.json
```

`scripts/check_data.py` validates script schema, disjoint splits, unique IDs/texts, mark coverage, manifest alignment, asset mirrors, byte reproducibility for both generators, all 40 Stickman integrity checks, and recomputes the committed CPU measurement report. `voice2reel/quality_gate.py` fits three front-end models on CPU; `voice2reel/stickman_artist.py` fits a joint predictor and a learned defect detector on the 60,000 drawing plans. Both report validation/test metrics. The Qwen scriptwriter is trained on the user's Colab GPU in `notebooks/05_quality_loop.ipynb`. For the Stickman path there is **one** end-to-end notebook, `06_stickman_generator.ipynb`: data check → CPU gate → PyTorch joint generator → deterministic MP4 video → `stickman_pipeline_report.json`. It does not replace the GPU writer notebook. See `docs/QUALITY_V2.md` for the measured numbers and their limits. Artificial tones in unit tests are fixtures only and are never training data.

Generated examples and recording prompts in this folder are offered under CC0-1.0: https://creativecommons.org/publicdomain/zero/1.0/. Model and upstream software licenses remain separate.
