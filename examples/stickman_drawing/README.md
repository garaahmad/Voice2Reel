# Large Stickman drawing measurement

This folder documents the **60,000-row** synthetic vector drawing corpus in [`datasets/stickman`](../../datasets/stickman). It is generated and measured by:

```bash
python scripts/build_stickman_drawing_data.py --check
python -m voice2reel.stickman_artist \
  --output examples/stickman_drawing/artist_rounds.json \
  --render-samples examples/stickman_drawing/samples
```

`artist_rounds.json` is a **real CPU measurement**, not a hand-written claim. Three documented rounds increase training rows and fitted features:

| Round | Train rows | Features | Validation score |
|---|---:|---:|---:|
| 1 | 8,000 | 5 | 0.7615 |
| 2 | 25,000 | 10 | 0.8698 |
| 3 (best) | 50,000 | 14 | **0.9304** |

Final held-out test split, evaluated once with round-3 settings: **0.9299**. Breakdown: defect-detector accuracy `1.0`, mean joint error `15.021 px`, artist score `0.7997`, professional clean rate `1.0`. Composite = `0.55 × defect accuracy + 0.35 × (1 − joint error / 75) + 0.10 × professional clean rate`.

## Sample frames

All PNGs are deterministic vector renders from the corpus, produced by `voice2reel/stickman_artist.py`; they are **not** AI-generated/diffusion artwork.

| File | Label | What it shows |
|---|---|---|
| `sample_00_present.png` | professional | connected head/neck/torso, present gesture |
| `sample_01_point.png` | professional | point gesture |
| `sample_02_wave.png` | professional | wave gesture |
| `sample_03_disconnected_joint.png` | `disconnected_joint` | broken elbow/lower-arm gap |
| `sample_04_bad_proportions.png` | `bad_proportions` | oversized head and wrong bone lengths |
| `sample_05_offscreen_actor.png` | `offscreen_actor` | actor pushed outside the canvas |
| `sample_06_cropped_head.png` | `cropped_head` | head detached into the headline zone |
| `sample_07_overlap.png` | `overlap` | actor overlaps the information panel |

`samples/samples.json` lists the same metadata.

## Limits

- Assistant-authored synthetic CC0-1.0 vector plans, with **no expert review** and no scraped artwork.
- The fitted models are small CPU ridge/max-margin models. They improve joint prediction and defect detection **on these authored splits**; they do not change `Qwen` weights, the production `Qwen3-4B` writer, voice, or video quality.
- `0.9299` is not a subjective professional-aesthetics score. A neural image generator would need a separately licensed raster dataset and GPU training.
- The QLoRA writer loop and `final_report.json` still require the user's Colab T4 GPU in `notebooks/05_quality_loop.ipynb`.
- For the drawing-data/demo path, run `notebooks/06_stickman_generator.ipynb`: it validates the corpus, fits the CPU gate, trains a PyTorch MLP joint generator, renders 240 frames, assembles `outputs/stickman_pipeline/stickman_demo.mp4`, and writes `stickman_pipeline_report.json`. The video is a deterministic vector demo with silent audio by default; optional `edge-tts` is not cleared for commercial use. It does not train `Qwen`.
- For the **full production studio**, run `notebooks/07_stickman_full_production.ipynb`: it trains Qwen + MLP once, saves `model_training_results.json`, then reuses it to generate content, silent video, audio and a synced final MP4 per idea without retraining. It still does not replace notebook 05 / `final_report.json`.

See [`docs/QUALITY_V2.md`](../../docs/QUALITY_V2.md) for the full quality report and limitations.
