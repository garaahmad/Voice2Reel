# Mascot V1 — mascot-first implementation (A)

This implementation **replaces the legacy hand-drawn joint lines in the default video path**. The main character is an original, authored SVG with nested rotations. Qwen still writes shot/gesture instructions; it does not regenerate the anatomy each frame.

## The 13 delivery stages

| Stage | What is implemented | Location |
|---|---|---|
| 1. Choose a single mascot | One original clean explainer identity, no hat | `assets/mascot_v1/base.svg` |
| 2. Design its appearance | Large face, expressive eyes, rounded arms/hands/shoes | `base.svg` |
| 3. Separate parts | 14 nested head/torso/arm/hand/leg/foot groups | `base.svg` |
| 4. Create the rig | Exact parent links, pivots, rotation limits | `rig.json` |
| 5. Author gesture poses | idle, present, point, wave, think, explain, celebrate, step | `poses.json` |
| 6. Validate the skeleton | Check SVG groups and hierarchy against rig config | `voice2reel/mascot_rig.py` |
| 7. Animate the pose | Angles interpolated with ease-in/out and small head motion | `voice2reel/mascot_animator.py` |
| 8. Render SVG frames | Vector artwork rasterized with CairoSVG | `mascot_animator.py` |
| 9. Compose into scene | Alpha-composite the animated character without old joint drawing | `voice2reel/mascot_renderer.py` |
| 10. Use in videos | Default `--character-engine rig` in production pipeline | `voice2reel/production_pipeline.py` |
| 11. Update notebook | Preview and select rig by default, leave training optional | `notebooks/10_ai_story_director_all_in_one.ipynb` |
| 12. Test poses and reels | pytest, visual pose sheet, 1080x1920 MP4 smoke script | `tests/test_mascot_v1.py`, `scripts/smoke_mascot_v1_reels.py` |
| 13. Review / release gates | GitHub Actions for regression and video export validation | `.github/workflows/mascot-v1.yml` |

## Quick start

On a machine with Python 3.11–3.13:

```bash
python -m pip install -r requirements/character-lab.txt
PYTHONPATH=. python -m scripts.preview_mascot_v1 --output mascot-v1.png
PYTHONPATH=. python -m pytest -q tests/test_mascot_v1.py
PYTHONPATH=. python -m scripts.smoke_mascot_v1_reels --output mascot-v1-smoke.mp4
```

Google Colab: open `notebooks/10_ai_story_director_all_in_one.ipynb`, run settings, Setup and the **Preview mascot_v1 poses** cell. Keep `CHARACTER_ENGINE="rig"` and `RUN_TRAINING=False` to reuse an existing scene adapter. Run story, voice, and Reels render cells only after accepting the visual preview.

## Modes

- **rig (default):** original mascot artwork animated by nested SVG transforms; no Qwen GPU required for the character.
- **qwen_static (optional):** PNG generated using the *separate* Qwen-to-SVG adapter. This drawing is a static overlay with a small bob, **not** a controllable skeletal rig.
- **legacy (optional):** older Pillow stickman character renderer for comparison only.

The `qwen_static` and `rig` modes are mutually exclusive. **Qwen story generation is retained** in all modes. A future Qwen-to-rig model would need to output the exact part hierarchy/pivot contract, not merely one finished SVG.

## Acceptance criteria and limitations

- Animation is geometric 2D rig animation, not frame-by-frame 3D or generative animation.
- All arms are attached through nested transforms around authored shoulder/elbow pivots.
- Pose transitions ease back to idle at shot boundaries. No old `POSE_OFFSETS` or line-by-line renderer is used in rig mode.
- Reels video export requires **1080×1920, 9:16**; production verifies the encoded MP4 dimensions.
- Current facial expressions are pre-authored SVG (one happy face), not lip-synced; sophisticated expressive lip-sync and advanced motion graphics remain separate work.
- Qwen training, Edge TTS and rendering are not run in GitHub tests; only the local/CI smoke video runs without model or voice downloads.
- The preview tests prove geometry, code integration and format, not subjective artistic acceptance. Visual feedback is still important.
