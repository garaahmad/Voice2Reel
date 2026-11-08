# Voice2Reel

Turn a **written idea** into an **animated stickman story** with expressive characters, flowing movement, purposeful camera work, motion graphics, and Arabic or English AI narration.

**Current direction: polished 2D stickman animation.** The realistic-video experiment is superseded. A reference video is optional as a creative reference; it is separate from an optional recording for voice cloning.

**Status: prototype.** Script/voice training and inference tools, a basic narrated stickman renderer, and a new silent motion study are implemented. The richer animation engine is not yet connected to AI-generated scene plans. Actual text-model tests exposed factual mistakes; generated narration still needs review. No custom-trained weights are shipped.

## The intended experience

1. Enter a written idea, optionally with a reference video.
2. Answer at most five questions: audience, duration, language, characters/setting/visual mood, and the main message or facts.
3. AI plans the story: scene composition, actions, interactions, expressions, camera movement, and timing.
4. AI writes narration aligned with the scene goals; review the content.
5. XTTS generates Arabic/English narration, with a preset voice or optional voice cloning.
6. The animation engine evaluates character rigs, scene objects and keyframes against actual speech timing, then exports MP4 with captions.

A stickman should walk into a scene, shift weight, reach for an object, react, and recover naturally. The target goes beyond changing a pose or moving a complete drawing up and down. Scenes, props, facial expressions and camera movement should carry the story together.

See the [Arabic animation direction and implementation notes](docs/STICKMAN_ANIMATION.md).

## See the current motion study

```bash
python -m pip install -r requirements/render.txt
python -m voice2reel.motion_preview --output outputs/stickman-motion-study.mp4
```

Requires ffmpeg and an Arabic-capable font. This CPU-rendered, silent nine-second study demonstrates walking, anticipation, reaching, prop contact, reaction, a jump, settling, facial animation, and a gentle camera push.

**The study is hand-authored engine code, not a model-generated story or finished narrated film.** The AI-to-animation integration, broader actions and scene transitions remain to be built. It uses a new clean visual layout; the old grid/card layout remains in the separate baseline renderer.

## Existing tools and Colab

Download this private repository with GitHub **Code → Download ZIP**. In Colab, use **File → Upload notebook**, then upload the source ZIP when asked. Results are saved to Drive.

| Goal | Notebook or module |
|---|---|
| Ready Arabic/English AI voice | [00_ready_voice_ar_en.ipynb](notebooks/00_ready_voice_ar_en.ipynb) |
| Scriptwriter QLoRA training baseline | [01_scriptwriter_qlora.ipynb](notebooks/01_scriptwriter_qlora.ipynb) |
| Optional voice cloning and fine-tuning | [02_arabic_voice_xtts.ipynb](notebooks/02_arabic_voice_xtts.ipynb) |
| New visual motion study | `python -m voice2reel.motion_preview` |
| Basic script + speech → stickman MP4 | [Baseline renderer guide](docs/VIDEO_DEMO.md) |

The 2D drawing engine can run on CPU. GPU can accelerate the AI components and is needed for the supplied training experiments. The full polished stickman workflow does not yet have an end-to-end Colab notebook.

## Implementation and limits

- `voice2reel/motion_preview.py`: new animation study with articulated limbs, two-bone inverse kinematics, gait, expressions, prop response and camera movement.
- `voice2reel/render.py`: existing basic renderer for validated script/audio timings. Its fixed visual vocabulary is a baseline, not the requested final animation engine.
- `voice2reel/scriptwriter/`, `voice2reel/voice/`: model adaptation and inference tools.
- `datasets/`: 40 authored script examples and 120 recording prompts; no recorded speech or production-sized dataset.
- `voice2reel/director.py`, `storyboard.py`, `realistic.py`, and notebook `03_realistic_video.ipynb`: earlier realistic-video experiment, retained for reference. Their current shot format is not an advanced stickman animation format.

Existing 1.5B adapters do not work with the separate Qwen3 4B experiment. JSON validation does not prove factual accuracy. See [verification](docs/VALIDATION.md), [dataset provenance](datasets/README.md), and [research/licensing](docs/RESEARCH.md). XTTS weights and outputs use the non-commercial CPML license.

The superseded realistic-video guide remains in [docs/REALISTIC_VIDEO.md](docs/REALISTIC_VIDEO.md). It is not the current product direction.
