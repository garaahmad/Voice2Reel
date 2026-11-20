# Python motion-graphics code dataset (experimental)

This is a **separate task** from the existing Qwen narration-to-JSON scriptwriter. The latter expects `{id, idea, script}` records and cannot train on these code pairs without a new training adapter.

Generate deterministic instruction → complete executable Python pairs:

```bash
python scripts/build_python_motion_data.py
python scripts/build_python_motion_data.py --check
```

Default: 240 synthetic examples, 192 train / 24 validation / 24 test. Each example has `id`, `instruction`, `response` (complete Python code), and `metadata`. No external assets are required for the generated source except Pillow, ffmpeg and a font fallback. Rendering uses absolute time, bounded easing, simple cards/bars/steps and consistent vertical composition.

**Important limitations:** These examples are generated from a small family of templates. They are a starter curriculum, not 240 independent professional motion designs. Validation/test are distinct exact examples but **share the same templates** with train; therefore these splits measure interpolation, not unseen-code generalization. No training, video render, visual review, or real-world smoothness evaluation has been performed. Before training Qwen to write arbitrary Python, add hand-reviewed diverse programs, runtime checks in an isolated environment, output frame quality gates, and an independent held-out template family. Never execute model-generated code directly on production servers.

Synthetic dataset and generator: CC0-1.0. Other repository assets and dependencies retain their own licenses.
