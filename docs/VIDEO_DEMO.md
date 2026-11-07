# Run the models and render a stickman video

The demo pipeline executes the repository's pinned **Qwen2.5-1.5B-Instruct** model to write the script and **XTTS-v2** to synthesize Arabic narration. It then draws a stick figure using a deterministic Pillow renderer and encodes MP4 with ffmpeg. The renderer is code, not a video diffusion model. No extra text or voice model is used to create the content.

No project-trained adapter exists yet; the demo uses the original pretrained weights. This tests the base-model pipeline, not a fine-tuned Voice2Reel model's quality.

Install the existing text/voice environments plus `requirements/render.txt`. ffmpeg and an Arabic-capable font are required. Pillow must support RAQM; the bundled Python wheel does, and Debian's DejaVu font supports both Arabic and Latin text.

```bash
# GPU: automatic QLoRA-compatible loading; no adapter needed for the base-model test.
python -m voice2reel.scriptwriter.generate --idea "فيديو عربي عن أخطاء الفاصلة المنقوطة أثناء تعلم البرمجة" --duration 30 --output outputs/demo/script.json

# Preset AI voice; language is inferred from script.json.
python -m voice2reel.voice.synthesize --script outputs/demo/script.json --output outputs/demo/narration.wav

# Fixed stickman animation, model-generated title/captions, and actual audio timing.
python -m voice2reel.render --script outputs/demo/script.json --audio outputs/demo/narration.wav --timing outputs/demo/narration.json --output outputs/demo/video.mp4
```

For a CPU-only machine, script generation supports `--device cpu --cpu-dtype float32`. A CPU with native BF16 support can use `--cpu-dtype bfloat16` to reduce memory and improve throughput. This is inference only; QLoRA training still requires CUDA. Run the language and voice models in separate processes so their weights do not occupy RAM together.

The renderer accepts `--width 720` for a smaller preview and `--fps 30`. Default output is 1080×1920 at 30 FPS, H.264 video with AAC audio. It preserves the complete narration duration, even when that differs from the script's target duration. Mismatched script/audio text, language, duration, or timing is rejected before rendering.

Current renderer scope: one fixed coder/stickman scene, model-selected poses, title, captions, movement, and a punctuation prop. It is a first runnable renderer, not the complete visual engine in the product design. It does not invent code examples, correct model facts, align individual words, or generate the drawings with an AI image model. Review actual generated content before using it as educational material.

## Raw-model result

The semicolon test deliberately keeps Qwen's narration unedited. Its final beat incorrectly implies that using semicolons in a for loop is itself an error; in C++, the separators in `for (initialization; condition; update)` are required. This sample tests whether the pipeline runs and exposes the small model's limitations; it is not a reviewed lesson. A production system needs content review and better training/evaluation data, not just JSON validation.
