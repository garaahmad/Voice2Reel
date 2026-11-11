# Qwen → Python Stickman Animation: Implementation Plan

## Goal
Fine-tune a code-capable Qwen model to turn a natural-language animation brief into a **Python program using a stable, versioned Stickman animation engine**. This is distinct from the existing narration/scriptwriter Qwen and from the 60,000 structured drawing-plan records.

## Current repo baseline
- `datasets/stickman/drawing_*.jsonl`: 50k/5k/5k vector drawing plans, not Python code generation.
- `datasets/scripts/*.jsonl`: narration/scriptwriter records, not runnable animation programs.
- `voice2reel/stickman_artist.py`: joint predictor and defect detector, not a scene-programming API.
- User-supplied reel uses `reel_v2.py` as a rig; it is an example of a desired *output style*, not a self-contained training example.

## Stable engine contract (proposed; not yet implemented)
Create `voice2reel/animation_engine/` with a documented, versioned Python API:
- `Scene(width=540, height=960, fps=30, duration=10)`
- `scene.add_stickman(name="hero") -> Actor`
- `actor.animate(action, start, end, **parameters)` where actions are validated against the engine registry
- `scene.camera.animate(kind, start, end, **parameters)`
- `scene.add_caption(text, start, end, highlight=None)`
- `scene.add_effect(name, at, **parameters)`
- `scene.render(path)`
The API above is a *design target*, not an existing import. Do not train on it as executable code until the implementation and smoke tests pass.

## Milestones
1. **Engine MVP**: implement deterministic pose/keyframe timeline, safe-area constraints, scene/camera/effect registry, Pillow frame output and optional FFmpeg video export. Provide a 1-second CPU smoke render.
2. **Gold examples**: hand-author and run 50–100 diverse examples with the real engine. Include animation briefs, valid scripts, expected frame/video checks, runtime dependencies, licenses, and source provenance.
3. **Dataset generator**: compose controlled variations only from validated gold templates, with explicit action/parameter combinations. Generate JSONL chat records, grouped train/validation/test by template/story family to prevent leakage. Never claim synthetic examples are human reviewed.
4. **Evaluation harness**: AST parse; allowed-import/static safety; sandboxed execution with timeout and resource limits; render success; duration/fps/size; frame sampling; actor safe-zone and joint connectivity; motion continuity; captions and camera timing. Run untrusted model output in an isolated container with network off, not directly in Colab host.
5. **Baseline before LoRA**: test the chosen Qwen *Coder* instruct checkpoint with the engine documentation in the prompt. Compare base vs LoRA on a frozen held-out set. Report syntax validity, API validity, render success, and quality checks independently.
6. **LoRA**: supervised fine-tuning on assistant code completions, with deterministic prompts, pinned engine version, reproducible seed and saved training configuration. Train/validation only; hold out test until final evaluation.
7. **Error-repair dataset**: only add failing programs paired with verified corrections and error logs. Do not treat auto-generated repairs as ground truth without execution.
8. **Production**: brief → code generation → AST/API validation → isolated render → quality gate → MP4 or actionable error. Retain code and render diagnostics for iterative improvement.

## Dataset format
Use `datasets/stickman_code/{train,validation,test}.jsonl` as future materialized output, with each row:
```json
{
  "id": "stickcode_000001",
  "family_id": "debug_overflow",
  "engine_version": "0.1.0",
  "provenance": "human-authored-and-render-verified",
  "license": "CC0-1.0",
  "messages": [
    {"role": "system", "content": "Use only the documented Voice2Reel animation engine API."},
    {"role": "user", "content": "Animate a stickman noticing a layout error and celebrating a fix."},
    {"role": "assistant", "content": "from voice2reel.animation_engine import Scene\n..."}
  ]
}
```
Do not fabricate verification or license status: set `provenance` and `license` according to actual review/rights. Store engine version and group family IDs. Training-only metadata must be excluded from the assistant completion.

## Suggested coverage targets (planning only)
- 2,000 poses, gestures, movement and timing
- 2,000 interactions and reactions
- 1,500 camera movement and transitions
- 1,500 captions, UI elements and effects
- 2,000 multi-scene stories
- 1,000 error correction cases
Total **10,000 aspirational records**, contingent on validated engine and examples. Start with 50–100 verified gold examples and 500–1,000 carefully validated variations.

## Acceptance criteria
- Engine API and documentation exist; pinned version; testable actions.
- At least 50 genuinely render-verified, diverse gold programs before scaling.
- Every split row parses and uses allowed engine API; no family overlap across splits.
- Generated scripts render under isolation with explicit resource/time limits.
- Baseline and fine-tuned Qwen evaluated on the same untouched test set.
- No claims of training success, aesthetic quality, or runnable examples without actual evidence.

## Next implementation tasks
1. Build the real `animation_engine` module, reusing deterministic stickman geometry rather than reimplementing the drawing model.
2. Add CPU render smoke tests and safe executor.
3. Write initial verified gold examples and an exporter with family-level splits.
4. Create `notebooks/08_qwen_stickman_code_lora.ipynb` **after** API and dataset are validated.
