# Verification record

Checked on 2026-11-06 in Linux, Python 3.12.14, CPU only.

| Check | Result |
|---|---|
| pytest | 21 passed |
| Ruff | Passed |
| Python module compilation | Passed |
| Both notebooks | nbformat validation, Python cell syntax, and empty-output checks passed |
| CLI entry points | All six --help commands passed |
| Script data | 30/5/5 records validated; no cross-split ID, idea, or narration duplicates |
| Actual pinned Qwen tokenizer | All 40 examples encoded without truncation; 670–700 tokens, including system prompt and answer |
| Actual pinned XTTS tokenizer | All 120 recording prompts encoded; 40–59 tokens, below the 200-token training limit |
| Real CPU LoRA smoke | Tiny randomly initialized Qwen2: two optimizer steps, finite loss, adapter parameter updates, save/reload, matching reloaded logits |
| Installed Coqui API | XTTS training config and resume arguments constructed and validated |
| Voice integration | Generated audio fixtures -> preparation -> upstream Coqui metadata loader -> actual XTTSDataset -> collated batch succeeded |

The tiny-model test validates the training machinery; it is not training or evaluating the 1.5B model. Audio test fixtures are artificial tones solely for pipeline testing, not speech training or evidence of voice quality.

## Not yet measured

- Full QLoRA training or inference with the 1.5B model on CUDA/T4.
- XTTS model-weight loading, forward/backward training, or speech synthesis with actual weights on GPU.
- End-to-end notebook execution inside a real Colab allocation.
- Arabic pronunciation, dialect handling, voice similarity, or model factual accuracy after adaptation.
- Peak GPU memory, training duration, or production quality scores.

A GPU was not available in the execution environment, and no user speech recordings were supplied. The notebooks include GPU checks and a short first training run so these can be verified before longer experiments. Real data and held-out comparisons are required before accepting an adapted model.

Core local versions: torch/torchaudio 2.8.0+cpu, transformers 4.57.3, PEFT 0.17.1, accelerate 1.10.1, coqui-tts 0.28.0, coqui-tts-trainer 0.4.0, Pydantic 2.11.7, NumPy 2.2.6, SciPy 1.15.3, soundfile 0.13.1, librosa 0.11.0. GPU notebook wheels use torch/torchaudio 2.8.0 cu126.
