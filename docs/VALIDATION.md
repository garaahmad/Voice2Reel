# Verification record

Checked on 2026-11-06 in Linux, Python 3.12.14, CPU only.

| Check | Result |
|---|---|
| pytest | 40 passed |
| Ruff | Passed |
| Python module compilation | Passed |
| All three notebooks | nbformat validation, Python cell syntax, and empty-output checks passed |
| CLI entry points | All six --help commands passed |
| Script data | 30/5/5 records validated; no cross-split ID, idea, or narration duplicates |
| Actual pinned Qwen tokenizer | All 40 examples encoded without truncation; 672–702 tokens, including system prompt and answer |
| Actual pinned XTTS tokenizer | All 120 recording prompts encoded; 40–59 tokens, below the 200-token training limit |
| Real CPU LoRA smoke | Tiny randomly initialized Qwen2: two optimizer steps, finite loss, adapter parameter updates, save/reload, matching reloaded logits |
| Installed Coqui API | XTTS training config and resume arguments constructed and validated |
| Preset voice inference | Actual pinned XTTS weights loaded on CPU; Arabic and English WAV generated without reference audio |
| Preset/language routing | Default voice, optional cloning, explicit language, script-language inference, invalid selections, and metadata inputs covered by tests |
| Voice integration | Generated audio fixtures -> preparation -> upstream Coqui metadata loader -> actual XTTSDataset -> collated batch succeeded |

The tiny-model test validates the training machinery; it is not training or evaluating the 1.5B model. Audio test fixtures are artificial tones solely for pipeline testing, not speech training or evidence of voice quality.

## Not yet measured

- Full QLoRA training or inference with the 1.5B model on CUDA/T4.
- XTTS model-weight loading, forward/backward training, or speech synthesis with actual weights on GPU.
- End-to-end notebook execution inside a real Colab allocation.
- Arabic pronunciation, dialect handling, voice similarity, or model factual accuracy after adaptation.
- Peak GPU memory, training duration, or production quality scores.

A GPU was not available in the execution environment, and no user speech recordings were supplied. Preset inference was exercised with actual pretrained weights on CPU; optional voice-cloning synthesis was not evaluated with a real reference recording. The notebooks include GPU checks and a short first training run so these can be verified before longer experiments. Real data and held-out comparisons are required before accepting an adapted model.

Core local versions: torch/torchaudio 2.8.0+cpu, transformers 4.57.3, PEFT 0.17.1, accelerate 1.10.1, coqui-tts 0.28.0, coqui-tts-trainer 0.4.0, Pydantic 2.11.7, NumPy 2.2.6, SciPy 1.15.3, soundfile 0.13.1, librosa 0.11.0. GPU notebook wheels use torch/torchaudio 2.8.0 cu126.

## Actual preset-voice smoke run

Using the pinned XTTS-v2 weights, Claribel Dervla, seed 42, CPU with two OpenMP/MKL threads:

| Language | Output duration | Generation wall time | Output |
|---|---|---|---|
| Arabic | 5.568 s | 17.32 s | 133632 finite, nonzero samples at 24 kHz |
| English | 3.755 s | 11.37 s | 90112 finite, nonzero samples at 24 kHz |

Model loading took approximately 14.5 seconds in this run. These are two short functional checks, not an Arabic pronunciation benchmark or a speed guarantee. Audio clarity still needs listening evaluation with representative text. A regression test covers the upstream XTTS eval() method returning None, which previously broke chained model loading.

## End-to-end raw-model video experiment (2026-11-07)

The pinned Qwen model was actually executed on CPU with BF16 and two threads (container quota: two CPU cores, 8 GiB RAM). The original parser rejected three outputs because the model wrapped JSON in Markdown fences. The parser now unwraps a whole-response JSON code fence before applying the same schema checks; it does not edit narration. A subsequent single attempt succeeded in 99.44 seconds including model loading, with about 3472 MiB peak RSS. The raw response is preserved alongside the demo.

The result reveals a content-quality limitation: the final beat frames semicolons in a for loop as a mistake without explaining their valid roles. The narration is deliberately retained unedited for a faithful raw-model evaluation; it must not be treated as reviewed programming instruction. Schema-valid output is not necessarily factually sound.

A real miniature H.264/AAC encode test checks the new renderer's output streams, dimensions, and duration. Rendering rejects mismatched script text, language, segment timing, and audio duration.

The complete raw-model demonstration produced 28.672 seconds of Arabic narration and a 28.700-second MP4, 1080×1920, 30 FPS, H.264 video plus AAC at 24 kHz. Full ffmpeg decoding succeeded. Frames from all four beats were inspected for Arabic shaping and layout. The narration and script were not rewritten by the assistant. The drawings are produced by the fixed code renderer. No GPU was used.

## Realistic workflow implementation (2026-11-07)

- Local suite at this stage: **53 passed**; Ruff, module compilation, and dataset validation passed.
- All **four** notebooks pass schema/syntax checks and contain no stored outputs.
- Timeline tests cover audio-derived frame bounds, splitting long speech, LTX's 8n+1 frame constraint, stale narration, and mismatched scene counts.
- A real ffmpeg test assembles three artificial color clips with Arabic captions and a tone into a 36-frame H.264/AAC MP4. This validates media assembly, not AI visual quality. The Arabic caption layout was inspected separately.
- Actual Diffusers 0.35.2 LTXConditionPipeline calls with tiny random LTX weights pass for text conditioning and image conditioning. No pretrained LTX weights are used in those tests.
- A separate tiny T5 test exercises the encoder-only pipeline with the video components absent, including embedding/mask order and shapes.
- CUDA absence is detected before video-weight downloads. This environment has two CPU cores, 8 GiB RAM, and no GPU.
- The first real Qwen director experiment produced Arabic visual prompts and one Portuguese caption despite the requested languages. The implementation now validates Latin-script visual prompts and rejects Arabic narration/captions without Arabic letters. These checks catch the observed failures; they are not comprehensive language identification or factual verification.
- Additional raw-model attempts produced malformed JSON. Syntax repair is now explicit and logged alongside the original response, followed by unchanged schema/language validation. A repair test verifies that text is preserved and missing required scenes still fail validation. Constrained decoding was tried but rejected after observed multilingual failures; it is not used in the shipped pipeline.

Full pretrained LTX inference, GPU NF4 T5 loading, free-Colab memory fit, end-to-end notebook execution, and realistic scene quality remain **unverified**. No realistic-video sample is claimed from this host. The existing stickman video remains the only complete pretrained-model demonstration generated here.

## Director quality checks and model selection (2026-11-08)

The 1.5B and 7B experiments passed some structural checks but failed content review: narration described a person's actions, mixed in unwanted language or programming placeholders, or gave weak explanations. Those runs are not accepted demonstrations. Model size alone did not solve the problem. Narration now receives the factual brief and normalized narrative roles, with Arabic instructions for Arabic output, instead of the full visual descriptions.

The current experimental default is Qwen3-4B-Instruct-2507. GPU inference uses the pinned official checkpoint; CPU inference uses Unsloth's pinned Q4_K_M conversion with a verified checksum. The separate 1.5B training baseline is preserved.

Current local suite: **59 passed**, plus Ruff and notebook checks. Regression checks cover source-code requests, internal camera cuts, mixed-script captions, code read aloud, localized titles, and director-to-narrator data routing. These are content constraints, not a factual-accuracy benchmark. GPU/NF4 execution remains unverified.

The CPU llama-cpp-python 0.3.16 runtime was built from source using GCC/G++. Prebuilt Linux wheels were incompatible with this host's C library. No GPU was used.

A subsequent real Qwen3 run exposed factual contamination from free-form shot purposes. The compiler now replaces these with neutral narrative roles before the writer runs; only brief.message supplies factual content. Raw director responses remain in the trace. This constraint does not guarantee that the writer will always follow the brief.

Manual review of the first completed Qwen3 script also rejected its factual wording. A follow-up source-grounded pass still confused the condition with its body. The project remains experimental; successful schema checks are not an acceptance criterion for educational accuracy. Review feedback is supplied to the same project model for correction, with original responses preserved.

The final focused sample was also rejected: it described the condition as independent rather than distinguishing the empty body from the following independent block. See examples/model_quality_review. No new narration/video was produced from that rejected sample. The content-quality goal remains unmet.
