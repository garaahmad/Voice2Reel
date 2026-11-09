# Model selection and source record

Reviewed 2026-11-06 using the model owners' cards, maintained upstream code, and official library documentation.

| Choice | Primary source | Decision |
|---|---|---|
| Qwen2.5-1.5B-Instruct | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Small instruction model; Arabic support and Apache-2.0. Selected for a free-Colab starting point, not claimed best-in-class. |
| QLoRA via PEFT | https://huggingface.co/docs/peft/developer_guides/quantization | NF4 base weights plus trainable LoRA adapters; reduced training memory. |
| XTTS-v2 | https://huggingface.co/coqui/XTTS-v2 | Arabic and reference-voice synthesis; established GPT fine-tuning path. |
| Maintained Coqui implementation | https://coqui-tts.readthedocs.io/en/latest/models/xtts.html | Use coqui-tts 0.28.0 rather than the old unmaintained TTS package. |
| XTTS training recipe | https://github.com/idiap/coqui-ai-TTS/blob/dev/recipes/ljspeech/xtts_v2/train_gpt_xtts.py | Reviewed GPT configuration, checkpoint loading, sample rates, and Trainer interfaces. Our batch/accumulation settings are smaller and require empirical evaluation. |
| XTTS weight license | https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt | CPML is not an unrestricted commercial model license; product deployment must respect its terms. |
| Qwen3-TTS alternative | https://github.com/QwenLM/Qwen3-TTS | Considered, but its documented original language set does not establish native Arabic support; community Arabic forks would require additional evaluation. |
| Colab limits | https://research.google.com/colaboratory/faq.html | GPU availability and session/resource limits vary; no runtime or training-time guarantee. |

Pinned model revisions:

- Qwen/Qwen2.5-1.5B-Instruct: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- coqui/XTTS-v2: `6c2b0d75eae4b7047358e3b6bd9325f857d43f77`

Dependencies pin the core API stack. The CUDA stack is torch/torchaudio 2.8.0 from the cu126 wheel index. There is no FlashAttention build, DeepSpeed, runtime monkeypatch, or automatic acceptance of model-license prompts. Additional dependencies are recorded by the environment freeze command in each notebook.

This project adapts existing pretrained models. It does not claim a novel foundation-model architecture, guaranteed factual correctness, perfect voice similarity, or benchmark superiority. Free Colab compatibility remains a target until the included GPU smoke runs pass on the user's allocated runtime.

## Preset voice extension

Reviewed the maintained [XTTS preset speaker API](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html). The pinned speakers_xtts.pth asset provides 58 named conditioning presets; both Arabic and English are documented languages. Presets use stored conditioning tensors, so the user does not supply audio and no fine-tuning is required. The inference download excludes training-only DVAE/mel-stat assets. Actual Arabic and English CPU generation is recorded in VALIDATION.md.

## Realistic video extension (2026-11-07)

Selected [LTX-Video-0.9.5](https://huggingface.co/Lightricks/LTX-Video-0.9.5), revision `e58e28c39631af4d1468ee57a853764e11c1d37e`, with Diffusers 0.35.2. Its 2B video transformer offers text and image conditioning through [LTXConditionPipeline](https://huggingface.co/docs/diffusers/v0.35.2/en/api/pipelines/ltx_video). This is a resource-conscious baseline, not a claim that it is the newest or strongest video model. LTX 0.9.7's 13B variants are larger; newer distilled variants require their own checkpoint-specific sampling settings. The selected non-distilled checkpoint uses 40 steps and guidance 3.0, not a distilled 4-step schedule.

The maintained official pipeline source was checked for image conditioning, precomputed prompt embeddings, argument order, and frame constraints. Actual offline CPU tests exercise both conditioning paths with tiny random LTX weights. The real video weights were not downloaded or run on the CPU-only host. T4 memory/speed and end-to-end visual quality are unmeasured.

## Director model upgrade (2026-11-08)

Earlier configurations with Qwen2.5 1.5B and 7B produced inadequate Arabic narration. Prompts and stage inputs also needed correction: narration now receives shot goals and the factual brief, with Arabic instructions for Arabic output. These were development experiments, not a controlled model benchmark.

The new default is [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its official card documents improved instruction following, multilingual capabilities, non-thinking operation, and Transformers >=4.51 compatibility. The project pins 4.57.3. The original 1.5B QLoRA training notebook remains separate.

CPU inference uses [Unsloth's GGUF conversion](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), revision `a06e946bb6b655725eafa393f4a9745d460374c9`, file `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, SHA-256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`. This is a community conversion of the official model, not a quantization published by Qwen. llama-cpp-python 0.3.16 uses JSON mode, seed 42, temperature 0.2, top-p 0.8. GPU inference uses Transformers NF4 with greedy decoding; output equivalence is not claimed.

[json-repair](https://github.com/mangiucugna/json_repair) repairs syntax only. Original/repaired outputs are retained, then schema and content checks run. Neither JSON mode nor model size guarantees factual accuracy.

## Fast presenter speech and rendering

The new fast path uses [Piper 1.3.0](https://github.com/OHF-Voice/piper1-gpl/tree/v1.3.0) and [its Python API](https://github.com/OHF-Voice/piper1-gpl/blob/v1.3.0/docs/API_PYTHON.md), with ONNX Runtime 1.23.2 CPU sessions limited to two intra-op threads and one inter-op thread by default. The renderer is local 2D drawing code with cached original artwork, not a video diffusion model.

Voice repository: `rhasspy/piper-voices`, revision `e7229f7d33a10e145e109b886d661984201ac3c4`. Selected assets:

- [Arabic Kareem medium model card](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/ar/ar_JO/kareem/medium/MODEL_CARD): Arabic (Jordan), one speaker, 22,050 Hz; dataset terms linked to the source project, not declared unrestricted here.
- [English Lessac medium model card](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/en/en_US/lessac/medium/MODEL_CARD): English (US), one speaker, 22,050 Hz; the Lessac Blizzard 2013 data has its own license terms.

Piper's maintained runtime is GPL-3.0. Voice/data rights are separate and require review. No voice weights or Instagram assets are redistributed in this repository. XTTS remains an optional slower cloning backend under its existing CPML constraints.


## Presenter v2 speech, typography and sound (2026-11-09)

The user rejected the first presenter/Piper result. The new default uses [edge-tts 7.3.1](https://pypi.org/project/edge-tts/7.3.1/) and the [maintainer's API](https://github.com/rany2/edge-tts). This is an unofficial client to Microsoft's **online** Edge speech service, not a local model upgrade or a project-trained voice. [Microsoft's voice documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts) describes locale-specific neural voices. Client license and service/output terms are separate. Real service generation was tested; long-term availability is not promised.

Word boundary events provide phrase timing for Edge output. No phoneme lip-sync is claimed. Original music and transition effects are synthesized in `audio_mix.py`; no external recordings were copied. Speech normalization, envelope-based music ducking and a final peak constraint keep the voice in front.

[Cairo](https://github.com/google/fonts/tree/main/ofl/cairo) supplies Arabic/Latin/punctuation in a single bundled font under SIL OFL 1.1. The original font and complete license ship with the package; hash/source are documented beside the asset.

The current Qwen3 4B writer remains limited in Arabic style. Actual failed model output is preserved, and script examples are marked as manually reviewed where applicable. Prompt changes, few-shot examples and validation are not presented as model training or proof of high writing quality.
