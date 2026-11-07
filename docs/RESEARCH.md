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
