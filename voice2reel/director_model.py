"""Stronger production experiment, separate from the small QLoRA training baseline."""

from voice2reel.common import digest_file

DIRECTOR_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DIRECTOR_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
GGUF_MODEL = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
GGUF_REVISION = "a06e946bb6b655725eafa393f4a9745d460374c9"
GGUF_FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
GGUF_SHA256 = "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"


class GGUFDirector:
    def __init__(self, model_path=None, threads=2):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        if not 1 <= threads <= 64:
            raise ValueError("threads must be 1–64")
        if model_path is None:
            model_path = hf_hub_download(GGUF_MODEL, GGUF_FILENAME, revision=GGUF_REVISION)
        if digest_file(model_path) != GGUF_SHA256:
            raise ValueError("GGUF checksum differs from the pinned Qwen3 Q4_K_M checkpoint")
        self.model_id, self.revision, self.backend = GGUF_MODEL, GGUF_REVISION, "llama.cpp-Q4_K_M"
        self.model = Llama(
            model_path=str(model_path), n_ctx=8192, n_threads=threads, n_threads_batch=threads,
            n_gpu_layers=0, chat_format="chatml", verbose=False, seed=42,
        )

    def complete(self, messages, max_tokens=1200):
        result = self.model.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.2, top_p=0.8,
            response_format={"type": "json_object"},
        )
        return result["choices"][0]["message"]["content"]
