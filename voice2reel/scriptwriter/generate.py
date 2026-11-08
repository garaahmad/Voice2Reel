import argparse
import json
from pathlib import Path

from voice2reel.common import write_json
from voice2reel.scriptwriter.schema import messages_for, parse_script
from voice2reel.scriptwriter.train import MODEL_ID, MODEL_REVISION


class ScriptGenerator:
    def __init__(self, adapter=None, device="auto", cpu_dtype="float32", *,
                 model_id=MODEL_ID, model_revision=MODEL_REVISION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if device not in ("auto", "cpu", "cuda") or cpu_dtype not in ("float32", "bfloat16"):
            raise ValueError("Unsupported device or CPU dtype")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        options = {"device_map": {"": device}, "torch_dtype": getattr(torch, cpu_dtype)}
        if device == "cuda":
            options.update(
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
                ),
            )
        self.model_id, self.revision = model_id, model_revision
        self.backend = "transformers"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            **options,
        )
        if adapter:
            from peft import PeftModel

            provenance = json.loads((Path(adapter) / "provenance.json").read_text())
            if provenance["base_model"] != model_id or provenance["revision"] != model_revision:
                raise ValueError("Adapter base model/revision mismatch")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def complete(self, messages, max_tokens=1200):
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            result = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, idea, duration=30, attempts=3, language="ar", trace_path=None):
        import torch

        if not 1 <= attempts <= 3:
            raise ValueError("attempts must be 1–3")
        messages = messages_for(idea, duration, language)
        last_error = ""
        trace = []
        for attempt in range(attempts):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=1200, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
                )
            raw = self.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            trace.append({"attempt": attempt + 1, "raw_response": raw})
            if trace_path:
                write_json(trace_path, trace)
            try:
                return parse_script(raw, duration, language), attempt + 1
            except ValueError as exc:
                last_error = str(exc)
                trace[-1]["error"] = last_error
                if trace_path:
                    write_json(trace_path, trace)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Fix the JSON, schema, duration, and language. Error: "
                                if language == "en"
                                else "صحح JSON والتزم بالمخطط واللغة والمدة. الخطأ: "
                            )
                            + last_error[:700],
                        },
                    ]
                )
        raise ValueError(f"No valid script after {attempts} attempts: {last_error}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--language", choices=["ar", "en"], default="ar")
    p.add_argument("--adapter")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--output", default="outputs/script.json")
    args = p.parse_args()
    script, attempts = ScriptGenerator(args.adapter, args.device, args.cpu_dtype).generate(
        args.idea,
        args.duration,
        language=args.language,
        trace_path=Path(args.output).with_suffix(".trace.json"),
    )
    write_json(args.output, script.model_dump())
    Path(args.output).with_suffix(".txt").write_text(script.narration(), encoding="utf-8")
    print(f"Validated script written after {attempts} attempt(s): {args.output}")


if __name__ == "__main__":
    main()
