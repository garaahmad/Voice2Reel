import argparse
import json
from pathlib import Path

from voice2reel.common import write_json
from voice2reel.scriptwriter.schema import messages_for, parse_script
from voice2reel.scriptwriter.train import MODEL_ID, MODEL_REVISION


class ScriptGenerator:
    def __init__(self, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            device_map={"": 0},
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            ),
        )
        if adapter:
            from peft import PeftModel

            provenance = json.loads((Path(adapter) / "provenance.json").read_text())
            if provenance["base_model"] != MODEL_ID or provenance["revision"] != MODEL_REVISION:
                raise ValueError("Adapter base model/revision mismatch")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, idea, duration=30, attempts=3):
        import torch

        if not 1 <= attempts <= 3:
            raise ValueError("attempts must be 1–3")
        messages = messages_for(idea, duration)
        last_error = ""
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
            try:
                return parse_script(raw, duration), attempt + 1
            except ValueError as exc:
                last_error = str(exc)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": "صحح JSON والتزم بالمخطط والمدة. الخطأ: " + last_error[:700],
                        },
                    ]
                )
        raise ValueError(f"No valid script after {attempts} attempts: {last_error}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--adapter")
    p.add_argument("--output", default="outputs/script.json")
    args = p.parse_args()
    script, attempts = ScriptGenerator(args.adapter).generate(args.idea, args.duration)
    write_json(args.output, script.model_dump())
    Path(args.output).with_suffix(".txt").write_text(script.narration(), encoding="utf-8")
    print(f"Validated script written after {attempts} attempt(s): {args.output}")


if __name__ == "__main__":
    main()
