"""Generate a new vector mascot with the independently trained Qwen adapter."""
import argparse
from pathlib import Path
from voice2reel.mascot_svg import to_png, validate_svg
from scripts.build_mascot_svg_dataset import SYSTEM

def generate(prompt, adapter, max_new_tokens=1600):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from voice2reel.train_mascot_svg import MODEL, REVISION
    tokenizer=AutoTokenizer.from_pretrained(adapter)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,
                                               dtype="auto",device_map="auto")
    model=PeftModel.from_pretrained(model,adapter).eval()
    messages=[{"role":"system","content":SYSTEM},
              {"role":"user","content":prompt}]
    inputs=tokenizer.apply_chat_template(messages,tokenize=True,
                                         add_generation_prompt=True,
                                         return_tensors="pt",return_dict=True)
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=max_new_tokens,
                              do_sample=False,pad_token_id=tokenizer.eos_token_id)
    raw=tokenizer.decode(output[0,inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
    start=raw.find("<svg")
    end=raw.rfind("</svg>")
    if start<0 or end<0:
        raise ValueError("Qwen did not output a complete SVG")
    return validate_svg(raw[start:end+6])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prompt",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--svg",required=True)
    p.add_argument("--png",required=True)
    args=p.parse_args()
    svg=generate(args.prompt,args.adapter)
    Path(args.svg).parent.mkdir(parents=True,exist_ok=True)
    Path(args.svg).write_text(svg,encoding="utf-8")
    to_png(svg,args.png)
    print("Generated vector mascot:",args.svg,args.png)

if __name__=="__main__":
    main()
