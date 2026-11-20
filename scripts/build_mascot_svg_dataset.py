"""Synthetic supervised text-to-SVG mascot dataset.

The target is an editable vector drawing, not a diffusion image model.
Use independent seeds and palettes for train/validation; generated samples
are procedural and cannot replace curated human-made art.
"""
import argparse
import json
import math
import random
from pathlib import Path

SYSTEM = ("You are a professional 2D mascot illustrator. Produce ONLY one complete "
          "SVG image, 512x768 viewBox, using paths, circles, ellipses and solid colors. "
          "Draw a friendly expressive stickman with smooth rounded limbs, large "
          "expressive eyes, eyebrows, rounded hands and shoes. No hat, no text, "
          "no external images, no script, no animation. SVG only.")
PALETTES = [
    ("charcoal", "#161c28", "#fefbf5", "#ff6955"),
    ("navy", "#192d45", "#f7fbff", "#52b9e9"),
    ("plum", "#29213d", "#fff9fc", "#f58ac1"),
    ("forest", "#193b34", "#f9fdf8", "#65cba7"),
    ("aubergine", "#332338", "#fffbf6", "#f7ad59"),
]
EMOTIONS = ["happy", "surprised", "curious", "confident", "thoughtful"]
POSES = ["greeting", "explaining", "pointing", "standing", "celebrating"]

def build(seed):
    rng = random.Random(seed)
    label, ink, paper, accent = rng.choice(PALETTES)
    emotion, pose = rng.choice(EMOTIONS), rng.choice(POSES)
    lean = rng.randint(-12, 12)
    head = rng.randint(74, 86)
    eye = rng.randint(12, 16)
    hand_y = rng.randint(390, 445)
    arm_left = 145 if pose in ("greeting", "celebrating") else 110
    arm_right = 145 if pose in ("pointing", "celebrating") else 400
    left_y = 245 if pose in ("greeting", "celebrating") else hand_y
    right_y = 235 if pose in ("pointing", "celebrating") else hand_y
    mouth = ('<ellipse cx="256" cy="265" rx="12" ry="17" fill="'+ink+'"/>'
             if emotion == "surprised" else
             '<path d="M233 267 Q256 289 279 267" fill="none" stroke="'+ink+'" stroke-width="5" stroke-linecap="round"/>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="512" height="768" viewBox="0 0 512 768">
<ellipse cx="256" cy="695" rx="116" ry="14" fill="#dbe0e4"/>
<g fill="none" stroke="{ink}" stroke-linecap="round" stroke-linejoin="round">
<path d="M256 349 Q{250+lean} 440 256 525" stroke-width="25"/>
<path d="M245 378 Q193 398 {arm_left} {left_y}" stroke-width="18"/>
<path d="M267 378 Q325 395 {arm_right} {right_y}" stroke-width="18"/>
<path d="M256 522 Q224 575 212 666" stroke-width="21"/>
<path d="M256 522 Q288 575 300 666" stroke-width="21"/>
<path d="M212 665 Q198 681 173 683" stroke-width="21"/>
<path d="M300 665 Q314 681 339 683" stroke-width="21"/>
</g>
<ellipse cx="{arm_left}" cy="{left_y}" rx="16" ry="19" fill="{ink}"/>
<ellipse cx="{arm_right}" cy="{right_y}" rx="16" ry="19" fill="{ink}"/>
<circle cx="256" cy="257" r="{head}" fill="{paper}" stroke="{ink}" stroke-width="8"/>
<path d="M203 218 Q220 205 236 219 M276 219 Q293 205 309 218" fill="none" stroke="{ink}" stroke-width="6" stroke-linecap="round"/>
<ellipse cx="224" cy="244" rx="{eye}" ry="{eye+5}" fill="white" stroke="{ink}" stroke-width="4"/>
<ellipse cx="288" cy="244" rx="{eye}" ry="{eye+5}" fill="white" stroke="{ink}" stroke-width="4"/>
<circle cx="227" cy="247" r="7" fill="{ink}"/><circle cx="291" cy="247" r="7" fill="{ink}"/>
<circle cx="224" cy="243" r="2.5" fill="white"/><circle cx="288" cy="243" r="2.5" fill="white"/>
{mouth}
<path d="M242 375 Q256 367 270 375" fill="none" stroke="{accent}" stroke-width="7" stroke-linecap="round"/>
</svg>"""
    request = (f"Draw an original {emotion} {label} stickman mascot, {pose} pose, "
               f"ink {ink}, background {paper}, accent {accent}. "
               "Rounded hands and feet, smooth silhouette, expressive face, no hat.")
    return {"messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":request},
                        {"role":"assistant","content":svg}]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--count",type=int,default=1200)
    args=p.parse_args()
    if not 100 <= args.count <= 10000:
        raise ValueError("count must be 100..10000")
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for split, start, count in (("train",0,int(args.count*.9)),
                                ("validation",100000,int(args.count*.1))):
        with (root/f"{split}.jsonl").open("w",encoding="utf-8") as f:
            for i in range(start,start+count):
                row=build(i+83)
                row["id"]=f"mascot-{split}-{i}"
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("SVG dataset:",root,"train",int(args.count*.9),"validation",int(args.count*.1))

if __name__=="__main__":
    main()
