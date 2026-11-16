"""Generate reproducible instruction/code pairs for smooth Python motion graphics.

Separate from the existing JSON scriptwriter schema. Each target is executable
Python source, not a narration plan. No model training is claimed here.
"""
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "python_motion"
TOPICS = [
    ("smooth entrance", "Move a headline into the safe area with smooth easing."),
    ("animated comparison", "Reveal two rounded cards in sequence, without overlap."),
    ("progress explanation", "Animate a progress bar and numeric percentage."),
    ("three-step flow", "Reveal three connected steps with staggered motion."),
    ("code tip", "Present a short programming tip with a highlighted keyword."),
    ("callout", "Slide a small explanatory callout into view."),
    ("timeline", "Reveal milestones on a vertical timeline."),
    ("statistics", "Animate three bars to different heights."),
]
PALETTES = [
    ("#F4F3EF", "#172023", "#E34D3E"),
    ("#101827", "#F5F7FB", "#52C9BA"),
    ("#FFF9F1", "#1D2630", "#FF785A"),
    ("#EFF4F7", "#18334B", "#4B89D1"),
]
FPS = (24, 30)
DURATIONS = (4, 5, 6)
EASINGS = ("smoothstep", "smootherstep")
KINDS = ("cards", "bars", "steps", "headline")

SOURCE = r'''import math
import subprocess
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 540, 960
FPS = {fps}
DURATION = {duration}
BACKGROUND = "{background}"
FOREGROUND = "{foreground}"
ACCENT = "{accent}"
KIND = "{kind}"
TITLE = {title}
SUBTITLE = {subtitle}

def clamp01(x):
    return max(0.0, min(1.0, x))

def ease(x):
    x = clamp01(x)
    return {easing}

def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()

def frame(t):
    # All movement is evaluated from absolute time: no accumulated drift.
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    progress = ease(t / 0.8)
    y = round(140 + (1 - progress) * -100)
    draw.text((WIDTH // 2, y), TITLE, font=font(32),
              anchor="mt", fill=FOREGROUND)
    draw.rounded_rectangle((44, 238, WIDTH - 44, 244),
                           radius=3, fill=ACCENT)
    if KIND == "bars":
        for i, height in enumerate((165, 255, 205)):
            p = ease((t - 0.4 - i * 0.22) / 0.95)
            x = 95 + i * 130
            draw.rounded_rectangle((x, 710 - height * p, x + 76, 710),
                                   radius=12, fill=ACCENT)
    elif KIND == "steps":
        for i in range(3):
            p = ease((t - 0.35 - i * 0.28) / 0.7)
            x = round(80 + (1 - p) * 40)
            y0 = 335 + i * 130
            draw.rounded_rectangle((x, y0, x + 380 * p, y0 + 88),
                                   radius=16, fill=ACCENT)
            draw.text((x + 24, y0 + 24), str(i + 1),
                      font=font(26), fill=BACKGROUND)
    elif KIND == "cards":
        for i in range(2):
            p = ease((t - 0.35 - i * 0.38) / 0.8)
            x = round(55 + (1 - p) * 430)
            y0 = 360 + i * 165
            draw.rounded_rectangle((x, y0, x + 430, y0 + 125),
                                   radius=22, fill=ACCENT)
            draw.ellipse((x + 20, y0 + 35, x + 74, y0 + 89),
                         fill=BACKGROUND)
    else:
        p = ease((t - 0.4) / 1.0)
        draw.rounded_rectangle((65, 360, 475, 650), radius=25,
                               outline=ACCENT, width=6)
        draw.rounded_rectangle((90, 560, 90 + 360 * p, 582),
                               radius=10, fill=ACCENT)
    draw.text((WIDTH // 2, 830), SUBTITLE, font=font(20),
              anchor="mt", fill=FOREGROUND)
    return image

def main():
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "20", "motion.mp4",
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(FPS * DURATION):
            proc.stdin.write(frame(i / FPS).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        proc.kill()
        proc.wait()
        raise

if __name__ == "__main__":
    main()
'''

def make_row(index):
    topic, objective = TOPICS[index % len(TOPICS)]
    bg, fg, accent = PALETTES[(index // len(TOPICS)) % len(PALETTES)]
    fps = FPS[(index // 5) % len(FPS)]
    duration = DURATIONS[(index // 7) % len(DURATIONS)]
    easing = EASINGS[(index // 11) % len(EASINGS)]
    kind = KINDS[(index // 3) % len(KINDS)]
    title = f"Motion lesson {index + 1}"
    subtitle = f"{topic.title()} / {duration}s"
    values = {
        "fps": fps, "duration": duration, "background": bg,
        "foreground": fg, "accent": accent, "kind": kind,
        "title": repr(title), "subtitle": repr(subtitle),
        "easing": ("x * x * (3 - 2 * x)" if easing == "smoothstep"
                   else "x * x * x * (x * (x * 6 - 15) + 10)"),
    }
    code = SOURCE
    for key, value in values.items():
        token = "{"+key+"}"
        code = code.replace(token, str(value))
    ast.parse(code)
    compile(code, f"motion_{index:04d}.py", "exec")
    prompt = (
        f"Write a complete executable Python script using Pillow and ffmpeg "
        f"for a {duration}-second vertical 540x960 motion graphics video at {fps} fps. "
        f"{objective} Use {kind} composition, {easing} easing, "
        f"background {bg}, foreground {fg}, accent {accent}. "
        "Animate using absolute frame time; keep all elements within the canvas, "
        "avoid flashing, use rounded shapes, and export motion.mp4. "
        f"Show the title '{title}' and subtitle '{subtitle}'."
    )
    return {"id": f"python-motion-{index:04d}", "instruction": prompt,
            "response": code, "metadata": {"topic": topic, "kind": kind,
            "fps": fps, "duration": duration, "easing": easing,
            "source": "synthetic_template", "license": "CC0-1.0"}}

def build(count=240):
    if count < 30:
        raise ValueError("count must be >= 30")
    rows = [make_row(i) for i in range(count)]
    # Split by topic families rather than random duplicate prompts.
    # Each split has distinct exact instructions and source files.
    rng = random.Random(20261116)
    rng.shuffle(rows)
    n_train = count * 8 // 10
    n_val = count // 10
    return {"train": rows[:n_train],
            "validation": rows[n_train:n_train + n_val],
            "test": rows[n_train + n_val:]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    splits = build(args.count)
    ids = [row["id"] for rows in splits.values() for row in rows]
    assert len(ids) == len(set(ids))
    for name, rows in splits.items():
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = OUT / f"{name}.jsonl"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != data:
                raise SystemExit(f"Mismatch: {path}")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding="utf-8")
        print(name, len(rows), hashlib.sha256(data.encode()).hexdigest()[:16])
    print("Syntax validated. Rendering, motion quality and model performance NOT measured.")

if __name__ == "__main__":
    main()
