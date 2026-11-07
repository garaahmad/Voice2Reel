"""Deterministic vertical stickman renderer for a script and synthesized segment timings."""

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

from voice2reel.scriptwriter.schema import Script


def validate_timeline(script, metadata):
    segments = metadata.get("segments", [])
    if len(segments) != len(script.beats):
        raise ValueError("Narration segments must correspond one-to-one to script beats")
    end = 0.0
    for beat, segment in zip(script.beats, segments):
        start, stop = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start + stop) or abs(start - end) > 1e-4 or stop <= start:
            raise ValueError("Narration timing must be finite, contiguous, and positive")
        if beat.narration != segment["text"]:
            raise ValueError("Narration differs from script; regenerate audio after editing the script")
        end = stop
    if not math.isfinite(float(metadata["duration"])) or abs(end - metadata["duration"]) > 1e-4:
        raise ValueError("Metadata duration differs from segment timings")
    if metadata.get("language") != script.language:
        raise ValueError("Audio language differs from script")
    return segments


def find_font(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f"Font not found: {path}")
        return path
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(name).is_file():
            return Path(name)
    raise RuntimeError("Install fonts-noto-core or supply --font with an Arabic-capable font")


def wrap_text(text, font, width, direction):
    lines, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if current and font.getlength(candidate, direction=direction) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


class StickmanRenderer:
    """Fixed visual vocabulary; all displayed story content comes from the script."""

    def __init__(self, script, segments, font_path, width=1080, height=1920, symbol=";"):
        if not features.check_feature("raqm"):
            raise RuntimeError("Pillow needs RAQM support to render Arabic correctly")
        self.script, self.segments = script, segments
        self.width, self.height = width, height
        self.scale = width / 1080
        self.symbol = symbol
        self.direction = "rtl" if script.language == "ar" else "ltr"
        self.font_path = str(font_path)
        self.body = self.font(43)
        self.heading = self.font(64)
        self.small = self.font(28)
        self.symbol_font = ImageFont.truetype("DejaVuSansMono.ttf", round(130 * self.scale))
        self.mono = ImageFont.truetype("DejaVuSansMono.ttf", round(30 * self.scale))
        self.layers = [self.static_layer(i) for i in range(len(script.beats))]

    def font(self, size):
        return ImageFont.truetype(self.font_path, round(size * self.scale))

    def pt(self, x, y):
        return (round(x * self.scale), round(y * self.scale))

    def rect(self, draw, xy, fill, outline=None, radius=24, width=2):
        draw.rounded_rectangle(
            tuple(round(v * self.scale) for v in xy),
            radius=round(radius * self.scale),
            fill=fill,
            outline=outline,
            width=max(1, round(width * self.scale)),
        )

    def centered(self, draw, text, y, font, fill, max_width=900):
        for line in wrap_text(text, font, max_width * self.scale, self.direction):
            draw.text(self.pt(540, y), line, font=font, fill=fill, anchor="mt", direction=self.direction)
            y += font.size / self.scale * 1.65
        return y

    def static_layer(self, index):
        image = Image.new("RGB", (self.width, self.height), "#101725")
        draw = ImageDraw.Draw(image)
        for y in range(0, 1920, 80):
            draw.line([self.pt(0, y), self.pt(1080, y)], fill="#172131", width=1)
        for x in range(0, 1080, 80):
            draw.line([self.pt(x, 0), self.pt(x, 1920)], fill="#172131", width=1)
        draw.text(self.pt(70, 60), "VOICE2REEL / MODEL DEMO", font=self.mono, fill="#84a5c3")
        self.rect(draw, (70, 137, 1010, 390), "#182438", "#2b425a")
        title_font = self.heading
        while len(wrap_text(self.script.title, title_font, 880 * self.scale, self.direction)) > 2:
            title_font = self.font(title_font.size / self.scale - 3)
        self.centered(draw, self.script.title, 173, title_font, "#f0f6ff", 880)
        self.rect(draw, (400, 425, 680, 495), "#ffce5c", radius=35)
        draw.text(
            self.pt(540, 438),
            f"{index + 1:02d} / {len(self.script.beats):02d}",
            anchor="mt",
            font=self.mono,
            fill="#142033",
        )
        # Floating caption card: the text is model-generated, not handwritten by the renderer.
        self.rect(draw, (110, 1150, 970, 1390), "#1c2e45", "#3b6389")
        caption_font = self.font(49)
        while (
            len(wrap_text(self.script.beats[index].caption, caption_font, 780 * self.scale, self.direction))
            * caption_font.size
            / self.scale
            * 1.65
            > 170
        ):
            caption_font = self.font(caption_font.size / self.scale - 2)
        self.centered(draw, self.script.beats[index].caption, 1190, caption_font, "#ffce5c", 780)
        self.rect(draw, (60, 1460, 1020, 1800), "#080e19", "#2b425a")
        # Keep sentence-level subtitles readable; actual audio timing determines the active beat.
        body = self.body
        while (
            len(wrap_text(self.script.beats[index].narration, body, 880 * self.scale, self.direction))
            * body.size
            / self.scale
            * 1.65
            > 285
        ):
            body = self.font(body.size / self.scale - 2)
        self.centered(draw, self.script.beats[index].narration, 1490, body, "#f1f5fc", 880)
        draw.text(self.pt(540, 1835), "QWEN 2.5  +  XTTS V2", font=self.mono, fill="#728ba5", anchor="mt")
        return image

    def frame(self, t):
        index = next((i for i, s in enumerate(self.segments) if t < s["end"]), len(self.segments) - 1)
        image = self.layers[index].copy()
        draw = ImageDraw.Draw(image)
        beat = self.script.beats[index]
        bob = math.sin(t * 3) * 8
        cx = 360 + (30 if beat.camera == "right" else -30 if beat.camera == "left" else 0)
        cy = 730 + bob
        if beat.pose == "LEAN":
            cx += 65
            cy += 30
        color = "#eff6ff"

        def line(points, width=15, fill=color):
            pts = [self.pt(x, y) for x, y in points]
            draw.line(pts, fill=fill, width=round(width * self.scale), joint="curve")
            r = width / 2
            for x, y in points:
                draw.ellipse((*self.pt(x - r, y - r), *self.pt(x + r, y + r)), fill=fill)

        # A reusable articulated stick figure, laptop and table.
        draw.ellipse(
            (*self.pt(cx - 72, cy - 72), *self.pt(cx + 72, cy + 72)),
            outline=color,
            width=round(13 * self.scale),
        )
        eye_y = cy - 8
        for dx in (-24, 24):
            draw.ellipse((*self.pt(cx + dx - 6, eye_y - 6), *self.pt(cx + dx + 6, eye_y + 6)), fill=color)
        if beat.pose == "SHOCK":
            draw.ellipse((*self.pt(cx - 12, cy + 25), *self.pt(cx + 12, cy + 52)), outline="#ffce5c", width=4)
        else:
            draw.arc((*self.pt(cx - 25, cy + 10), *self.pt(cx + 25, cy + 48)), 0, 180, fill=color, width=4)
        line([(cx, cy + 78), (cx, 970)])
        line([(cx, 970), (cx - 65, 1085)])
        line([(cx, 970), (cx + 80, 1085)])
        if beat.pose == "CELEB":
            line([(cx, 830), (cx - 105, 760), (cx - 135, 650 + bob)])
            line([(cx, 830), (cx + 115, 750), (cx + 150, 650 - bob)])
        elif beat.pose == "SHOCK":
            line([(cx, 830), (cx - 105, 790), (cx - 90, 700)])
            line([(cx, 830), (cx + 100, 790), (cx + 90, 700)])
        elif beat.pose == "SUSPECT":
            line([(cx, 830), (cx - 85, 925), (cx + 60, 925)])
            line([(cx, 830), (cx + 75, 865), (cx + 20, cy + 50)])
        else:
            line([(cx, 830), (cx - 85, 925), (cx + 100, 960)])
            line([(cx, 830), (cx + 85, 910), (cx + 145, 940 + 8 * math.sin(t * 12))])
        self.rect(draw, (520, 775, 940, 1005), "#23364d", "#72c6f3", radius=18, width=5)
        self.rect(draw, (545, 800, 915, 975), "#090e19", radius=8)
        draw.text(self.pt(730, 805), self.symbol, font=self.symbol_font, fill="#ffce5c", anchor="mt")
        line([(495, 1010), (960, 1010)], width=16, fill="#82a9ca")
        line([(145, 1110), (980, 1110)], width=11, fill="#41617c")
        for x in (215, 905):
            line([(x, 1110), (x, 1140)], width=9, fill="#41617c")
        # Moving cue makes timing visible without altering any spoken text.
        for k in range(7):
            level = 15 + 55 * abs(math.sin(t * 5 + k * 0.7))
            x = 660 + 28 * k
            line([(x, 650 - level / 2), (x, 650 + level / 2)], width=8, fill="#54d4c7")
        self.rect(draw, (75, 1890, 1005, 1904), "#28384c", radius=7)
        progress = t / self.segments[-1]["end"]
        if progress > 0:
            self.rect(draw, (75, 1890, 75 + 930 * min(progress, 1), 1904), "#ffce5c", radius=7)
        return image


def render(script_path, narration_path, metadata_path, output, font=None, fps=30, width=1080):
    script = Script.model_validate_json(Path(script_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(metadata_path).read_text())
    segments = validate_timeline(script, metadata)
    if width < 360 or width % 2 or not 1 <= fps <= 60:
        raise ValueError("Use an even width >= 360 and FPS between 1 and 60")
    height = round(width * 16 / 9 / 2) * 2
    import soundfile as sf

    if abs(sf.info(narration_path).duration - metadata["duration"]) > 0.02:
        raise ValueError("Narration file duration differs from metadata")
    rig = StickmanRenderer(script, segments, find_font(font), width, height)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = math.ceil(metadata["duration"] * fps)
    with tempfile.TemporaryDirectory(prefix="v2r-render-", dir=output.parent) as work:
        silent = Path(work) / "visuals.mp4"
        log = Path(work) / "ffmpeg.log"
        with log.open("wb") as errors:
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "19",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    str(silent),
                ],
                stdin=subprocess.PIPE,
                stderr=errors,
            )
            try:
                for frame in range(count):
                    process.stdin.write(rig.frame(frame / fps).tobytes())
                    if frame % fps == 0:
                        print(json.dumps({"frame": frame, "total": count}), flush=True)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(log.read_text())
            except BaseException:
                process.kill()
                process.wait()
                raise
        completed = Path(work) / "final.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent),
                "-i",
                str(narration_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-t",
                str(metadata["duration"]),
                str(completed),
            ],
            check=True,
        )
        completed.replace(output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--timing", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--font")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1080)
    a = p.parse_args()
    print(render(a.script, a.audio, a.timing, a.output, a.font, a.fps, a.width))
