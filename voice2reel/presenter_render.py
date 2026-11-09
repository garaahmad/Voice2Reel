"""Fast raster 2D cartoon presenter, driven by typed scenes and actual speech timing."""

import math
from bisect import bisect_right
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from voice2reel.arabic_text import speech_text
from voice2reel.motion_plan import fit_actor, scene_plan
from voice2reel.motion_preview import ease, joint, mix
from voice2reel.render import find_font, validate_timeline, wrap_text
from voice2reel.text_layout import draw_text, text_length

VERSION = "presenter-2"
PROFILES = {"fast": (540, 24, "ultrafast"), "quality": (720, 30, "veryfast")}
INK, RED = "#172023", "#e34d3e"


def curve(points, steps=20):
    a, b, c, d = points
    return [
        (
            (1 - u) ** 3 * a[0] + 3 * (1 - u) ** 2 * u * b[0] + 3 * (1 - u) * u * u * c[0] + u**3 * d[0],
            (1 - u) ** 3 * a[1] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u * u * c[1] + u**3 * d[1],
        )
        for u in np.linspace(0, 1, steps)
    ]


def rounded_line(draw, points, width, color=INK):
    draw.line(points, fill=color, width=width, joint="curve")
    r = width / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


@lru_cache(maxsize=192)
def face(emotion, mouth, blink, look=0, tilt=0):
    """Original expressive vector face, rendered at 2x and shared between frames."""
    im = Image.new("RGBA", (440, 500))
    d = ImageDraw.Draw(im)

    def ellipse(box, fill, outline=None, width=1):
        d.ellipse(tuple(round(v * 2) for v in box), fill=fill, outline=outline, width=round(width * 2))

    def path(points, fill, outline=None, width=1):
        pts = [(round(x * 2), round(y * 2)) for x, y in points]
        d.polygon(pts, fill=fill)
        if outline:
            d.line(pts + [pts[0]], fill=outline, width=round(width * 2), joint="curve")

    def line(points, width=3, color=INK):
        d.line(
            [(round(x * 2), round(y * 2)) for x, y in points],
            fill=color,
            width=round(width * 2),
            joint="curve",
        )

    # A clean circular face, with just enough edge shading to separate it from paper.
    ellipse((35, 58, 193, 219), "#cdd6d6", INK, 4.5)
    ellipse((40, 62, 190, 214), "#fffef9")
    ellipse((43, 74, 57, 174), "#eef1eb")
    for ex in (89, 143):
        if blink:
            line(curve([(ex - 14, 141), (ex - 6, 148), (ex + 8, 148), (ex + 15, 139)]), 3.3)
        else:
            ellipse((ex - 16, 119, ex + 16, 157), "#fff", INK, 2.2)
            shift = 3 if emotion == "thinking" else 0
            ellipse((ex - 6 + look, 127 - shift, ex + 8 + look, 149 - shift), INK)
            ellipse((ex - 3 + look, 129 - shift, ex + 2 + look, 135 - shift), "#fff")
        y = 108 if emotion in ("surprised", "thinking") else 113
        if emotion == "concerned":
            pts = [
                (ex - 15, y + 2 if ex < 110 else y - 6),
                (ex, y - 7),
                (ex + 15, y - 6 if ex < 110 else y + 2),
            ]
        elif emotion == "thinking" and ex > 110:
            pts = curve([(ex - 15, y + 3), (ex - 2, y - 8), (ex + 8, y - 6), (ex + 15, y - 2)])
        else:
            pts = curve([(ex - 15, y + 2), (ex - 8, y - 6), (ex + 7, y - 6), (ex + 15, y + 2)])
        line(pts, 4)
    # Closed mouth during pauses, even when the expression is surprised.
    if mouth == 3:
        ellipse((105, 175, 128, 198), INK)
        ellipse((110, 191, 125, 197), "#df7b72")
    elif mouth == 2:
        path(
            curve([(91, 175), (106, 182), (126, 182), (140, 173)])
            + curve([(140, 173), (134, 205), (102, 207), (91, 175)])[1:],
            INK,
        )
        path([(97, 179), (134, 179), (130, 186), (102, 186)], "#fffef9")
        ellipse((110, 195, 129, 202), "#df7b72")
    elif mouth == 1:
        ellipse((101, 179, 132, 193), INK)
    elif emotion == "concerned":
        line(curve([(100, 188), (109, 180), (125, 180), (135, 188)]), 2.7)
    elif emotion == "thinking":
        line(curve([(105, 186), (113, 189), (123, 184), (132, 185)]), 2.7)
    elif emotion == "surprised":
        line(curve([(106, 187), (114, 184), (122, 184), (130, 187)]), 2.7)
    else:
        line(curve([(96, 181), (108, 197), (129, 193), (139, 178)]), 2.7)
    # Compact illustrated beret: a simple readable silhouette rather than a long pointed cap.
    hat = curve([(22, 98), (6, 65), (58, 39), (107, 33)])
    hat += curve([(107, 33), (153, 14), (206, 34), (207, 65)])[1:]
    hat += curve([(207, 65), (207, 88), (169, 96), (133, 93)])[1:]
    hat += curve([(133, 93), (92, 92), (56, 117), (22, 98)])[1:]
    path(hat, "#e84b3c", INK, 4)
    highlight = curve([(35, 77), (59, 50), (124, 31), (163, 38)])
    highlight += curve([(163, 38), (187, 43), (187, 57), (164, 56)])[1:]
    highlight += curve([(164, 56), (113, 45), (69, 66), (35, 77)])[1:]
    path(highlight, "#fa7158")
    brim = curve([(28, 95), (69, 94), (115, 75), (169, 83)])
    brim += curve([(169, 83), (140, 99), (93, 99), (47, 110)])[1:]
    path(brim, "#bd332c", INK, 2)
    path([(112, 34), (105, 19), (109, 15), (116, 18), (122, 32)], "#d74235", INK, 2)
    if tilt:
        im = im.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(220, 350))
    return im


@lru_cache(maxsize=128)
def glove(kind, angle):
    im = Image.new("RGBA", (112, 132))
    d = ImageDraw.Draw(im)
    d.ellipse((33, 43, 81, 98), fill=INK)
    if kind == "point":
        rounded_line(d, [(59, 61), (59, 17)], 15)
        rounded_line(d, [(36, 68), (23, 84)], 14)
        for x in (49, 64, 77):
            rounded_line(d, [(x, 75), (x, 94)], 13)
    elif kind == "thumb":
        rounded_line(d, [(38, 61), (28, 27)], 16)
        for y in (60, 75, 88):
            rounded_line(d, [(53, y), (81, y)], 14)
    elif kind == "open":
        for x, y in [(36, 34), (50, 24), (65, 26), (79, 39)]:
            rounded_line(d, [(x, 65), (x - 2, y)], 13)
        rounded_line(d, [(37, 77), (20, 60)], 14)
    else:
        for x, y in [(38, 103), (52, 111), (66, 108), (78, 99)]:
            rounded_line(d, [(x, 77), (x - 3, y)], 13)
        rounded_line(d, [(37, 65), (21, 83)], 14)
    return im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)


GESTURES = {
    "present": ((109, 653), (284, 570), "rest", "open", 0, -75),
    "point": ((109, 653), (295, 518), "rest", "point", 0, -90),
    "think": ((109, 653), (228, 456), "rest", "rest", -25, 145),
    "shrug": ((72, 534), (280, 526), "open", "open", 65, -65),
    "thumbs_up": ((110, 650), (278, 519), "rest", "thumb", 0, -5),
    "wave": ((108, 650), (270, 395), "rest", "open", 0, 5),
}
CAMERAS = {"wide": (1.0, 0.0, 0.0), "medium": (1.13, -7.0, -4.0), "close": (1.5, 105.0, 35.0)}


def word_boundaries(segment):
    words = segment.get("words", [])
    try:
        valid = (
            isinstance(words, list)
            and words
            and all(
                isinstance(w.get("text"), str)
                and math.isfinite(w["start"] + w["end"])
                and segment["start"] <= w["start"] < w["end"] <= segment["end"]
                for w in words
            )
        )
        ordered = valid and all(a["end"] <= b["start"] + 0.001 for a, b in zip(words, words[1:]))
        return words if ordered else []
    except (AttributeError, KeyError, TypeError, ValueError):
        return []


class PresenterRenderer:
    def __init__(self, script, segments, audio, rate, width=540):
        self.script, self.segments = script, segments
        self.audio, self.rate = audio, rate
        self.width = width
        bundled_font = Path(__file__).parent / "assets" / "Cairo.ttf"
        self.font_path = str(bundled_font if bundled_font.is_file() else find_font())
        self.fonts = {}
        self.direction = "rtl" if script.language == "ar" else "ltr"
        # Compile comfort-bounded camera/layout plans from the authored vocabulary.
        # A wide -> close request becomes two adjacent steps instead of one jump.
        self.plans = []
        previous_camera = "wide"
        for scene, segment in zip(script.scenes, segments):
            plan = scene_plan(
                {
                    "camera": scene.camera,
                    "gesture": scene.gesture,
                    "emotion": scene.emotion,
                },
                previous_camera=previous_camera,
                duration=segment["end"] - segment["start"],
                panel_visible=scene.visual.kind != "none",
            )
            self.plans.append(plan)
            previous_camera = plan["camera"]
        # Paint the background and all text once, rather than shaping Arabic every frame.
        self.bg = Image.new("RGB", (540, 960), "#f3f2ec")
        d = ImageDraw.Draw(self.bg)
        for x in range(0, 540, 45):
            d.line((x, 0, x, 960), fill="#e8e9e3")
        for y in range(0, 960, 45):
            d.line((0, y, 540, y), fill="#e8e9e3")
        d.ellipse((-135, 290, 390, 823), fill="#dbe8e8")
        d.ellipse((-75, 355, 326, 756), outline="#ccdddd", width=1)
        d.line((36, 42, 74, 42), fill=RED, width=5)
        self.headlines = [self.headline(s.caption, i) for i, s in enumerate(script.scenes)]
        self.panels = [self.panel(s.visual) for s in script.scenes]
        self.subtitles = []
        self.subtitle_ends = []
        for scene, segment in zip(script.scenes, segments):
            boundaries = word_boundaries(segment)
            if boundaries:
                groups = [boundaries[i : i + 4] for i in range(0, len(boundaries), 4)]
                source_words = speech_text(scene).split()
                phrases = (
                    [" ".join(source_words[i : i + 4]) for i in range(0, len(source_words), 4)]
                    if len(source_words) == len(boundaries)
                    else [" ".join(w["text"] for w in g) for g in groups]
                )
                self.subtitles.append([self.subtitle(text) for text in phrases])
                duration = segment["end"] - segment["start"]
                self.subtitle_ends.append(
                    [(g[0]["start"] - segment["start"]) / duration for g in groups[1:]] + [1.0]
                )
            else:
                words = speech_text(scene).split()
                groups = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)]
                self.subtitles.append([self.subtitle(text) for text in groups])
                cumulative = np.cumsum([len(text.split()) for text in groups])
                self.subtitle_ends.append((cumulative / cumulative[-1]).tolist())

    def font(self, size, mono=False):
        key = (size, mono)
        if key not in self.fonts:
            font = ImageFont.truetype("DejaVuSansMono.ttf" if mono else self.font_path, size)
            if not mono and Path(self.font_path).name == "Cairo.ttf":
                font.set_variation_by_axes([700, 0])
            self.fonts[key] = font
        return self.fonts[key]

    def text_block(self, draw, text, xy, width, size=26, color=INK, max_lines=3, direction=None):
        direction = direction or self.direction
        font = self.font(size)
        while size > 12 and (
            len(wrap_text(text, font, width, direction)) > max_lines
            or any(
                text_length(line, font, direction) > width
                for line in wrap_text(text, font, width, direction)
            )
        ):
            size -= 1
            font = self.font(size)
        lines = wrap_text(text, font, width, direction)
        for i, line in enumerate(lines[:max_lines]):
            while text_length(line, font, direction) > width and len(line) > 1:
                line = line[:-2] + "…"
            draw_text(
                draw,
                (xy[0], xy[1] + i * size * 1.45),
                line,
                font,
                direction,
                fill=color,
                anchor="mt",
            )

    def headline(self, text, index):
        im = Image.new("RGBA", (540, 250))
        d = ImageDraw.Draw(im)
        d.text((500, 29), f"{index + 1:02d}", font=self.font(14), fill="#647e82", anchor="rt")
        self.text_block(d, text, (270, 87), 465, 46, max_lines=2)
        d.rounded_rectangle((238, 222, 302, 227), radius=2, fill=RED)
        return im

    def subtitle(self, text):
        im = Image.new("RGBA", (540, 100))
        d = ImageDraw.Draw(im)
        font = self.font(23)
        lines = wrap_text(text, font, 444, self.direction)[:2]
        height = 20 + len(lines) * 34
        d.rounded_rectangle((30, 0, 510, height), radius=13, fill="#172023f4")
        for i, line in enumerate(lines):
            while text_length(line, font, self.direction) > 444 and len(line) > 1:
                line = line[:-2] + "…"
            draw_text(d, (270, 10 + i * 34), line, font, self.direction, fill="#fffef8", anchor="mt")
        return im

    def panel(self, visual):
        im = Image.new("RGBA", (245, 355))
        d = ImageDraw.Draw(im)
        if visual.kind == "none":
            return im
        items = visual.items
        if visual.kind == "code":
            count = min(12, len(visual.code.splitlines()))
            panel_height = min(350, max(190, 122 + count * 26))
        elif visual.kind in ("list", "flow"):
            panel_height = min(350, max(130, 40 + len(items) * 78))
        else:
            panel_height = 350
        d.rounded_rectangle((3, 5, 241, panel_height), radius=17, fill="#fffef8", outline="#b9cccc", width=2)
        if visual.kind == "code":
            d.rounded_rectangle((13, 22, 231, panel_height - 40), radius=9, fill="#172023")
            lines = visual.code.splitlines()
            longest = max((len(line) for line in lines), default=1)
            size = min(24, max(12, int(200 / max(1, longest) / 0.61)))
            if len(lines) > 8:
                size = min(size, 15)
            font = self.font(size, True)
            limit = max(1, int(198 / font.getlength("M")))
            for i, line in enumerate(lines[:12]):
                focused = bool(
                    visual.focus
                    and visual.focus in line
                    and i == next((j for j, code_line in enumerate(lines) if visual.focus in code_line), -1)
                )
                if focused and visual.status != "neutral":
                    y = 47 + i * min(26, max(20, size + 3))
                    d.rounded_rectangle(
                        (18, y, 227, y + size + 7),
                        radius=4,
                        fill="#4b2a2b" if visual.status == "warning" else "#22483f",
                    )
                if len(line) > limit:
                    line = line[: max(0, limit - 1)] + "…"
                draw_text(
                    d,
                    (23, 49 + i * min(26, max(20, size + 3))),
                    line,
                    font,
                    "ltr",
                    fill=("#ffac99" if visual.status == "warning" else "#a4ebc8") if focused else "#e7f0e9",
                )
            if items:
                self.text_block(d, items[0], (123, panel_height - 32), 218, 16, max_lines=1)
        elif visual.kind == "compare":
            for i, label in enumerate(items[:2]):
                y = 50 + i * 145
                d.rounded_rectangle((17, y, 228, y + 110), radius=12, fill="#e0eee6" if i else "#f6e1dc")
                self.text_block(d, label, (123, y + 22), 190, 23, max_lines=2)
        else:
            for i, label in enumerate(items):
                y = 32 + i * 78
                d.rounded_rectangle((17, y, 228, y + 57), radius=10, fill="#dfeaf0")
                self.text_block(d, label, (123, y + 12), 190, 21, max_lines=2)
                if visual.kind == "flow" and i < len(items) - 1:
                    d.line((123, y + 59, 123, y + 76), fill=RED, width=3)
                    d.polygon([(117, y + 70), (123, y + 77), (129, y + 70)], fill=RED)
        return im.crop((0, 0, 245, panel_height + 5))

    def character(self, t, index, local):
        scene = self.script.scenes[index]
        previous = self.script.scenes[max(index - 1, 0)]
        u = ease(local / 0.48)
        old = GESTURES[previous.gesture] if index else GESTURES["present"]
        new = GESTURES[scene.gesture]
        im = Image.new("RGBA", (660, 1240))
        d = ImageDraw.Draw(im)
        breath = 1.4 * math.sin(t * 2.1)
        lean = mix(0 if index == 0 else 2, 4 if scene.gesture == "point" else -2, u)
        lean += 1.6 * math.sin(t * 0.8)
        hip = (164 + 2 * math.sin(t * 0.7), 650)

        def line(points, width):
            rounded_line(d, [(round(x * 2), round(y * 2) - 500) for x, y in points], round(width * 2))

        # Feet stay planted while hips and knees shift the weight. Separate shoulder anchors.
        d.ellipse((176, 1168, 506, 1212), fill="#223b3b20")
        for points in [[hip, (144, 746), (127, 834)], [hip, (198, 743), (214, 834)]]:
            line(points, 14)
        for x, mirror in [(127, -1), (214, 1)]:
            shoe = [
                (x - 10 * mirror, 825),
                (x + 9 * mirror, 825),
                (x + 12 * mirror, 833),
                (x + 31 * mirror, 835),
                (x + 39 * mirror, 842),
                (x + 40 * mirror, 848),
                (x + 35 * mirror, 852),
                (x - 11 * mirror, 852),
            ]
            d.polygon([(round(px * 2), round(py * 2) - 500) for px, py in shoe], fill=INK)
        torso = curve([(164 + lean, 468 + breath), (164 + lean, 530 + breath), (169, 601), hip])
        line(torso, 22)
        shoulders = [(148 + lean, 498 + breath), (181 + lean, 499 + breath)]
        line([shoulders[0], (164 + lean, 488 + breath), shoulders[1]], 17)
        for side in [0, 1]:
            a, b = old[side], new[side]
            wrist = (mix(a[0], b[0], u), mix(a[1], b[1], u) + breath)
            # A short anticipation, hold, and recovery follows each spoken phrase.
            phrase = local % 4.7
            recover = ease((phrase - 2.8) / 0.65) * (1 - ease((phrase - 3.8) / 0.75)) * 0.48
            rest = GESTURES["present"][side]
            wrist = (mix(wrist[0], rest[0], recover), mix(wrist[1], rest[1], recover))
            if side == 1:
                amp = 9 if scene.gesture == "wave" else 3
                wrist = (
                    wrist[0] + amp * math.sin(local * (6.8 if amp == 9 else 2.6)) * u,
                    wrist[1] + 2 * math.sin(local * 2.1) * u,
                )
            elbow, end = joint(shoulders[side], wrist, 91, 98, 1 if side else -1)
            # A curved elbow makes the silhouette feel drawn, without disconnected joints.
            points = curve(
                [
                    shoulders[side],
                    (mix(shoulders[side][0], elbow[0], 0.8), mix(shoulders[side][1], elbow[1], 0.8)),
                    elbow,
                    end,
                ]
            )
            line(points, 12)
            kind = new[2 + side] if u > 0.5 else old[2 + side]
            angle = round(mix(old[4 + side], new[4 + side], u) / 5) * 5
            hand = glove(kind, angle)
            # Palm center is the solved wrist; artwork is already 2x.
            im.alpha_composite(hand, (round(end[0] * 2 - 56), round(end[1] * 2 - 76) - 500))
        start = max(0, int((t - 0.025) * self.rate))
        end = min(len(self.audio), int((t + 0.025) * self.rate))
        energy = float(np.sqrt(np.mean(self.audio[start:end] ** 2))) if end > start else 0
        mouth = 0 if energy < 0.013 else (1 if energy < 0.045 else (2 if energy < 0.11 else 3))
        blink = abs((t + 1.2) % 4.4 - 2.2) < 0.065
        tilt = round(mix(0, -3 if scene.emotion in ("thinking", "concerned") else 2, u))
        head = face(scene.emotion, mouth, blink, 2 if scene.gesture == "point" else 0, tilt)
        im.alpha_composite(head, (round((50 + lean) * 2), round((250 + breath) * 2) - 500))
        plan = self.plans[index]
        prior_camera = self.plans[index - 1]["camera"] if index else plan["camera"]
        prior = CAMERAS[prior_camera]
        current = CAMERAS[plan["camera"]]
        zoom, panx, pany = [mix(a, b, u) for a, b in zip(prior, current)]
        zoom += 0.012 * ease(local / max(0.1, self.segments[index]["end"] - self.segments[index]["start"]))
        tx, ty = 164 * (1 - zoom) + panx, 450 * (1 - zoom) + pany
        crop = (25, 255, 330, 860)
        actor = im.crop((crop[0] * 2, crop[1] * 2 - 500, crop[2] * 2, crop[3] * 2 - 500))
        actor = actor.resize(
            (round((crop[2] - crop[0]) * zoom), round((crop[3] - crop[1]) * zoom)), Image.Resampling.BICUBIC
        )
        position = (round(crop[0] * zoom + tx), round(crop[1] * zoom + ty))
        scale, x, y = fit_actor(actor.size, position, tuple(plan["boxes"]["actor"]))
        if scale < 1.0:
            actor = actor.resize(
                (round(actor.width * scale), round(actor.height * scale)), Image.Resampling.LANCZOS
            )
        return actor, (x, y)

    def frame(self, t):
        index = next((i for i, s in enumerate(self.segments) if t < s["end"]), len(self.segments) - 1)
        segment = self.segments[index]
        local = t - segment["start"]
        image = self.bg.copy().convert("RGBA")
        actor, position = self.character(t, index, local)
        image.alpha_composite(actor, position)
        if self.plans[index]["camera"] != "close" and self.script.scenes[index].visual.kind != "none":
            enter = ease(local / 0.4)
            panel = self.panels[index]
            image.alpha_composite(panel, (round(540 - 245 * enter), 450))
            visual = self.script.scenes[index].visual
            if visual.kind == "code" and visual.focus and 0.6 < local < 3.4:
                lines = visual.code.splitlines()
                for row, line in enumerate(lines[:12]):
                    if visual.focus in line:
                        longest = max((len(code_line) for code_line in lines), default=1)
                        size = min(24, max(12, int(200 / max(1, longest) / 0.61)))
                        if len(lines) > 8:
                            size = min(size, 15)
                        font = self.font(size, True)
                        x = 295 + 23 + font.getlength(line[: line.index(visual.focus)])
                        x += font.getlength(visual.focus) / 2
                        y = 450 + 49 + row * min(26, max(20, size + 3)) + size * 0.55
                        radius = 13 + 2 * math.sin(local * 4)
                        color = RED if visual.status == "warning" else "#3c8871"
                        ImageDraw.Draw(image).ellipse(
                            (x - radius, y - radius, x + radius, y + radius), outline=color, width=2
                        )
                        break
        image.alpha_composite(self.headlines[index])
        subtitles = self.subtitles[index]
        # Provider word boundaries when supplied; otherwise estimated phrase pacing.
        fraction = max(0, min(0.999, local / (segment["end"] - segment["start"])))
        chunk = min(len(subtitles) - 1, bisect_right(self.subtitle_ends[index], fraction))
        image.alpha_composite(subtitles[chunk], (0, 865))
        if index and 0 <= local < 0.20:
            progress = local / 0.20
            x = round(-90 + 720 * progress)
            overlay = Image.new("RGBA", (540, 960))
            od = ImageDraw.Draw(overlay)
            od.polygon([(x - 55, 0), (x + 15, 0), (x - 55, 960), (x - 125, 960)], fill="#e34d3e60")
            image.alpha_composite(overlay)
        image = image.convert("RGB")
        if self.width != 540:
            image = image.resize((self.width, round(self.width * 16 / 9 / 2) * 2), Image.Resampling.LANCZOS)
        return image


def validate_audio(script, metadata, audio_path):
    # Reuse the strict contiguous timing check with the text actually spoken.
    legacy = SimpleNamespace(
        beats=[SimpleNamespace(narration=speech_text(s)) for s in script.scenes],
        language=script.language,
    )
    segments = validate_timeline(legacy, metadata)
    info = sf.info(audio_path)
    if not math.isfinite(info.duration) or abs(info.duration - metadata["duration"]) > 0.02:
        raise ValueError("WAV duration differs from narration metadata")
    return segments


def render(script, audio_path, metadata, output, profile="fast", *, soundtrack_path=None):
    if profile not in PROFILES:
        raise ValueError("Choose fast/quality profile")
    width, fps, preset = PROFILES[profile]
    segments = validate_audio(script, metadata, audio_path)
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("Non-finite audio")
    audio = audio.mean(axis=1)
    started = time.perf_counter()
    rig = PresenterRenderer(script, segments, audio, rate, width)
    count = math.ceil(metadata["duration"] * fps)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    height = round(width * 16 / 9 / 2) * 2
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        temp = Path(work) / "video.mp4"
        log = Path(work) / "ffmpeg.log"
        with log.open("wb") as errors:
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
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
                    "-i",
                    str(soundtrack_path or audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "2",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-t",
                    str(metadata["duration"]),
                    "-movflags",
                    "+faststart",
                    str(temp),
                ],
                stdin=subprocess.PIPE,
                stderr=errors,
            )
            try:
                for frame in range(count):
                    process.stdin.write(rig.frame(frame / fps).tobytes())
                process.stdin.close()
                if process.wait():
                    raise RuntimeError(log.read_text()[-1500:])
                temp.replace(output)
            except BaseException:
                process.kill()
                process.wait()
                raise
    return {
        "wall_seconds": time.perf_counter() - started,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": count,
        "comfort_scores": [p["comfort_score"] for p in rig.plans],
        "comfort_violations": [p["violations"] for p in rig.plans],
        "camera_plan": [p["camera"] for p in rig.plans],
    }
