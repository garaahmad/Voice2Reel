"""Selectable Stickman character shapes for the interactive character lab.

This module is deliberately separate from ``stickman_artist.draw_frame``: that
function renders ONE authored house style. Here a catalog of visually distinct
character templates is drawn from the same 18-joint skeleton, so a user can pick
a shape from a list, preview it as a PNG, lock the choice, and reuse it for every
later frame, clip and report.

All artwork is deterministic vector code (Pillow). No diffusion model, no
downloaded artwork, no randomness in the default previews.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from voice2reel.common import write_json
from voice2reel.text_layout import draw_text

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "voice2reel" / "assets" / "Cairo.ttf"

# The 18 named joints used by datasets/stickman/vocabulary.json.
JOINT_ORDER = (
    "hip", "shoulder_left", "shoulder_right", "head_center", "head_top",
    "head_bottom", "neck", "torso_mid", "elbow_left", "elbow_right",
    "wrist_left", "wrist_right", "knee_left", "knee_right", "ankle_left",
    "ankle_right", "foot_left", "foot_right",
)

BASE_POSE = {
    "hip": (0.50, 0.62),
    "shoulder_left": (0.38, 0.33),
    "shoulder_right": (0.62, 0.33),
    "head_center": (0.50, 0.18),
    "head_top": (0.50, 0.075),
    "head_bottom": (0.50, 0.285),
    "neck": (0.50, 0.305),
    "torso_mid": (0.50, 0.47),
    "elbow_left": (0.30, 0.46),
    "elbow_right": (0.70, 0.46),
    "wrist_left": (0.26, 0.58),
    "wrist_right": (0.74, 0.58),
    "knee_left": (0.43, 0.76),
    "knee_right": (0.57, 0.76),
    "ankle_left": (0.43, 0.885),
    "ankle_right": (0.57, 0.885),
    "foot_left": (0.375, 0.915),
    "foot_right": (0.625, 0.915),
}

# Per-gesture normalized offsets. Kept small so every chain stays connected.
POSE_OFFSETS = {
    "idle": {},
    "present": {
        "wrist_left": (-0.10, -0.13), "elbow_left": (-0.05, -0.07),
        "wrist_right": (0.03, -0.02),
    },
    "point": {
        "wrist_right": (0.12, -0.12), "elbow_right": (0.07, -0.05),
        "head_center": (0.015, 0.0), "head_top": (0.015, 0.0),
        "head_bottom": (0.015, 0.0), "neck": (0.015, 0.0),
    },
    "wave": {
        "wrist_right": (0.02, -0.30), "elbow_right": (0.07, -0.15),
    },
    "think": {
        "wrist_right": (-0.16, -0.27), "elbow_right": (-0.04, -0.08),
        "head_center": (-0.025, 0.01), "head_top": (-0.025, 0.01),
        "head_bottom": (-0.025, 0.01), "neck": (-0.025, 0.01),
    },
    "explain": {
        "wrist_left": (-0.06, -0.08), "wrist_right": (0.06, -0.08),
        "elbow_left": (-0.02, -0.03), "elbow_right": (0.02, -0.03),
    },
    "celebrate": {
        "wrist_left": (-0.08, -0.27), "elbow_left": (-0.05, -0.13),
        "wrist_right": (0.08, -0.27), "elbow_right": (0.05, -0.13),
    },
    "step": {"knee_left": (-0.04, -0.05), "ankle_left": (-0.07, -0.04),
             "foot_left": (-0.07, -0.04), "knee_right": (0.03, 0.01)},
}
POSES = tuple(POSE_OFFSETS)

TEMPLATES = (
    {
        "id": "classic", "name_ar": "الكلاسيكي", "name_en": "Classic",
        "description": "ستيكمان نظيف بدون قبعة أو قفازات، بخطوط ناعمة",
        "paper": "#f3f2ec", "ink": "#172023", "accent": "#e34d3e",
        "panel": "#fffef8", "stroke": 9, "head": "circle", "body": "line",
        "joints": "none", "face": "eyes", "extras": (),
    },
    {
        "id": "bold", "name_ar": "العريض", "name_en": "Bold",
        "description": "خطوط سميكة مستديرة ورأس مربع بتباين عالٍ",
        "paper": "#fff9f1", "ink": "#1d2630", "accent": "#ff785a",
        "panel": "#ffe9d8", "stroke": 19, "head": "square", "body": "line",
        "joints": "circle", "face": "eyes", "extras": ("scarf",),
    },
    {
        "id": "neon", "name_ar": "النيون", "name_en": "Neon",
        "description": "خلفية داكنة وحدود متوهجة بسماوي ووردي",
        "paper": "#101827", "ink": "#52c9ba", "accent": "#ff5fa2",
        "panel": "#1b2740", "stroke": 10, "head": "capsule", "body": "glow",
        "joints": "circle", "face": "visor", "extras": ("aura", "wristband"),
    },
    {
        "id": "robot", "name_ar": "الروبوت", "name_en": "Robot",
        "description": "رأس مستطيل، هوائي، ومفاصل صفائح معدنية",
        "paper": "#eff4f7", "ink": "#18334b", "accent": "#4b89d1",
        "panel": "#dce8f2", "stroke": 15, "head": "block", "body": "segment",
        "joints": "bolt", "face": "pixel_eyes", "extras": ("antenna", "backpack"),
    },
    {
        "id": "sketch", "name_ar": "قلم الرصاص", "name_en": "Sketch",
        "description": "خطوط مزدوجة مرتجفة كأنها رسمة على دفتر",
        "paper": "#faf7ee", "ink": "#4a443d", "accent": "#c2813c",
        "panel": "#f0eadc", "stroke": 6, "head": "circle", "body": "sketch",
        "joints": "none", "face": "eyes", "extras": ("hat",),
    },
    {
        "id": "ribbon", "name_ar": "الشريط", "name_en": "Ribbon",
        "description": "جسم على شكل شريط ممتلئ بدون مفاصل مرسومة",
        "paper": "#f4f3ef", "ink": "#233b46", "accent": "#1fa38b",
        "panel": "#e2eeeb", "stroke": 14, "head": "circle", "body": "ribbon",
        "joints": "none", "face": "eyes", "extras": ("badge",),
    },
    {
        "id": "minimal", "name_ar": "البسيط", "name_en": "Minimal",
        "description": "أقل عدد من الخطوط، بلا وجه ولا إكسسوارات",
        "paper": "#ffffff", "ink": "#111111", "accent": "#888888",
        "panel": "#f2f2f2", "stroke": 5, "head": "circle", "body": "line",
        "joints": "none", "face": "none", "extras": (),
    },
    {
        "id": "professor", "name_ar": "الأستاذ", "name_en": "Professor",
        "description": "نظارات، ربطة عنق، ولوحة شرح بألوان هادئة",
        "paper": "#f7f2e7", "ink": "#2f3a34", "accent": "#8a5a2b",
        "panel": "#efe4cc", "stroke": 10, "head": "circle", "body": "line",
        "joints": "dot", "face": "glasses", "extras": ("tie", "book"),
    },
    {
        "id": "astro", "name_ar": "رائد الفضاء", "name_en": "Astro",
        "description": "خوذة بزجاج عاكس وحقيبة ظهر برتقالية",
        "paper": "#0d1b2a", "ink": "#e8eef7", "accent": "#ff8c42",
        "panel": "#1b2d43", "stroke": 13, "head": "helmet", "body": "segment",
        "joints": "circle", "face": "helmet_visor", "extras": ("backpack", "badge"),
    },
    {
        "id": "shadow", "name_ar": "الظل", "name_en": "Shadow",
        "description": "صورة ظلية داكنة ممتلئة مع هالة خفيفة",
        "paper": "#e8e4dc", "ink": "#20252b", "accent": "#c9553f",
        "panel": "#d9d4ca", "stroke": 22, "head": "circle", "body": "silhouette",
        "joints": "none", "face": "none", "extras": ("aura", "cape"),
    },
    {
        "id": "pixel", "name_ar": "البيكسل", "name_en": "Pixel",
        "description": "خطوات مربعة بأسلوب ثماني البتات",
        "paper": "#1b1b2f", "ink": "#f4d35e", "accent": "#ee964b",
        "panel": "#2d2b4e", "stroke": 12, "head": "pixel_head", "body": "pixel",
        "joints": "square", "face": "pixel_eyes", "extras": ("cap",),
    },
    {
        "id": "ink", "name_ar": "الحبري", "name_en": "Ink",
        "description": "حبر أسود رفيع مع لمسات ذهبية وعمامة",
        "paper": "#fbf6e9", "ink": "#191919", "accent": "#b8912f",
        "panel": "#f2e8cc", "stroke": 7, "head": "hex", "body": "line",
        "joints": "dot", "face": "eyes", "extras": ("turban", "badge"),
    },
)
TEMPLATE_IDS = [item["id"] for item in TEMPLATES]
_BY_ID = {item["id"]: item for item in TEMPLATES}


def catalog():
    """Return the template catalog as JSON-safe dictionaries."""
    return [dict(item, extras=list(item["extras"])) for item in TEMPLATES]


def get_template(template_id):
    if template_id not in _BY_ID:
        raise ValueError(f"Unknown character template: {template_id!r}. Choose from {TEMPLATE_IDS}")
    return _BY_ID[template_id]


def choice_list():
    """Numbered list for a notebook dropdown or a plain printed menu."""
    return [f"{i + 1:02d}. {item['id']} — {item['name_ar']}: {item['description']}"
            for i, item in enumerate(TEMPLATES)]


def pose_points(pose="present", phase=0.0, width=360, height=520, actor_box=None):
    """18 joint positions in absolute pixels for a gesture and animation phase."""
    if pose not in POSE_OFFSETS:
        raise ValueError(f"Unknown pose: {pose!r}. Choose from {POSES}")
    if actor_box is None:
        actor_box = (width * 0.10, height * 0.08, width * 0.90, height * 0.88)
    x0, y0, x1, y1 = actor_box
    sway = math.sin(2 * math.pi * phase)
    # Small continuous motion; keep head and torso aligned to avoid joint tearing.
    breathe = 0.004 * math.sin(2 * math.pi * phase)
    points = {}
    for joint, (nx, ny) in BASE_POSE.items():
        dx, dy = POSE_OFFSETS[pose].get(joint, (0.0, 0.0))
        if pose == "wave" and joint.startswith(("wrist_right", "elbow_right")):
            dx += 0.02 * sway
        if pose == "celebrate" and joint.startswith("wrist_"):
            dy += 0.02 * abs(sway)
        if pose == "step" and joint.startswith(("knee_", "ankle_", "foot_")):
            dy += 0.01 * sway
        if joint in ("neck", "head_center", "head_top", "head_bottom", "shoulder_left", "shoulder_right"):
            dy += breathe
        points[joint] = (x0 + (nx + dx) * (x1 - x0), y0 + (ny + dy) * (y1 - y0))
    return points


def _font(size):
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def _segment(draw, start, end, width, ink, accent, body):
    """Draw one bone in the template's body style."""
    a, b = tuple(start), tuple(end)
    if body == "sketch":
        for shift in (-1.5, 1.5):
            draw.line((a[0] + shift, a[1] - shift, b[0] - shift, b[1] + shift),
                      fill=ink, width=max(2, width))
        return
    if body == "glow":
        draw.line((*a, *b), fill=accent, width=width + 8)
        draw.line((*a, *b), fill=ink, width=max(2, width - 2))
        return
    if body == "pixel":
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(length / 9))
        for i in range(steps + 1):
            t = i / steps
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            draw.rectangle((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=ink)
        return
    if body == "segment":
        draw.line((*a, *b), fill=ink, width=width, joint="curve")
        for t in (0.32, 0.68):
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            draw.rounded_rectangle((x - width * 0.8, y - width * 0.45,
                                    x + width * 0.8, y + width * 0.45),
                                   radius=3, fill=accent, outline=ink, width=2)
        return
    if body == "silhouette":
        draw.line((*a, *b), fill=ink, width=width, joint="curve")
        draw.ellipse((a[0] - width / 2, a[1] - width / 2, a[0] + width / 2, a[1] + width / 2),
                     fill=ink)
        draw.ellipse((b[0] - width / 2, b[1] - width / 2, b[0] + width / 2, b[1] + width / 2),
                     fill=ink)
        return
    draw.line((*a, *b), fill=ink, width=width, joint="curve")
    # Rounded endpoints hide gaps between connected segments.
    if body == "line":
        radius = width / 2
        for cx, cy in (a, b):
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=ink)
    if body == "ribbon":
        draw.line((*a, *b), fill=accent, width=max(2, width // 3))


def _joint(draw, point, size, ink, accent, kind):
    if kind == "none":
        return
    x, y = point
    if kind == "dot":
        draw.ellipse((x - size / 2, y - size / 2, x + size / 2, y + size / 2), fill=ink)
    elif kind == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=accent, outline=ink, width=2)
    elif kind == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=ink)
    elif kind == "bolt":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=accent, outline=ink, width=2)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=ink)


def _head(draw, points, template, emotion):
    head = points["head_center"]
    radius = (points["head_bottom"][1] - points["head_top"][1]) / 2
    ink, accent = template["ink"], template["accent"]
    paper = template["panel"] if template["id"] in ("neon", "astro", "pixel") else template["paper"]
    kind = template["head"]
    if kind == "square":
        draw.rounded_rectangle((head[0] - radius, head[1] - radius,
                                head[0] + radius, head[1] + radius),
                               radius=radius * 0.28, fill=paper, outline=ink, width=4)
    elif kind == "capsule":
        draw.rounded_rectangle((head[0] - radius * 0.82, head[1] - radius,
                                head[0] + radius * 0.82, head[1] + radius),
                               radius=radius * 0.82, fill=paper, outline=ink, width=4)
    elif kind in ("block", "pixel_head"):
        draw.rectangle((head[0] - radius, head[1] - radius * 0.8,
                        head[0] + radius, head[1] + radius * 0.8),
                       fill=paper, outline=ink, width=4)
    elif kind == "hex":
        vertices = [(head[0] + radius * math.cos(math.radians(60 * i - 30)),
                     head[1] + radius * math.sin(math.radians(60 * i - 30))) for i in range(6)]
        draw.polygon(vertices, fill=paper, outline=ink, width=4)
    elif kind == "helmet":
        draw.ellipse((head[0] - radius * 1.12, head[1] - radius * 1.12,
                      head[0] + radius * 1.12, head[1] + radius * 1.12),
                     fill=paper, outline=ink, width=4)
    else:
        draw.ellipse((head[0] - radius, head[1] - radius,
                      head[0] + radius, head[1] + radius),
                     fill=paper, outline=ink, width=4)

    face = template["face"]
    if face == "visor":
        draw.rounded_rectangle((head[0] - radius * 0.62, head[1] - radius * 0.18,
                                head[0] + radius * 0.62, head[1] + radius * 0.35),
                               radius=8, fill=accent, outline=ink, width=2)
    elif face == "helmet_visor":
        draw.rounded_rectangle((head[0] - radius * 0.78, head[1] - radius * 0.35,
                                head[0] + radius * 0.78, head[1] + radius * 0.38),
                               radius=radius * 0.35, fill=accent, outline=ink, width=3)
    elif face == "pixel_eyes":
        for dx in (-0.32, 0.32):
            cx, cy = head[0] + radius * dx, head[1] - radius * 0.1
            draw.rectangle((cx - 6, cy - 6, cx + 6, cy + 6), fill=accent)
        draw.rectangle((head[0] - radius * 0.25, head[1] + radius * 0.4,
                        head[0] + radius * 0.25, head[1] + radius * 0.5), fill=ink)
    elif face == "glasses":
        for dx in (-0.34, 0.34):
            cx, cy = head[0] + radius * dx, head[1] - radius * 0.08
            draw.ellipse((cx - radius * 0.24, cy - radius * 0.18,
                          cx + radius * 0.24, cy + radius * 0.18), outline=ink, width=3)
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=ink)
        draw.line((head[0] - radius * 0.1, head[1] - radius * 0.08,
                   head[0] + radius * 0.1, head[1] - radius * 0.08), fill=ink, width=3)
        draw.arc((head[0] - radius * 0.3, head[1] + radius * 0.15,
                  head[0] + radius * 0.3, head[1] + radius * 0.6), 10, 170, fill=ink, width=3)
    elif face == "eyes":
        for dx in (-0.32, 0.32):
            cx, cy = head[0] + radius * dx, head[1] - radius * 0.1
            draw.ellipse((cx - 5, cy - 7, cx + 5, cy + 7), fill=paper, outline=ink, width=2)
            draw.ellipse((cx - 2, cy - 3, cx + 2, cy + 3), fill=ink)
        if emotion in ("happy", "surprised"):
            draw.arc((head[0] - radius * 0.28, head[1] + radius * 0.12,
                      head[0] + radius * 0.28, head[1] + radius * 0.52), 10, 170,
                     fill=ink, width=3)
        elif emotion == "concerned":
            draw.arc((head[0] - radius * 0.28, head[1] + radius * 0.42,
                      head[0] + radius * 0.28, head[1] + radius * 0.1), 190, 350,
                     fill=ink, width=3)
        else:
            draw.line((head[0] - radius * 0.22, head[1] + radius * 0.42,
                       head[0] + radius * 0.22, head[1] + radius * 0.42), fill=ink, width=3)

    extras = template["extras"]
    if "beret" in extras:
        draw.ellipse((head[0] - radius * 1.05, head[1] - radius * 1.35,
                      head[0] + radius * 0.75, head[1] - radius * 0.35),
                     fill=accent, outline=ink, width=3)
        draw.ellipse((head[0] - 4, head[1] - radius * 1.45,
                      head[0] + 4, head[1] - radius * 1.3), fill=ink)
    if "hat" in extras:
        draw.line((head[0] - radius * 1.15, head[1] - radius * 0.85,
                   head[0] + radius * 1.15, head[1] - radius * 0.85), fill=ink, width=6)
        draw.rounded_rectangle((head[0] - radius * 0.62, head[1] - radius * 1.5,
                                head[0] + radius * 0.62, head[1] - radius * 0.82),
                               radius=6, fill=accent, outline=ink, width=3)
    if "cap" in extras:
        draw.pieslice((head[0] - radius, head[1] - radius * 1.2,
                       head[0] + radius, head[1] + radius * 0.1), 180, 360, fill=accent,
                      outline=ink, width=3)
        draw.rectangle((head[0] - radius * 0.2, head[1] - radius * 0.98,
                        head[0] + radius * 1.25, head[1] - radius * 0.86), fill=ink)
    if "turban" in extras:
        draw.arc((head[0] - radius * 1.05, head[1] - radius * 1.3,
                  head[0] + radius * 1.05, head[1] - radius * 0.2), 185, 355,
                 fill=accent, width=9)
        draw.arc((head[0] - radius * 0.85, head[1] - radius * 1.15,
                  head[0] + radius * 0.85, head[1] - radius * 0.35), 185, 355,
                 fill=ink, width=3)
    if "antenna" in extras:
        draw.line((head[0], head[1] - radius, head[0], head[1] - radius * 1.5),
                  fill=ink, width=4)
        draw.ellipse((head[0] - 7, head[1] - radius * 1.65,
                      head[0] + 7, head[1] - radius * 1.45), fill=accent, outline=ink, width=2)


def draw_character(draw, points, template_id="classic", emotion="happy", phase=0.0):
    """Draw the chosen character onto an existing PIL ImageDraw surface."""
    template = get_template(template_id)
    ink, accent = template["ink"], template["accent"]
    width = max(3, template["stroke"])
    body = template["body"]

    if "aura" in template["extras"]:
        hx, hy = points["head_center"]
        draw.ellipse((hx - 125, hy - 125, hx + 125, hy + 125), fill=template["panel"])
    if "backpack" in template["extras"]:
        sl, sr = points["shoulder_left"], points["shoulder_right"]
        draw.rounded_rectangle((sl[0] - width * 1.4, sl[1] - 6,
                                sr[0] - width * 0.2, points["torso_mid"][1] + 20),
                               radius=12, fill=template["panel"], outline=ink, width=3)
    if "cape" in template["extras"]:
        draw.polygon([points["shoulder_left"], points["shoulder_right"],
                      (points["foot_right"][0] + 26, points["foot_right"][1]),
                      (points["foot_left"][0] - 26, points["foot_left"][1])], fill=accent)

    shadow_y = max(points["foot_left"][1], points["foot_right"][1]) + 10
    draw.ellipse((points["foot_left"][0] - 70, shadow_y - 10,
                  points["foot_right"][0] + 70, shadow_y + 14), fill=template["panel"])

    for side in ("left", "right"):
        _segment(draw, points["hip"], points[f"knee_{side}"], width + 2, ink, accent, body)
        _segment(draw, points[f"knee_{side}"], points[f"ankle_{side}"], width + 2, ink, accent, body)
        _segment(draw, points[f"ankle_{side}"], points[f"foot_{side}"], width, ink, accent, body)
        _segment(draw, points[f"shoulder_{side}"], points[f"elbow_{side}"], width, ink, accent, body)
        _segment(draw, points[f"elbow_{side}"], points[f"wrist_{side}"], width, ink, accent, body)

    if body == "ribbon":
        draw.polygon([points["shoulder_left"], points["shoulder_right"],
                      points["hip"], points["torso_mid"]], fill=ink)
        draw.line((*points["neck"], *points["hip"]), fill=accent, width=4)
    else:
        _segment(draw, points["hip"], points["torso_mid"], width + 4, ink, accent, body)
        _segment(draw, points["torso_mid"], points["neck"], width + 4, ink, accent, body)
        _segment(draw, points["shoulder_left"], points["shoulder_right"], width, ink, accent, body)
    _segment(draw, points["neck"], points["head_bottom"], width, ink, accent, body)

    if "tie" in template["extras"]:
        n = points["neck"]
        draw.polygon([(n[0] - 6, n[1] + 4), (n[0] + 6, n[1] + 4),
                      (n[0], n[1] + 46)], fill=accent)
    if "scarf" in template["extras"]:
        n = points["neck"]
        draw.rounded_rectangle((n[0] - 32, n[1] - 8, n[0] + 32, n[1] + 12),
                               radius=6, fill=accent)
    if "badge" in template["extras"]:
        c = points["torso_mid"]
        draw.ellipse((c[0] - 13, c[1] - 13, c[0] + 13, c[1] + 13),
                     fill=accent, outline=ink, width=2)
    if "book" in template["extras"]:
        w = points["wrist_left"]
        draw.rounded_rectangle((w[0] - 40, w[1] - 26, w[0] + 26, w[1] + 18),
                               radius=4, fill=template["panel"], outline=ink, width=3)
        draw.line((w[0] - 7, w[1] - 24, w[0] - 7, w[1] + 16), fill=accent, width=2)

    for joint in ("elbow_left", "elbow_right", "wrist_left", "wrist_right",
                  "knee_left", "knee_right", "hip", "neck"):
        _joint(draw, points[joint], max(3, width * 0.42), ink, accent, template["joints"])
    if "gloves" in template["extras"] or "wristband" in template["extras"]:
        for side in ("left", "right"):
            w = points[f"wrist_{side}"]
            draw.ellipse((w[0] - 11, w[1] - 11, w[0] + 11, w[1] + 11),
                         fill=accent if "wristband" in template["extras"] else template["panel"],
                         outline=ink, width=3)

    _head(draw, points, template, emotion)
    return template


def render_preview(template_id="classic", pose="present", emotion="happy", phase=0.0,
                   size=(360, 520), output=None):
    """One PNG card showing the chosen shape and its name. Returns a PIL Image."""
    template = get_template(template_id)
    width, height = size
    image = Image.new("RGB", size, template["paper"])
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 40):
        draw.line((x, 0, x, height), fill=template["panel"], width=1)
    for y in range(0, height, 40):
        draw.line((0, y, width, y), fill=template["panel"], width=1)
    draw.rounded_rectangle((14, 14, width - 14, height - 14), radius=18,
                           outline=template["accent"], width=3)
    points = pose_points(pose, phase, width, height,
                         actor_box=(width * 0.13, height * 0.14, width * 0.87, height * 0.79))
    draw_character(draw, points, template_id, emotion, phase)
    draw.rounded_rectangle((20, height - 76, width - 20, height - 20), radius=12,
                           fill=template["panel"])
    draw_text(draw, (width / 2, height - 68), template["name_ar"], _font(24), "rtl",
              fill=template["ink"], anchor="mt")
    draw.text((width / 2, height - 36), f"{template_id}  /  {pose}", font=_font(15),
              fill=template["accent"], anchor="mt")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    return image


def render_gallery(output, columns=4, cell=(270, 390)):
    """Contact sheet of every selectable shape. One PNG, no external assets."""
    rows = math.ceil(len(TEMPLATES) / columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#ffffff")
    for index, template in enumerate(TEMPLATES):
        card = render_preview(template["id"], size=(cell[0], cell[1]))
        sheet.paste(card, ((index % columns) * cell[0], (index // columns) * cell[1]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return sheet


def template_diversity(size=(180, 260)):
    """Real pixel-level measurement that the catalog is not 12 copies of one shape."""
    arrays = []
    digests = []
    for template_id in TEMPLATE_IDS:
        image = render_preview(template_id, size=size).convert("RGB")
        array = np.asarray(image, dtype=np.int16)
        arrays.append(array)
        digests.append(hashlib.sha256(array.tobytes()).hexdigest()[:16])
    distances = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            distances.append(float(np.abs(arrays[i] - arrays[j]).mean()))
    return {
        "templates": len(TEMPLATES),
        "unique_render_sha256": len(set(digests)),
        "min_pairwise_mean_abs_pixel_diff": round(min(distances), 2),
        "mean_pairwise_mean_abs_pixel_diff": round(float(np.mean(distances)), 2),
        "hashes": digests,
    }


def lock_template(template_id, path):
    """Persist an agreed shape so all later cells/stages reuse exactly one character."""
    template = get_template(template_id)
    preview = render_preview(template_id, size=(180, 260)).convert("RGB")
    signature = hashlib.sha256(np.asarray(preview).tobytes()).hexdigest()
    record = {
        "template_id": template_id,
        "name_ar": template["name_ar"],
        "render_sha256": signature,
        "catalog_sha256": hashlib.sha256(
            json.dumps(catalog(), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "poses": list(POSES),
    }
    write_json(path, record)
    return record


def load_lock(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No locked character at {path}. Run the preview + lock cell first."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    get_template(record["template_id"])
    expected = hashlib.sha256(
        json.dumps(catalog(), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if record.get("catalog_sha256") != expected:
        raise ValueError("Template catalog changed since locking; re-run the selection cell.")
    preview = render_preview(record["template_id"], size=(180, 260)).convert("RGB")
    signature = hashlib.sha256(np.asarray(preview).tobytes()).hexdigest()
    if record.get("render_sha256") != signature:
        raise ValueError("Character preview changed; preview and confirm again.")
    return record


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", help="Write a PNG contact sheet of all shapes")
    parser.add_argument("--preview", help="Write one preview PNG for --template")
    parser.add_argument("--template", default=TEMPLATE_IDS[0])
    parser.add_argument("--pose", default="present")
    args = parser.parse_args()
    if args.gallery:
        render_gallery(args.gallery)
        print("gallery:", args.gallery)
    if args.preview:
        render_preview(args.template, args.pose, output=args.preview)
        print("preview:", args.preview, args.template, args.pose)
    print(json.dumps({"ids": TEMPLATE_IDS, "diversity": template_diversity()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
