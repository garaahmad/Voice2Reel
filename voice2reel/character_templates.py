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


def _smooth_limb(draw, a, b, thickness, color):
    """Capsule-like segment, with rounded ends and no exposed joint markers."""
    thickness = max(2, int(round(thickness)))
    draw.line((*a, *b), fill=color, width=thickness, joint="curve")
    r = thickness / 2
    for x, y in (a, b):
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)


def _two_bone_joint(root, target, upper, lower, bend=1):
    """Place an elbow/knee while preserving both limb lengths."""
    dx, dy = target[0]-root[0], target[1]-root[1]
    distance = max(1e-6, math.hypot(dx, dy))
    distance = min(max(distance, abs(upper-lower)+0.01), upper+lower-0.01)
    ux, uy = dx/max(1e-6, math.hypot(dx, dy)), dy/max(1e-6, math.hypot(dx, dy))
    along = (upper*upper - lower*lower + distance*distance)/(2*distance)
    offset = math.sqrt(max(0.0, upper*upper - along*along))
    return (root[0] + ux*along - uy*offset*bend,
            root[1] + uy*along + ux*offset*bend)


def draw_character(draw, points, template_id="classic", emotion="happy", phase=0.0):
    """Premium, consistent character silhouette for every visual theme.

    Uses the supplied pose targets but reconstructs arms/legs with two-bone IK.
    No hats, exposed joint dots, robotic plates or disconnected segments.
    """
    template = get_template(template_id)
    ink, accent = template["ink"], template["accent"]
    center = points["head_center"]
    head_radius = max(8.0, abs(points["head_bottom"][1]-points["head_top"][1])/2)
    shoulder_l, shoulder_r = points["shoulder_left"], points["shoulder_right"]
    shoulder = ((shoulder_l[0]+shoulder_r[0])/2, (shoulder_l[1]+shoulder_r[1])/2)
    hip = points["hip"]
    body_height = max(35.0, abs(hip[1]-shoulder[1]))
    thickness = max(3, round(body_height*0.065))
    arm_thickness = max(3, round(thickness*0.78))
    leg_thickness = max(3, round(thickness*0.92))

    # A restrained accent shirt makes all themes feel like one designed series.
    torso_top = (shoulder[0], shoulder[1]+body_height*0.02)
    torso_bottom = (hip[0], hip[1]-body_height*0.02)
    _smooth_limb(draw, torso_top, torso_bottom, thickness*1.25, ink)
    _smooth_limb(draw, shoulder_l, shoulder_r, thickness*0.86, ink)
    chest = (shoulder[0], shoulder[1]+body_height*0.23)
    r = thickness*0.50
    draw.ellipse((chest[0]-r, chest[1]-r, chest[0]+r, chest[1]+r), fill=accent)

    for side in ("left", "right"):
        root = points[f"shoulder_{side}"]
        wrist_target = points[f"wrist_{side}"]
        original_elbow = points[f"elbow_{side}"]
        # Fit lengths to each character scale; preserve reachable targets.
        arm_length = max(body_height*0.46,
                         math.hypot(original_elbow[0]-root[0], original_elbow[1]-root[1]))
        forearm_length = max(body_height*0.43,
                             math.hypot(wrist_target[0]-original_elbow[0],
                                        wrist_target[1]-original_elbow[1]))
        dx, dy = wrist_target[0]-root[0], wrist_target[1]-root[1]
        dist = math.hypot(dx, dy)
        max_reach = (arm_length+forearm_length)*0.97
        if dist > max_reach:
            wrist_target = (root[0]+dx*max_reach/dist,
                            root[1]+dy*max_reach/dist)
        elbow = _two_bone_joint(root, wrist_target, arm_length, forearm_length,
                                bend=-1 if side=="left" else 1)
        _smooth_limb(draw, root, elbow, arm_thickness, ink)
        _smooth_limb(draw, elbow, wrist_target, arm_thickness, ink)
        # Small filled mitten rather than a disconnected circle.
        hr = arm_thickness*0.68
        draw.ellipse((wrist_target[0]-hr, wrist_target[1]-hr,
                      wrist_target[0]+hr, wrist_target[1]+hr), fill=ink)

        # Leg chains use one continuous silhouette with small natural feet.
        knee = points[f"knee_{side}"]
        ankle = points[f"ankle_{side}"]
        foot = points[f"foot_{side}"]
        _smooth_limb(draw, hip, knee, leg_thickness, ink)
        _smooth_limb(draw, knee, ankle, leg_thickness, ink)
        toe_direction = -1 if side=="left" else 1
        toe = (ankle[0]+toe_direction*body_height*0.15, ankle[1]+body_height*0.025)
        _smooth_limb(draw, ankle, toe, leg_thickness*0.78, ink)

    # Head is a clean circle, independent of old head variants and accessories.
    outline_width = max(2, round(thickness*0.40))
    draw.ellipse((center[0]-head_radius, center[1]-head_radius,
                  center[0]+head_radius, center[1]+head_radius),
                 fill=template["paper"], outline=ink, width=outline_width)
    # Minimal expressive face, with symmetric eyes and subtle smile.
    eye_y = center[1]-head_radius*0.08
    eye_r = max(1.5, head_radius*0.055)
    for sign in (-1, 1):
        eye_x = center[0]+sign*head_radius*0.31
        draw.ellipse((eye_x-eye_r, eye_y-eye_r, eye_x+eye_r, eye_y+eye_r), fill=ink)
    mouth_box = (center[0]-head_radius*0.27, center[1]+head_radius*0.04,
                 center[0]+head_radius*0.27, center[1]+head_radius*0.40)
    draw.arc(mouth_box, 5, 175, fill=ink, width=max(2, outline_width-1))
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
