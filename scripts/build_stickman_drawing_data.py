"""Generate a very large, deterministic stickman-DRAWING training corpus.

The current renderer is deterministic vector code, not a neural image generator.
Therefore this corpus teaches *how to draw*: 2D joint coordinates from two-bone
IK, proportions, stroke style, layer order, camera placement, safe zones, easing,
and named visual defects. A small CPU model in ``voice2reel/stickman_artist.py``
is fitted on these rows; a real image-diffusion renderer would need a GPU and a
separate training loop.

All rows are assistant-authored synthetic CC0-1.0 material. No expert review,
no scraped artwork, and no claim about subjective professional aesthetics.
"""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets" / "stickman"
ASSETS = ROOT / "voice2reel" / "assets"
SEED = 20261110
COUNTS = {"train": 50_000, "validation": 5_000, "test": 5_000}
CANVAS = (540, 960)
MARGIN = 24

GESTURES = {
    "present": ((109, 653), (284, 570)),
    "point": ((109, 653), (295, 518)),
    "think": ((109, 653), (228, 456)),
    "shrug": ((72, 534), (280, 526)),
    "thumbs_up": ((110, 650), (278, 519)),
    "wave": ((108, 650), (270, 395)),
}
CAMERA_ZOOM = {"wide": 1.0, "medium": 1.13, "close": 1.5}
CAMERAS = list(CAMERA_ZOOM)
EMOTIONS = ["neutral", "happy", "surprised", "concerned", "thinking"]
INTENTS = ["hook", "explain", "close"]
VISUALS = ["none", "list", "flow", "compare", "code"]
CAPTIONS = [
    "خطوة واضحة", "ابدأ بمثال", "راجع النتيجة", "قسّم المسألة", "افحص المدخل",
    "لا تكرر الكود", "اكتب اسما مفهوما", "احفظ نسخة", "قارن الحالتين", "صحح الترتيب",
    "اختبر جزءا صغيرا", "اقرأ الرسالة", "ثبّت القيمة", "بسّط القرار", "ارسم الفكرة",
    "الفرق بسيط", "قبل وبعد", "تحقق أولا", "خلاصة المشهد", "نتيجة قابلة للقياس",
]
DEFECTS = [
    "camera_jump", "gesture_flicker", "jitter", "cut_too_fast", "harsh_flash",
    "overlap", "dense_subtitle", "abrupt_easing", "disconnected_joint",
    "bad_proportions", "offscreen_actor", "cropped_head", "foot_lift",
]
STYLE = {
    "canvas": list(CANVAS), "paper": "#f3f2ec", "ink": "#172023", "accent": "#e34d3e",
    "panel": "#fffef8", "stroke_width": 14, "head_radius": 72,
    "upper_arm": 91, "lower_arm": 98, "upper_leg": 96, "lower_leg": 98,
    "shoulder_width": 33, "curve_steps": 20, "supersampling": 2,
    "beret": True, "gloves": True, "shadow": True,
}
SAFE_ZONES = {
    "headline": [0, 0, 540, 250], "subtitle": [0, 860, 540, 960],
    "panel": [295, 430, 540, 820], "actor": [24, 274, 271, 836],
}
LIMITS = {
    "max_camera_step_per_scene": 1, "max_gesture_switches_per_second": 0.5,
    "max_jitter_amplitude": 2.0, "min_scene_seconds": 2.5,
    "max_flash_alpha": 0.12, "max_subtitle_words": 4,
}
KEYPOINT_ORDER = [
    "hip", "shoulder_left", "shoulder_right", "head_center", "head_top", "head_bottom",
    "neck", "torso_mid",
    "elbow_left", "elbow_right", "wrist_left", "wrist_right",
    "knee_left", "knee_right", "ankle_left", "ankle_right", "foot_left", "foot_right",
]
FEATURE_NAMES = [
    "camera_step", "gesture_switches_per_second", "jitter_amplitude", "scene_seconds",
    "flash_alpha", "overlap_count", "subtitle_words", "easing_smooth",
    "proportion_error", "joint_gap", "headline_overlap", "foot_lift",
    "offscreen", "head_crop",
]


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def mix(a, b, u):
    return a + (b - a) * u


def joint(root, target, upper, lower, bend=1):
    """Two-bone IK; same geometry as the deterministic renderer."""
    dx, dy = target[0] - root[0], target[1] - root[1]
    distance = max(abs(upper - lower) + 1e-6, min(math.hypot(dx, dy), upper + lower - 1e-6))
    angle = math.atan2(dy, dx)
    offset = math.acos(max(-1, min(1, (upper**2 + distance**2 - lower**2) / (2 * upper * distance))))
    elbow = (root[0] + upper * math.cos(angle + bend * offset),
             root[1] + upper * math.sin(angle + bend * offset))
    end = (root[0] + distance * math.cos(angle), root[1] + distance * math.sin(angle))
    return elbow, end


# Local copies of motion-plan geometry: avoids rereading the vocabulary JSON
# 60,000 times and keeps generation fast and byte-stable.
def overlaps(a, b, margin=0):
    return not (
        a[2] + margin <= b[0] or b[2] + margin <= a[0]
        or a[3] + margin <= b[1] or b[3] + margin <= a[1]
    )


def safe_box(camera, panel_visible):
    headline, subtitle, panel = SAFE_ZONES["headline"], SAFE_ZONES["subtitle"], SAFE_ZONES["panel"]
    return (
        MARGIN,
        headline[3] + MARGIN,
        (panel[0] - MARGIN) if panel_visible else (CANVAS[0] - MARGIN),
        subtitle[1] - MARGIN,
    )


def fit(size, position, box):
    bw, bh = box[2] - box[0], box[3] - box[1]
    # Slight slack: rounding joint coordinates must never push the actor into
    # the headline/subtitle safety margin.
    scale = math.floor(min(1.0, bw / size[0], bh / size[1]) * 0.995 * 10000) / 10000
    sw, sh = size[0] * scale, size[1] * scale
    x = max(box[0], min(box[2] - sw, position[0]))
    y = max(box[1], min(box[3] - sh, position[1]))
    return scale, x, y


def overlap_count(boxes):
    keys = list(boxes)
    total = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a == "headline" and b == "subtitle":
                continue
            margin = 0 if {a, b} == {"actor", "panel"} else MARGIN
            total += int(overlaps(boxes[a], boxes[b], margin))
    return total


def rule_violations(features, easing):
    f = dict(zip(FEATURE_NAMES, features))
    problems = []
    if f["camera_step"] > LIMITS["max_camera_step_per_scene"]:
        problems.append("camera_jump")
    if f["gesture_switches_per_second"] > LIMITS["max_gesture_switches_per_second"]:
        problems.append("gesture_flicker")
    if f["jitter_amplitude"] > LIMITS["max_jitter_amplitude"]:
        problems.append("jitter")
    if f["scene_seconds"] < LIMITS["min_scene_seconds"]:
        problems.append("cut_too_fast")
    if f["flash_alpha"] > LIMITS["max_flash_alpha"]:
        problems.append("harsh_flash")
    if f["overlap_count"] > 0:
        problems.append("overlap")
    if f["subtitle_words"] > LIMITS["max_subtitle_words"]:
        problems.append("dense_subtitle")
    if easing != "smooth":
        problems.append("abrupt_easing")
    return problems


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_point(point):
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def adjacent_previous(camera, rng):
    index = CAMERAS.index(camera)
    choices = [CAMERAS[max(0, index - 1)], CAMERAS[index], CAMERAS[min(2, index + 1)]]
    return rng.choice(choices)


def camera_step(previous, current):
    return abs(CAMERAS.index(current) - CAMERAS.index(previous))


def raw_keypoints(gesture, previous_gesture, phase):
    u = ease(phase)
    lean = mix(0, 4 if gesture == "point" else -2, u) + 1.6 * math.sin(phase * 2 * math.pi)
    breath = 1.4 * math.sin(phase * 2 * math.pi * 2.1)
    hip = (164 + 2 * math.sin(phase * 2 * math.pi * 0.7), 650)
    shoulders = [(148 + lean, 498 + breath), (181 + lean, 499 + breath)]
    previous, current = GESTURES[previous_gesture], GESTURES[gesture]
    shoulder_center = ((shoulders[0][0] + shoulders[1][0]) / 2,
                       (shoulders[0][1] + shoulders[1][1]) / 2)
    radius = STYLE["head_radius"] / 2
    head_center = (164 + lean, shoulder_center[1] - 22 - radius)
    points = {
        "hip": hip,
        "shoulder_left": shoulders[0],
        "shoulder_right": shoulders[1],
        "torso_mid": (mix(hip[0], shoulder_center[0], 0.55), 575 + breath / 2),
        "head_center": head_center,
        # Reserve the beret's upper edge in the actor bounds so a close camera
        # never crops it under the headline. The neck and head bottom are
        # explicit joints: a professional silhouette is fully connected.
        "head_top": (head_center[0], head_center[1] - radius * 1.25),
        "head_bottom": (head_center[0], head_center[1] + radius),
        "neck": (shoulder_center[0], shoulder_center[1] - 12),
    }
    for side, name in enumerate(("left", "right")):
        wrist = (
            mix(previous[side][0], current[side][0], u),
            mix(previous[side][1], current[side][1], u) + breath,
        )
        elbow, end = joint(shoulders[side], wrist, STYLE["upper_arm"], STYLE["lower_arm"], 1 if side else -1)
        points[f"elbow_{name}"] = elbow
        points[f"wrist_{name}"] = end
    for side, name in enumerate(("left", "right")):
        ankle_target = (127 if side == 0 else 214, 834)
        knee, ankle = joint(hip, ankle_target, STYLE["upper_leg"], STYLE["lower_leg"], 1 if side else -1)
        points[f"knee_{name}"] = knee
        points[f"ankle_{name}"] = ankle
        points[f"foot_{name}"] = (ankle[0] + (26 if side else -26), ankle[1] + 12)
    return {key: round_point(value) for key, value in points.items()}


def place_keypoints(points, camera, panel_visible):
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    zoom = CAMERA_ZOOM[camera]
    size = ((maxx - minx) * zoom, (maxy - miny) * zoom)
    box = safe_box(camera, panel_visible)
    scale, x, y = fit(size, (minx * zoom, miny * zoom), box)
    placed = {
        key: [
            round((p[0] - minx) * zoom * scale + x, 2),
            round((p[1] - miny) * zoom * scale + y, 2),
        ]
        for key, p in points.items()
    }
    return placed, scale


def boxes_from(points, panel_visible):
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    boxes = {
        "headline": list(SAFE_ZONES["headline"]),
        "subtitle": list(SAFE_ZONES["subtitle"]),
        "actor": [min(xs), min(ys), max(xs), max(ys)],
    }
    if panel_visible:
        boxes["panel"] = list(SAFE_ZONES["panel"])
    return boxes


def geometry_features(points, boxes, style, easing):
    actor, headline = boxes["actor"], boxes["headline"]
    head = points["head_center"]
    zoom, scale = style["camera_zoom"], style["actor_scale"]
    radius = style["head_radius"] * zoom * scale / 2
    expected_lower = style["lower_arm"] * zoom * scale
    left_ankle, right_ankle = points["ankle_left"], points["ankle_right"]
    arms = [(points[f"elbow_{side}"], points[f"wrist_{side}"]) for side in ("left", "right")]
    joint_gap = max(abs(math.dist(elbow, wrist) - expected_lower) for elbow, wrist in arms)
    proportion_error = (
        abs(style["head_radius"] / STYLE["head_radius"] - 1)
        + abs(style["upper_arm"] / STYLE["upper_arm"] - 1)
        + abs(style["lower_arm"] / STYLE["lower_arm"] - 1)
    ) / 3
    return {
        "camera_step": 0,
        "gesture_switches_per_second": 0.25,
        "jitter_amplitude": 1.0,
        "scene_seconds": 4.0,
        "flash_alpha": 0.05,
        "overlap_count": overlap_count(boxes),
        "subtitle_words": 4,
        "easing_smooth": 1 if easing == "smooth" else 0,
        "proportion_error": round(proportion_error, 3),
        "joint_gap": round(joint_gap, 3),
        "headline_overlap": int(actor[1] < headline[3] + MARGIN),
        "foot_lift": int(abs(left_ankle[1] - right_ankle[1]) > 8),
        "offscreen": int(actor[0] < 0 or actor[2] > CANVAS[0] or actor[1] < 0 or actor[3] > CANVAS[1]),
        "head_crop": int(head[1] - radius < headline[3]),
    }


def professional_row(index, rng):
    gesture = rng.choice(list(GESTURES))
    previous_gesture = rng.choice(list(GESTURES))
    camera = rng.choice(CAMERAS)
    previous_camera = adjacent_previous(camera, rng)
    emotion = rng.choice(EMOTIONS)
    intent = rng.choice(INTENTS)
    visual = rng.choice(VISUALS)
    panel_visible = visual != "none"
    phase = round(rng.random(), 3)
    duration = round(rng.uniform(2.6, 8.0), 2)
    jitter = round(rng.uniform(0.1, 1.8), 2)
    flash = round(rng.uniform(0.0, 0.10), 2)
    switches = round(rng.uniform(0.10, 0.45), 2)
    subtitle_words = rng.choice([3, 4])
    easing = "smooth"
    points, scale = place_keypoints(
        raw_keypoints(gesture, previous_gesture, phase), camera, panel_visible
    )
    boxes = boxes_from(points, panel_visible)
    style = dict(STYLE, actor_scale=round(scale, 4), camera_zoom=CAMERA_ZOOM[camera])
    features = geometry_features(points, boxes, style, easing)
    features.update({
        "camera_step": camera_step(previous_camera, camera),
        "gesture_switches_per_second": switches,
        "jitter_amplitude": jitter,
        "scene_seconds": duration,
        "flash_alpha": flash,
        "subtitle_words": subtitle_words,
        "overlap_count": overlap_count(boxes),
    })
    return {
        "id": f"sd_{index:06d}",
        "defect": None,
        "intent": intent,
        "caption": rng.choice(CAPTIONS),
        "gesture": gesture,
        "previous_gesture": previous_gesture,
        "emotion": emotion,
        "camera": camera,
        "previous_camera": previous_camera,
        "visual": visual,
        "panel_visible": panel_visible,
        "phase": phase,
        "easing": easing,
        "style": style,
        "keypoints": points,
        "boxes": boxes,
        "features": features,
    }


def shift_points(points, dx=0, dy=0, only=None):
    out = copy.deepcopy(points)
    for key, value in out.items():
        if only is None or key in only:
            value[0] = round(value[0] + dx, 2)
            value[1] = round(value[1] + dy, 2)
    return out


def apply_defect(row, defect, rng):
    r = copy.deepcopy(row)
    r["defect"] = defect
    f = dict(r["features"])
    style = dict(r["style"])
    points = dict(r["keypoints"])

    if defect == "camera_jump":
        f["camera_step"] = rng.choice([2, 3])
    elif defect == "gesture_flicker":
        f["gesture_switches_per_second"] = round(rng.uniform(0.6, 1.6), 2)
    elif defect == "jitter":
        f["jitter_amplitude"] = round(rng.uniform(2.2, 6.0), 2)
    elif defect == "cut_too_fast":
        f["scene_seconds"] = round(rng.uniform(1.0, 2.4), 2)
    elif defect == "harsh_flash":
        f["flash_alpha"] = round(rng.uniform(0.14, 0.45), 2)
    elif defect == "dense_subtitle":
        f["subtitle_words"] = rng.choice([5, 6, 8, 9])
    elif defect == "abrupt_easing":
        r["easing"] = rng.choice(["linear", "snap"])
        f["easing_smooth"] = 0
    elif defect == "overlap":
        points = shift_points(points, dx=120)
        f["overlap_count"] = 1
    elif defect == "disconnected_joint":
        # Displace one elbow and its wrist in different directions so the lower
        # arm is visibly detached, not merely a slightly bent silhouette.
        points = shift_points(points, dx=18, dy=9, only={"elbow_left"})
        points = shift_points(points, dx=-55, dy=-30, only={"wrist_left"})
        f["joint_gap"] = 18.0
    elif defect == "bad_proportions":
        style["head_radius"] = 112
        style["upper_arm"] = 68
        style["lower_arm"] = 130
        f["proportion_error"] = 0.42
        zoom = style["camera_zoom"]
        scale = style["actor_scale"]
        radius = style["head_radius"] * zoom * scale / 2
        head = points["head_center"]
        points["head_top"] = [head[0], round(head[1] - radius * 1.25, 2)]
        points["head_bottom"] = [head[0], round(head[1] + radius, 2)]
    elif defect == "offscreen_actor":
        points = shift_points(points, dx=-220)
        f["offscreen"] = 1
    elif defect == "cropped_head":
        points = shift_points(points, dy=-230, only={"head_center", "head_top"})
        f["head_crop"] = 1
    elif defect == "foot_lift":
        points = shift_points(points, dy=-45, only={"ankle_left", "foot_left"})
        f["foot_lift"] = 1

    r["keypoints"] = points
    r["style"] = style
    r["boxes"] = boxes_from(points, r["panel_visible"])
    r["features"] = f
    return r


def compact_row(row, split):
    features = [row["features"][name] for name in FEATURE_NAMES]
    return {
        "id": row["id"],
        "split": split,
        "defect": row["defect"],
        "intent": row["intent"],
        "caption": row["caption"],
        "gesture": row["gesture"],
        "previous_gesture": row["previous_gesture"],
        "emotion": row["emotion"],
        "camera": row["camera"],
        "visual": row["visual"],
        "panel_visible": row["panel_visible"],
        "phase": row["phase"],
        "scale": row["style"]["actor_scale"],
        "style": {
            "head_radius": row["style"]["head_radius"],
            "upper_arm": row["style"]["upper_arm"],
            "lower_arm": row["style"]["lower_arm"],
            "stroke_width": row["style"]["stroke_width"],
        },
        "points": [[round(v, 1) for v in row["keypoints"][key]] for key in KEYPOINT_ORDER],
        "features": features,
        "actor_box": [round(v, 1) for v in row["boxes"]["actor"]],
    }


def build():
    rng = random.Random(SEED)
    rows = []
    for index in range(sum(COUNTS.values())):
        base = professional_row(index, rng)
        rows.append(base if index % 2 == 0 else apply_defect(base, DEFECTS[index % len(DEFECTS)], rng))
    rng.shuffle(rows)
    cursor = 0
    for split, count in COUNTS.items():
        for row in rows[cursor:cursor + count]:
            row["split"] = split
        cursor += count
    assert cursor == len(rows)

    vocabulary = {
        "version": "stickman-drawing-v2",
        "gestures": list(GESTURES),
        "cameras": CAMERAS,
        "camera_zoom": CAMERA_ZOOM,
        "emotions": EMOTIONS,
        "intents": INTENTS,
        "visuals": VISUALS,
        "defects": DEFECTS,
        "captions": CAPTIONS,
        "style": STYLE,
        "safe_zones": SAFE_ZONES,
        "limits": LIMITS,
        "keypoint_order": KEYPOINT_ORDER,
        "feature_names": FEATURE_NAMES,
        "render_order": ["background", "panel", "actor", "headline", "subtitle"],
        "note": "Structured vector drawing plans for a deterministic renderer; not raster artwork or a diffusion dataset.",
        "license": "CC0-1.0",
    }
    dump(DATA / "vocabulary.json", vocabulary)
    dump(ASSETS / "stickman_drawing_vocabulary.json", vocabulary)

    counts = {}
    for split, count in COUNTS.items():
        part = [compact_row(row, split) for row in rows if row["split"] == split]
        write_jsonl(DATA / f"drawing_{split}.jsonl", part)
        professional = sum(r["defect"] is None for r in part)
        counts[split] = {
            "total": len(part), "professional": professional,
            "needs_fix": len(part) - professional,
            "defects": {d: sum(r["defect"] == d for r in part) for d in DEFECTS},
        }
    manifest = {
        "seed": SEED, "counts": counts, "total": len(rows),
        "keypoints": len(KEYPOINT_ORDER), "features": len(FEATURE_NAMES),
        "provenance": "assistant-authored synthetic vector plans; CC0-1.0; no expert review",
        "license": "CC0-1.0",
    }
    dump(DATA / "build_manifest.json", manifest)
    return {"total": len(rows), "counts": {k: v["total"] for k, v in counts.items()},
            "defects": len(DEFECTS), "keypoints": len(KEYPOINT_ORDER)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed files are reproducible")
    args = parser.parse_args()
    if args.check:
        before = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in DATA.rglob("*") if p.is_file()}
        before.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in ASSETS.glob("stickman_*.json")})
        counts = build()
        after = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in DATA.rglob("*") if p.is_file()}
        after.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in ASSETS.glob("stickman_*.json")})
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        if changed:
            raise SystemExit("Stickman drawing dataset drift: " + ", ".join(changed))
        print(json.dumps({"status": "reproducible", **counts}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
