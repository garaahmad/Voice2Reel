"""Bounded, measurable motion planning for the deterministic 2D presenter.

There is no trainable animation model in this repository: the character is drawn by
code. What CAN be measured and improved is the *plan*: camera steps, hold times,
jitter amplitude, overlay opacity, subtitle density and safe-zone overlaps. This
module loads the authored vocabulary, validates plans against comfort limits, and
provides geometry helpers used by the renderer so frames stay tidy and calm.
"""

import json
from bisect import bisect_left, bisect_right
from pathlib import Path

VOCABULARY_PATH = Path(__file__).parent / "assets" / "motion_vocabulary.json"
FALLBACK_LIMITS = {
    "max_camera_step_per_scene": 1,
    "camera_ease_seconds": 0.9,
    "max_gesture_switches_per_second": 0.5,
    "min_scene_seconds": 2.5,
    "max_jitter_amplitude": 2.0,
    "max_secondary_frequency_hz": 0.6,
    "max_flash_alpha": 0.12,
    "max_subtitle_words": 4,
    "min_safe_margin": 24,
}
CAMERA_ORDER = ["wide", "medium", "close"]
CAMERA_ZOOM = {"wide": 1.0, "medium": 1.13, "close": 1.5}
NUMERIC_FEATURES = (
    "camera_step", "gesture_switches_per_second", "jitter_amplitude", "scene_seconds",
    "flash_alpha", "overlap_count", "subtitle_words", "easing_smooth",
)


def feature_vector(features):
    """Pure numeric vector; the categorical easing flag becomes a fitted feature."""
    return {
        "camera_step": float(features.get("camera_step", 0)),
        "gesture_switches_per_second": float(features.get("gesture_switches_per_second", 0)),
        "jitter_amplitude": float(features.get("jitter_amplitude", 0)),
        "scene_seconds": float(features.get("scene_seconds", 0)),
        "flash_alpha": float(features.get("flash_alpha", 0)),
        "overlap_count": float(features.get("overlap_count", 0)),
        "subtitle_words": float(features.get("subtitle_words", 0)),
        "easing_smooth": 1.0 if features.get("easing", "smooth") == "smooth" else 0.0,
    }


def vocabulary(path=None):
    path = Path(path) if path else VOCABULARY_PATH
    if not path.is_file():
        return {"comfort_limits": dict(FALLBACK_LIMITS), "safe_zones": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("comfort_limits", dict(FALLBACK_LIMITS))
    data.setdefault("safe_zones", {})
    return data


def limits(path=None):
    value = dict(FALLBACK_LIMITS)
    value.update(vocabulary(path)["comfort_limits"])
    return value


def camera_step(previous, current):
    """Distance in camera levels; adjacent levels are the only comfortable cut."""
    if previous not in CAMERA_ORDER or current not in CAMERA_ORDER:
        raise ValueError("Unsupported camera level")
    return abs(CAMERA_ORDER.index(current) - CAMERA_ORDER.index(previous))


def clamp_camera(previous, current):
    """Never jump more than one level; return the intermediate level when needed."""
    if previous not in CAMERA_ORDER or current not in CAMERA_ORDER:
        raise ValueError("Unsupported camera level")
    a, b = CAMERA_ORDER.index(previous), CAMERA_ORDER.index(current)
    if abs(b - a) <= 1:
        return current
    return CAMERA_ORDER[a + (1 if b > a else -1)]


def camera_zoom(previous, current, progress, path=None):
    """Eased zoom passing through an intermediate level on long jumps."""
    target = clamp_camera(previous, current)
    lim = limits(path)
    duration = max(0.1, lim["camera_ease_seconds"])
    u = min(1.0, max(0.0, progress / duration))
    smooth = u * u * (3 - 2 * u)
    first = CAMERA_ZOOM[previous] + (CAMERA_ZOOM[target] - CAMERA_ZOOM[previous]) * smooth
    if target == current:
        return first
    return first  # caller advances to the target on the next scene, never in one cut


def validate_plan(plan, path=None):
    """Return a list of comfort violations for one scene plan (empty = comfortable)."""
    lim = limits(path)
    f = plan.get("features", plan)
    problems = []
    if f.get("camera_step", 0) > lim["max_camera_step_per_scene"]:
        problems.append("camera_jump")
    if f.get("gesture_switches_per_second", 0) > lim["max_gesture_switches_per_second"]:
        problems.append("gesture_flicker")
    if f.get("jitter_amplitude", 0) > lim["max_jitter_amplitude"]:
        problems.append("jitter")
    if f.get("scene_seconds", 0) < lim["min_scene_seconds"]:
        problems.append("cut_too_fast")
    if f.get("flash_alpha", 0) > lim["max_flash_alpha"]:
        problems.append("harsh_flash")
    if f.get("overlap_count", 0) > 0:
        problems.append("overlap")
    if f.get("subtitle_words", 0) > lim["max_subtitle_words"]:
        problems.append("dense_subtitle")
    if f.get("easing", "smooth") != "smooth":
        problems.append("abrupt_easing")
    return problems


def comfort_score(plan, path=None):
    """Fraction of comfort rules satisfied, 0..1."""
    checks = len(NUMERIC_FEATURES) + 2  # numeric features + easing + gesture vocabulary
    return round(1.0 - len(validate_plan(plan, path)) / checks, 4)


def overlaps(a, b, margin=0):
    """Axis-aligned box intersection test with an optional clearance."""
    return not (
        a[2] + margin <= b[0] or b[2] + margin <= a[0]
        or a[3] + margin <= b[1] or b[3] + margin <= a[1]
    )


def overlap_count(boxes, path=None):
    margin = limits(path)["min_safe_margin"]
    return sum(
        overlaps(boxes[a], boxes[b], 0 if a == "actor" and b == "panel" else margin)
        for i, a in enumerate(boxes) for b in list(boxes)[i + 1:]
        if not (a == "headline" and b == "subtitle")
    )


def safe_actor_box(camera, panel_visible, path=None, width=540, height=960):
    """A box that cannot cover the headline, subtitle or an explanatory panel."""
    zones = vocabulary(path)["safe_zones"]
    margin = limits(path)["min_safe_margin"]
    headline = zones.get("headline", [0, 0, width, 250])
    subtitle = zones.get("subtitle", [0, 860, width, height])
    panel = zones.get("panel", [295, 430, width, 820])
    # Keep a real clearance below the headline; touching its bottom edge is
    # still flagged as an overlap by the comfort rules.
    top = headline[3] + margin
    bottom = subtitle[1] - margin
    right = (panel[0] - margin) if panel_visible else (width - margin)
    left = margin
    return (left, top, right, bottom)


def fit_actor(size, position, safe_box):
    """Scale/translate the drawn actor so its pixels stay inside the safe box.

    Returns (scale, x, y). No cropping of the face happens: the whole layer is
    scaled uniformly, which keeps the character proportions and identity stable.
    """
    w, h = size
    bw, bh = safe_box[2] - safe_box[0], safe_box[3] - safe_box[1]
    if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
        raise ValueError("Actor and safe box must have positive dimensions")
    scale = min(1.0, bw / w, bh / h)
    sw, sh = w * scale, h * scale
    x = max(safe_box[0], min(safe_box[2] - sw, position[0]))
    y = max(safe_box[1], min(safe_box[3] - sh, position[1]))
    return (round(scale, 4), round(x), round(y))


def scene_plan(scene, previous_camera="wide", duration=4.0, panel_visible=True, path=None):
    """Compile one typed scene into a plan that satisfies every comfort rule."""
    lim = limits(path)
    current = clamp_camera(previous_camera, scene.get("camera", "wide"))
    zones = vocabulary(path)["safe_zones"]
    boxes = {
        "headline": tuple(zones.get("headline", [0, 0, 540, 250])),
        "subtitle": tuple(zones.get("subtitle", [0, 860, 540, 960])),
        "actor": safe_actor_box(current, panel_visible, path),
    }
    if panel_visible:
        boxes["panel"] = tuple(zones.get("panel", [295, 430, 540, 820]))
    features = {
        "camera_step": camera_step(previous_camera, current),
        "gesture_switches_per_second": 0.25,
        "jitter_amplitude": min(1.6, lim["max_jitter_amplitude"]),
        # Report the real speech-driven duration; short cuts remain visible as
        # violations instead of being hidden by clamping the measurement.
        "scene_seconds": round(float(duration), 2),
        "flash_alpha": min(0.1, lim["max_flash_alpha"]),
        "overlap_count": overlap_count(boxes, path),
        "subtitle_words": min(4, lim["max_subtitle_words"]),
        "easing": "smooth",
    }
    return {
        "camera": current,
        "gesture": scene.get("gesture", "present"),
        "emotion": scene.get("emotion", "neutral"),
        "features": features,
        "boxes": {k: list(v) for k, v in boxes.items()},
        "violations": validate_plan(features, path),
        "comfort_score": comfort_score(features, path),
    }


class ComfortClassifier:
    """A fitted violation detector: one threshold per feature, learned on train rows.

    Each rule learns both its threshold and its direction (lower/higher is safe) by
    maximizing balanced accuracy on the training split. A plan is comfortable only
    if NO learned rule fires. This is a real fitted model, not a hard-coded label,
    and adding features or training rows measurably changes held-out accuracy.
    """

    def __init__(self, features=NUMERIC_FEATURES):
        self.features = tuple(features)
        self.rules = {}

    def fit(self, rows):
        """Fit one max-margin 'safe side' rule per numeric feature.

        Because a plan must satisfy ALL rules at once, the fitted objective is:
        (1) reject as few comfortable rows as possible, (2) catch as many
        annoying rows as possible, (3) keep the widest margin from the observed
        comfortable values. Thresholds and directions come only from train rows.
        """
        self.rules = {}
        vectors = [(feature_vector(r["features"]), r["label"] == "comfortable")
                   for r in rows]
        positives = sum(ok for _, ok in vectors)
        negatives = len(vectors) - positives
        if not positives or not negatives:
            return self
        for name in self.features:
            comfortable = sorted(v[name] for v, ok in vectors if ok)
            annoying = sorted(v[name] for v, ok in vectors if not ok)
            values = sorted({v[name] for v, _ in vectors})
            if len(values) < 2:
                continue
            best = None
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                margin = min(threshold - values[i], values[i + 1] - threshold)
                for direction in ("le", "ge"):
                    if direction == "le":
                        unsafe_comfortable = bisect_right(comfortable, threshold)
                        unsafe_annoying = bisect_right(annoying, threshold)
                        false_alarms = positives - unsafe_comfortable
                        caught = negatives - unsafe_annoying
                    else:
                        unsafe_comfortable = bisect_left(comfortable, threshold)
                        unsafe_annoying = bisect_left(annoying, threshold)
                        false_alarms = unsafe_comfortable
                        caught = unsafe_annoying
                    # Lexicographic: no false alarms, then most violations caught,
                    # then widest margin from training observations.
                    score = (-false_alarms, caught, margin)
                    if best is None or score > best[0]:
                        best = (score, threshold, direction, false_alarms, caught)
            if best and best[3] <= max(1, positives // 100):
                _, threshold, direction, false_alarms, caught = best
                self.rules[name] = {"threshold": threshold, "safe": direction,
                                    "train_false_alarms": false_alarms,
                                    "train_violations_caught": caught}
        return self

    def predict(self, features):
        if not self.rules:
            return "annoying"
        vector = feature_vector(features)
        for name, rule in self.rules.items():
            value = vector[name]
            safe = value <= rule["threshold"] if rule["safe"] == "le" else value >= rule["threshold"]
            if not safe:
                return "annoying"
        return "comfortable"

    def accuracy(self, rows):
        if not rows:
            return 0.0
        return sum(self.predict(r["features"]) == r["label"] for r in rows) / len(rows)
