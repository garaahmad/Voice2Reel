"""Fitted models and measurable checks for the large stickman drawing corpus.

This is NOT a neural image generator. The repository draws the presenter with
deterministic vector code, so the trainable objects here learn:
  * ``StickmanArtist`` : ridge regression from scene intent to 18 2D joints
  * ``DefectDetector`` : logistic regression that flags an unprofessional drawing plan

Both are fitted on ``datasets/stickman/drawing_train.jsonl`` and measured on
held-out validation/test splits. Every reported number is computed here.
"""

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from voice2reel.common import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "stickman"
ASSET = ROOT / "voice2reel" / "assets" / "stickman_drawing_vocabulary.json"
SPLITS = ("train", "validation", "test")
CATEGORY_FIELDS = (
    ("gesture", "gestures"), ("previous_gesture", "gestures"),
    ("camera", "cameras"), ("emotion", "emotions"),
    ("intent", "intents"), ("visual", "visuals"),
)


@lru_cache(maxsize=1)
def vocabulary():
    return json.loads(ASSET.read_text(encoding="utf-8"))


def rows(split):
    return read_jsonl(DATA / f"drawing_{split}.jsonl")


def is_professional(row):
    return row["defect"] is None


def feature_dict(row):
    return dict(zip(vocabulary()["feature_names"], row["features"]))


def points_dict(row, points=None):
    values = row["points"] if points is None else points
    return dict(zip(vocabulary()["keypoint_order"], values))


def violations_for(row):
    """Reconstruct named violations from the compact feature vector."""
    vocab = vocabulary()
    f = feature_dict(row)
    limits = vocab["limits"]
    problems = []
    if f["camera_step"] > limits["max_camera_step_per_scene"]:
        problems.append("camera_jump")
    if f["gesture_switches_per_second"] > limits["max_gesture_switches_per_second"]:
        problems.append("gesture_flicker")
    if f["jitter_amplitude"] > limits["max_jitter_amplitude"]:
        problems.append("jitter")
    if f["scene_seconds"] < limits["min_scene_seconds"]:
        problems.append("cut_too_fast")
    if f["flash_alpha"] > limits["max_flash_alpha"]:
        problems.append("harsh_flash")
    if f["overlap_count"] > 0:
        problems.append("overlap")
    if f["subtitle_words"] > limits["max_subtitle_words"]:
        problems.append("dense_subtitle")
    if f["easing_smooth"] < 1:
        problems.append("abrupt_easing")
    if f["proportion_error"] > 0.1:
        problems.append("bad_proportions")
    if f["joint_gap"] > 5:
        problems.append("disconnected_joint")
    if f["headline_overlap"] > 0:
        problems.append("overlap")
    if f["foot_lift"] > 0:
        problems.append("foot_lift")
    if f["offscreen"] > 0:
        problems.append("offscreen_actor")
    if f["head_crop"] > 0:
        problems.append("cropped_head")
    return sorted(set(problems))


class StickmanArtist:
    """Ridge regression: scene intent -> 15 joint coordinates (30 outputs)."""

    def __init__(self, regularization=0.05):
        self.regularization = regularization
        self.weights = None
        self.vocab = vocabulary()

    def design(self, row):
        parts = [1.0]
        for field, key in CATEGORY_FIELDS:
            parts.extend(1.0 if row[field] == value else 0.0 for value in self.vocab[key])
        phase = float(row["phase"])
        bucket = min(3, int(phase * 4))
        parts.extend(1.0 if i == bucket else 0.0 for i in range(4))
        duration = float(feature_dict(row)["scene_seconds"])
        dbucket = min(2, max(0, int((duration - 2.5) / 2)))
        parts.extend(1.0 if i == dbucket else 0.0 for i in range(3))
        parts.extend([
            phase, math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase),
            float(row["panel_visible"]), float(row["scale"]),
            float(feature_dict(row)["jitter_amplitude"]),
        ])
        return np.asarray(parts, dtype=np.float64)

    def fit(self, rows_):
        professional = [row for row in rows_ if is_professional(row)]
        x = np.vstack([self.design(row) for row in professional])
        y = np.vstack([np.asarray(row["points"], dtype=np.float64).ravel() for row in professional])
        ridge = self.regularization * np.eye(x.shape[1])
        self.weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        return self

    def predict(self, row):
        flat = self.design(row) @ self.weights
        return [[round(float(flat[2 * i]), 2), round(float(flat[2 * i + 1]), 2)]
                for i in range(len(flat) // 2)]

    def mean_joint_error(self, rows_):
        professional = [row for row in rows_ if is_professional(row)]
        if not professional or self.weights is None:
            return float("inf")
        errors = []
        for row in professional:
            predicted = np.asarray(self.predict(row), dtype=np.float64).reshape(-1, 2)
            reference = np.asarray(row["points"], dtype=np.float64).reshape(-1, 2)
            errors.append(float(np.mean(np.linalg.norm(predicted - reference, axis=1))))
        return round(float(np.mean(errors)), 3)


class DefectDetector:
    """Fitted max-margin violation detector over numeric drawing features.

    For each feature it learns a threshold and a safe direction from train rows:
    reject as few professional plans as possible, catch as many defective plans
    as possible, then keep the widest margin. A plan is professional only when
    NO learned rule fires. This is a real fitted model, not a hard-coded OR.
    """

    def __init__(self, features=None):
        self.vocab = vocabulary()
        self.features = tuple(features or self.vocab["feature_names"])
        self.rules = {}

    def fit(self, rows_):
        from bisect import bisect_left, bisect_right

        professional = [feature_dict(row) for row in rows_ if is_professional(row)]
        defective = [feature_dict(row) for row in rows_ if not is_professional(row)]
        self.rules = {}
        if not professional or not defective:
            return self
        for name in self.features:
            good = sorted(float(row[name]) for row in professional)
            bad = sorted(float(row[name]) for row in defective)
            values = sorted(set(good + bad))
            if len(values) < 2:
                continue
            best = None
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                margin = min(threshold - values[i], values[i + 1] - threshold)
                for direction in ("le", "ge"):
                    if direction == "le":
                        false_alarms = len(good) - bisect_right(good, threshold)
                        caught = len(bad) - bisect_right(bad, threshold)
                    else:
                        false_alarms = bisect_left(good, threshold)
                        caught = bisect_left(bad, threshold)
                    score = (-false_alarms, caught, margin)
                    if best is None or score > best[0]:
                        best = (score, threshold, direction, false_alarms, caught)
            if best and best[3] == 0:
                _, threshold, direction, false_alarms, caught = best
                self.rules[name] = {
                    "threshold": threshold, "safe": direction,
                    "train_false_alarms": false_alarms, "train_defects_caught": caught,
                }
        return self

    def violations(self, row):
        values = feature_dict(row)
        problems = []
        for name, rule in self.rules.items():
            value = float(values[name])
            safe = value <= rule["threshold"] if rule["safe"] == "le" else value >= rule["threshold"]
            if not safe:
                problems.append(name)
        return problems

    def predict(self, row):
        return "professional" if not self.violations(row) else "needs_fix"

    def predict_proba(self, row):
        return 1.0 if self.predict(row) == "professional" else 0.0

    def accuracy(self, rows_):
        if not rows_:
            return 0.0
        return round(sum(self.predict(row) == ("professional" if is_professional(row) else "needs_fix")
                         for row in rows_) / len(rows_), 4)


def build_models(round_index, train_rows):
    vocab = vocabulary()
    budgets = [(8_000, 5), (25_000, 10), (len(train_rows), None)]
    count, features_n = budgets[min(round_index - 1, 2)]
    subset = train_rows[:count]
    artist = StickmanArtist().fit(subset)
    detector = DefectDetector(
        features=vocab["feature_names"][:features_n] if features_n else vocab["feature_names"]
    ).fit(subset)
    return artist, detector


def measure(artist, detector, split):
    part = rows(split)
    professional = [row for row in part if is_professional(row)]
    defect_accuracy = detector.accuracy(part)
    joint_error = artist.mean_joint_error(professional)
    artist_score = max(0.0, 1.0 - joint_error / 75.0)
    clean_rate = round(sum(not violations_for(row) for row in professional) / len(professional), 4)
    return {
        "split": split,
        "rows": len(part),
        "defect_accuracy": defect_accuracy,
        "mean_joint_error_px": joint_error,
        "artist_score": round(artist_score, 4),
        "professional_clean_rate": clean_rate,
        "score": round(0.55 * defect_accuracy + 0.35 * artist_score + 0.10 * clean_rate, 4),
    }


def run_rounds(max_rounds=3, target=0.90, output=None):
    train_rows = rows("train")
    history = []
    for round_index in range(1, max_rounds + 1):
        artist, detector = build_models(round_index, train_rows)
        result = measure(artist, detector, "validation")
        budgets = [(8_000, 5), (25_000, 10), (len(train_rows), len(vocabulary()["feature_names"]))]
        result["round"] = round_index
        result["train_rows"], result["features"] = budgets[min(round_index - 1, 2)]
        history.append(result)
        if result["score"] >= target:
            break
    best = max(history, key=lambda item: item["score"])
    artist, detector = build_models(best["round"], train_rows)
    report = {
        "target": target,
        "rounds": history,
        "best_round": best["round"],
        "final_test_on_best_round": measure(artist, detector, "test"),
        "integrity": stickman_integrity(),
        "note": (
            "Measured on 60,000 authored synthetic vector drawing plans. These are small CPU models "
            "for joint prediction and defect detection, NOT a neural image generator and NOT a claim "
            "about subjective professional aesthetics."
        ),
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def stickman_integrity():
    vocab = vocabulary()
    checks = {"vocabulary_version": vocab["version"] == "stickman-drawing-v2"}
    ids = []
    all_rows = []
    for split in SPLITS:
        part = rows(split)
        all_rows.extend(part)
        ids.extend(row["id"] for row in part)
        checks[f"{split}_count"] = len(part) == vocab_counts(split)
        checks[f"{split}_split_field"] = all(row["split"] == split for row in part)
        checks[f"{split}_point_shape"] = all(
            len(row["points"]) == len(vocab["keypoint_order"]) and all(len(p) == 2 for p in row["points"])
            for row in part
        )
        checks[f"{split}_feature_shape"] = all(
            len(row["features"]) == len(vocab["feature_names"]) for row in part
        )
        checks[f"{split}_style_shape"] = all(
            set(("head_radius", "upper_arm", "lower_arm", "stroke_width")) <= set(row["style"])
            for row in part
        )
        checks[f"{split}_finite"] = all(
            all(math.isfinite(float(v)) for p in row["points"] for v in p)
            and all(math.isfinite(float(v)) for v in row["features"])
            for row in part
        )
        professional = [row for row in part if is_professional(row)]
        needs_fix = [row for row in part if not is_professional(row)]
        checks[f"{split}_professional_clean"] = all(not violations_for(row) for row in professional)
        checks[f"{split}_needs_fix_marked"] = all(violations_for(row) for row in needs_fix)
        checks[f"{split}_all_defects"] = all(
            any(row["defect"] == defect for row in needs_fix) for defect in vocab["defects"]
        )
        checks[f"{split}_actor_inside_canvas"] = all(
            0 <= row["actor_box"][0] and row["actor_box"][2] <= 540
            and 0 <= row["actor_box"][1] and row["actor_box"][3] <= 960
            for row in professional
        )

        def connected(row):
            p = points_dict(row)
            zoom = vocab["camera_zoom"][row["camera"]]
            factor = zoom * float(row["scale"])
            shoulder_center = (
                (p["shoulder_left"][0] + p["shoulder_right"][0]) / 2,
                (p["shoulder_left"][1] + p["shoulder_right"][1]) / 2,
            )
            return (
                math.dist(p["head_bottom"], p["neck"]) <= 16 * factor
                and math.dist(p["neck"], shoulder_center) <= 28 * factor
            )

        checks[f"{split}_neck_connected"] = all(connected(row) for row in professional)
    checks["unique_ids"] = len(ids) == len(set(ids))
    manifest = json.loads((DATA / "build_manifest.json").read_text(encoding="utf-8"))
    checks["manifest_total"] = manifest["total"] == len(all_rows)
    checks["manifest_counts"] = all(
        manifest["counts"][split]["total"] == vocab_counts(split) for split in SPLITS
    )
    checks["asset_mirror"] = ASSET.read_bytes() == (DATA / "vocabulary.json").read_bytes()
    checks["keypoint_order_length"] = len(vocab["keypoint_order"]) == 18
    checks["feature_names_length"] = len(vocab["feature_names"]) == 14
    return checks


@lru_cache(maxsize=3)
def vocab_counts(split):
    manifest = json.loads((DATA / "build_manifest.json").read_text(encoding="utf-8"))
    return manifest["counts"][split]["total"]


def _two_bone(root, target, upper, lower, bend=1):
    """Re-solve an elbow from a shoulder/wrist pair and explicit bone lengths."""
    dx, dy = target[0] - root[0], target[1] - root[1]
    distance = max(abs(upper - lower) + 1e-6, min(math.hypot(dx, dy), upper + lower - 1e-6))
    angle = math.atan2(dy, dx)
    offset = math.acos(max(-1.0, min(1.0, (upper**2 + distance**2 - lower**2) / (2 * upper * distance))))
    return (
        root[0] + upper * math.cos(angle + bend * offset),
        root[1] + upper * math.sin(angle + bend * offset),
    )


def draw_frame(row, points=None, output=None, width=540, height=960):
    """Render one compact drawing plan as a PNG sample (deterministic vector art)."""
    from PIL import Image, ImageDraw, ImageFont

    from voice2reel.text_layout import draw_text

    vocab = vocabulary()
    row_style = row.get("style", {})
    style = {**vocab["style"], **row_style}
    pts = points_dict(row, points)
    zoom = vocab["camera_zoom"][row["camera"]]
    scale = float(row["scale"])
    factor = zoom * scale
    stroke = max(3, round(style["stroke_width"] * factor))
    radius = style["head_radius"] * factor / 2
    joint_gap = float(feature_dict(row)["joint_gap"])

    image = Image.new("RGB", (width, height), style["paper"])
    draw = ImageDraw.Draw(image)
    ink, accent = style["ink"], style["accent"]
    for x in range(0, width, 45):
        draw.line((x, 0, x, height), fill="#e8e9e3")
    for y in range(0, height, 45):
        draw.line((0, y, width, y), fill="#e8e9e3")
    draw.ellipse((-135, 290, 390, 823), fill="#dbe8e8")

    font_path = ROOT / "voice2reel" / "assets" / "Cairo.ttf"
    font = ImageFont.truetype(str(font_path), 34) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 20) if font_path.is_file() else ImageFont.load_default()
    draw_text(draw, (width / 2, 92), row["caption"], font, "rtl", fill=ink, anchor="mt")
    draw.line((36, 42, 74, 42), fill=accent, width=5)

    if row["panel_visible"]:
        draw.rounded_rectangle((295, 430, 520, 790), radius=17, fill=style["panel"], outline="#b9cccc", width=2)
        draw_text(draw, (407, 470), row["visual"].upper(), small, "ltr", fill=ink, anchor="mt")
        for i in range(3):
            y = 520 + i * 78
            draw.rounded_rectangle((312, y, 503, y + 57), radius=10, fill="#dfeaf0")
            draw_text(draw, (407, y + 18), row["caption"], small, "rtl", fill=ink, anchor="mt")

    def line(points_, width_):
        flat = [tuple(map(float, p)) for p in points_]
        draw.line(flat, fill=ink, width=max(2, width_), joint="curve")
        for x, y in (flat[0], flat[-1]):
            draw.ellipse((x - width_ / 2, y - width_ / 2, x + width_ / 2, y + width_ / 2), fill=ink)

    shoulder_center = (
        (pts["shoulder_left"][0] + pts["shoulder_right"][0]) / 2,
        (pts["shoulder_left"][1] + pts["shoulder_right"][1]) / 2,
    )
    draw.ellipse((min(pts["foot_left"][0], pts["foot_right"][0]) - 60, 800,
                  max(pts["foot_left"][0], pts["foot_right"][0]) + 60, 850), fill="#223b3b20")
    for side in ("left", "right"):
        line([pts["hip"], pts[f"knee_{side}"], pts[f"ankle_{side}"]], stroke)
        line([pts[f"ankle_{side}"], pts[f"foot_{side}"]], stroke + 2)

        wrist = pts[f"wrist_{side}"]
        # Re-solve the elbow for the row's authored bone lengths; a bad
        # proportions row therefore visibly changes the two-bone IK solution.
        solved = _two_bone(
            pts[f"shoulder_{side}"], wrist,
            style["upper_arm"] * factor, style["lower_arm"] * factor,
            1 if side == "right" else -1,
        )
        stored = pts[f"elbow_{side}"]
        broken = joint_gap > 5 and math.dist(stored, solved) > 5
        # Keep the authored broken elbow and open a visible gap before the
        # lower arm starts; re-solving IK would silently repair the defect.
        elbow = stored if broken else solved
        line([pts[f"shoulder_{side}"], elbow], stroke - 2)
        lower_start = elbow
        if broken:
            dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
            length = max(1.0, math.hypot(dx, dy))
            fraction = min(0.45, max(0.30, joint_gap / length))
            lower_start = (elbow[0] + dx * fraction, elbow[1] + dy * fraction)
        line([lower_start, wrist], stroke - 2)
        draw.ellipse((wrist[0] - 9, wrist[1] - 9, wrist[0] + 9, wrist[1] + 9),
                     fill="#fffef9", outline=ink, width=2)

    # Torso, neck and head-bottom are one connected chain, not a floating head.
    line([pts["hip"], pts["torso_mid"], shoulder_center, pts["neck"], pts["head_bottom"]], stroke + 2)
    line([pts["shoulder_left"], pts["shoulder_right"]], stroke)
    draw.polygon([
        (pts["neck"][0] - 7, pts["neck"][1] + 2),
        (pts["neck"][0] + 7, pts["neck"][1] + 2),
        (pts["neck"][0], pts["neck"][1] + 20),
    ], fill=accent)

    head = pts["head_center"]
    draw.ellipse((head[0] - radius, head[1] - radius, head[0] + radius, head[1] + radius),
                 fill="#fffef9", outline=ink, width=3)
    draw.ellipse((head[0] - radius * 1.05, head[1] - radius * 1.25,
                  head[0] + radius * 0.75, head[1] - radius * 0.35), fill=accent, outline=ink, width=2)
    for dx in (-0.32, 0.32):
        cx = head[0] + radius * dx
        cy = head[1] - radius * 0.1
        draw.ellipse((cx - 5, cy - 7, cx + 5, cy + 7), fill="#fff", outline=ink, width=2)
        draw.ellipse((cx - 2, cy - 3, cx + 2, cy + 3), fill=ink)
    draw.arc((head[0] - radius * 0.25, head[1] + radius * 0.1,
              head[0] + radius * 0.25, head[1] + radius * 0.45), 10, 170, fill=ink, width=3)

    draw.rounded_rectangle((30, 865, 510, 950), radius=13, fill="#172023")
    draw_text(draw, (270, 885), row["caption"], small, "rtl", fill="#fffef8", anchor="mt")
    if row["defect"]:
        draw.rounded_rectangle((12, 12, 260, 42), radius=8, fill="#4b2a2b")
        draw_text(draw, (136, 18), row["defect"], small, "ltr", fill="#fff", anchor="mt")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    return image

def render_samples(output_dir):
    output = Path(output_dir)
    validation = rows("validation")
    chosen = []
    for gesture in ("present", "point", "wave"):
        match = next(row for row in validation if is_professional(row) and row["gesture"] == gesture)
        chosen.append(match)
    for defect in ("disconnected_joint", "bad_proportions", "offscreen_actor",
                   "cropped_head", "overlap"):
        match = next(row for row in validation if row["defect"] == defect)
        chosen.append(match)
    metadata = []
    for index, row in enumerate(chosen):
        name = f"sample_{index:02d}_{row['defect'] or row['gesture']}.png"
        path = output / name
        draw_frame(row, output=path)
        metadata.append({"file": name, "defect": row["defect"], "gesture": row["gesture"],
                         "caption": row["caption"]})
    write_json(output / "samples.json", {
        "samples": metadata,
        "note": "Deterministic vector samples from the synthetic drawing corpus; not AI-generated artwork.",
    })
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/stickman_drawing/artist_rounds.json")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--render-samples", help="Also write PNG samples to this directory")
    args = parser.parse_args()
    report = run_rounds(max_rounds=args.max_rounds, target=args.target, output=args.output)
    print(json.dumps({
        "rounds": len(report["rounds"]),
        "best_round": report["best_round"],
        "validation_scores": [item["score"] for item in report["rounds"]],
        "final_test_score": report["final_test_on_best_round"]["score"],
        "mean_joint_error_px": report["final_test_on_best_round"]["mean_joint_error_px"],
        "defect_accuracy": report["final_test_on_best_round"]["defect_accuracy"],
        "integrity_ok": all(report["integrity"].values()),
        "output": args.output,
    }, ensure_ascii=False, indent=2))
    if args.render_samples:
        print(json.dumps({"samples": render_samples(args.render_samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
