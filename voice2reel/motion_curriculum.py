"""Compositional motion curriculum with genuinely held-out gesture families.

Earlier datasets in this repository split *rows* from the same template family,
so validation only measured interpolation. This curriculum splits whole motion
families: the learner sees six families during training and is measured on two
families it has never seen, including one with out-of-range knob values.

Trajectories are linear combinations of a small bank of shared motion primitives
(vertical pulse, arm swing, lift, sway, head nod, stride, tempo, balance). A
model that learns that shared basis can compose unseen gestures; a model that
memorises family labels cannot. Everything here is synthetic, deterministic and
measured on CPU. It is a curriculum for smooth vector motion, NOT a claim about
real human motion capture or subjective animation quality.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from voice2reel.character_templates import JOINT_ORDER, pose_points

CANVAS = np.array([540.0, 960.0], dtype=np.float64)
FRAMES = 24
KNOBS = ("pulse", "swing", "lift", "sway", "nod", "stride", "tempo", "balance")
HARMONICS = (1, 2, 3)
SIGNALS = ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (1, 1), (2, 2))

TRAIN_FAMILIES = ("wave", "point", "present", "explain", "step", "breathe")
VALIDATION_FAMILY = "nod"
TEST_FAMILY = "celebrate"
FAMILIES = TRAIN_FAMILIES + (VALIDATION_FAMILY, TEST_FAMILY)

# Eight semantic knobs per family. Held-out families combine knobs in ways the
# six training families never do (e.g. high nod+balance, high stride+lift).
FAMILY_RECIPES = {
    "wave":     (0.12, 0.92, 0.62, 0.10, 0.08, 0.04, 0.28, 0.10),
    "point":    (0.10, 0.30, 0.86, 0.18, 0.12, 0.06, 0.16, 0.42),
    "present":  (0.20, 0.55, 0.38, 0.30, 0.10, 0.08, 0.20, 0.55),
    "explain":  (0.26, 0.68, 0.34, 0.22, 0.36, 0.05, 0.55, 0.18),
    "step":     (0.18, 0.22, 0.16, 0.24, 0.10, 0.90, 0.34, 0.62),
    "breathe":  (0.88, 0.10, 0.08, 0.16, 0.24, 0.03, 0.10, 0.20),
    "nod":      (0.32, 0.14, 0.12, 0.12, 0.94, 0.04, 0.72, 0.66),
    "celebrate": (0.46, 0.88, 0.92, 0.36, 0.44, 0.78, 0.80, 0.28),
}

# Unit offsets in fractions of the canvas, one (18, 2) map per primitive.
def _basis():
    maps = {name: np.zeros((len(JOINT_ORDER), 2)) for name in KNOBS}
    index = {joint: i for i, joint in enumerate(JOINT_ORDER)}

    def put(name, joint, dx, dy):
        maps[name][index[joint]] = (dx, dy)

    upper = ("head_center", "head_top", "head_bottom", "neck",
             "shoulder_left", "shoulder_right", "torso_mid",
             "elbow_left", "elbow_right", "wrist_left", "wrist_right")
    for joint in upper:
        put("pulse", joint, 0.0, -0.010)
    for joint, dx in (("wrist_right", 0.042), ("elbow_right", 0.024),
                      ("wrist_left", -0.042), ("elbow_left", -0.024)):
        put("swing", joint, dx, 0.0)
    for joint, dy in (("wrist_right", -0.050), ("elbow_right", -0.022),
                      ("wrist_left", -0.050), ("elbow_left", -0.022)):
        put("lift", joint, 0.0, dy)
    for joint in JOINT_ORDER:
        put("sway", joint, 0.016, 0.0)
    for joint, dy in (("head_center", -0.014), ("head_top", -0.014),
                      ("head_bottom", -0.014), ("neck", -0.006)):
        put("nod", joint, 0.0, dy)
    for joint, dx, dy in (("knee_left", -0.026, -0.014), ("ankle_left", -0.040, -0.010),
                          ("foot_left", -0.040, -0.010), ("knee_right", 0.026, 0.008),
                          ("ankle_right", 0.040, 0.008), ("foot_right", 0.040, 0.008)):
        put("stride", joint, dx, dy)
    for joint, dx in (("wrist_right", 0.012), ("wrist_left", -0.012),
                      ("elbow_right", 0.006), ("elbow_left", -0.006)):
        put("tempo", joint, dx, 0.0)
    for joint, dx in (("hip", 0.010), ("torso_mid", -0.008),
                      ("shoulder_left", -0.010), ("shoulder_right", 0.010)):
        put("balance", joint, dx, 0.0)
    return np.stack([maps[name] for name in KNOBS])


BASIS = _basis()
TIME_BASIS_SIZE = 1 + 2 * len(HARMONICS)
WIDTH = 1 + TIME_BASIS_SIZE + len(KNOBS) * TIME_BASIS_SIZE


def time_basis(phase):
    values = [1.0]
    for harmonic in HARMONICS:
        values.append(math.sin(2 * math.pi * harmonic * phase))
        values.append(math.cos(2 * math.pi * harmonic * phase))
    return np.asarray(values, dtype=np.float64)


def design(knobs, phase):
    """Feature map: phase harmonics plus every knob crossed with every harmonic."""
    basis = time_basis(phase)
    knobs = np.asarray(knobs, dtype=np.float64)
    return np.concatenate(([1.0], basis, np.outer(knobs, basis).ravel()))


def _signal(index, phase):
    first, second = SIGNALS[index]
    return (math.sin(2 * math.pi * first * phase)
            + 0.35 * math.cos(2 * math.pi * second * phase))


def base_points(canvas=CANVAS):
    width, height = float(canvas[0]), float(canvas[1])
    pts = pose_points("idle", 0.0, width, height,
                      actor_box=(width * 0.12, height * 0.16,
                                 width * 0.88, height * 0.86))
    return np.stack([pts[joint] for joint in JOINT_ORDER])


def _noise(plan, frame_index):
    seed = int(hashlib.sha256(f"{plan['id']}:{frame_index}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(0.0, 0.35, (len(JOINT_ORDER), 2))


def trajectory(plan, frames=FRAMES, canvas=CANVAS, noise=True):
    """(frames, 18, 2) absolute joint positions from the shared primitive bank."""
    knobs = np.asarray(plan["knobs"], dtype=np.float64)
    base = base_points(canvas)
    result = np.empty((frames, len(JOINT_ORDER), 2), dtype=np.float64)
    for frame in range(frames):
        phase = frame / frames
        offsets = np.zeros((len(JOINT_ORDER), 2))
        for index in range(len(KNOBS)):
            offsets += knobs[index] * _signal(index, phase) * BASIS[index]
        points = base + offsets * canvas
        if noise:
            points = points + _noise(plan, frame)
        result[frame] = points
    return result


def _family_plan(family, index, split, regime="in_distribution"):
    recipe = np.asarray(FAMILY_RECIPES[family], dtype=np.float64)
    seed = int(hashlib.sha256(f"{split}:{family}:{index}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.16, 0.16, len(KNOBS))
    knobs = np.clip(recipe + jitter, 0.0, 1.4)
    if regime == "extrapolated":
        knobs = np.clip(recipe * 1.45 + jitter, 0.0, 1.4)
    return {
        "id": f"{split}-{family}-{index:04d}",
        "family": family, "split": split, "regime": regime,
        "knobs": [round(float(value), 5) for value in knobs],
    }


def build_curriculum(train_per_family=120, held_out_per_family=80, seed=20261116):
    """Disjoint family-level splits. No family appears in two splits."""
    del seed  # IDs are content-hashed; reproducibility does not depend on global RNG.
    train = [_family_plan(family, i, "train") for family in TRAIN_FAMILIES
             for i in range(train_per_family)]
    validation = [_family_plan(VALIDATION_FAMILY, i, "validation")
                  for i in range(held_out_per_family)]
    test = [_family_plan(TEST_FAMILY, i, "test") for i in range(held_out_per_family)]
    extrapolation = [_family_plan(TEST_FAMILY, i, "extrapolation", "extrapolated")
                     for i in range(held_out_per_family)]
    splits = {"train": train, "validation": validation, "test": test,
              "extrapolation": extrapolation}
    train_families = {plan["family"] for plan in train}
    for name in ("validation", "test", "extrapolation"):
        assert not ({plan["family"] for plan in splits[name]} & train_families)
    ids = [plan["id"] for plans in splits.values() for plan in plans]
    assert len(ids) == len(set(ids))
    return splits


def _matrices(plans, frames=FRAMES):
    x = np.vstack([design(plan["knobs"], frame / frames)
                   for plan in plans for frame in range(frames)])
    y = np.vstack([(trajectory(plan, frames) - base_points()).reshape(frames, -1)
                   for plan in plans])
    return x, y


class MotionLearner:
    """Linear compositional learner trained with Adam; measured, not claimed."""

    def __init__(self, epochs=60, batch_size=512, learning_rate=0.09, seed=7):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.weights = None
        self.history = []

    def _offsets(self, knobs, phase):
        return (design(knobs, phase) @ self.weights).reshape(-1, 2)

    def predict(self, plan, frames=FRAMES):
        base = base_points()
        return np.stack([base + self._offsets(plan["knobs"], frame / frames)
                         for frame in range(frames)])

    def fit(self, plans, validation_plans=None):
        x, y = _matrices(plans)
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, (x.shape[1], y.shape[1]))
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        step = 0
        best_score, best_weights = float("inf"), w.copy()
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(x))
            for start in range(0, len(x), self.batch_size):
                batch = order[start:start + self.batch_size]
                xb, yb = x[batch], y[batch]
                residual = (xb @ w - yb) / len(batch)
                gradient = xb.T @ residual
                step += 1
                m = 0.9 * m + 0.1 * gradient
                v = 0.999 * v + 0.001 * gradient ** 2
                mhat = m / (1 - 0.9 ** step)
                vhat = v / (1 - 0.999 ** step)
                w -= self.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            self.weights = w
            loss = float(np.mean((x @ w - y) ** 2))
            record = {"epoch": epoch, "train_mse": round(loss, 4)}
            if validation_plans and (epoch % 10 == 0 or epoch == self.epochs):
                record["validation_mae_px"] = self.mean_error(validation_plans)
                if record["validation_mae_px"] < best_score:
                    best_score, best_weights = record["validation_mae_px"], w.copy()
            self.history.append(record)
        self.weights = best_weights if validation_plans else w
        return self

    def mean_error(self, plans, frames=FRAMES):
        errors = [np.abs(self.predict(plan, frames) - trajectory(plan, frames)).mean()
                  for plan in plans]
        return round(float(np.mean(errors)), 3)

    def max_error(self, plans, frames=FRAMES):
        return round(float(max(np.abs(self.predict(plan, frames)
                                     - trajectory(plan, frames)).max() for plan in plans)), 3)


class FamilyMemoryBaseline:
    """Honest comparison: copy the closest training plan's trajectory by knobs."""

    def __init__(self, train_plans):
        self.train_plans = train_plans
        self.vectors = np.stack([np.asarray(plan["knobs"]) for plan in train_plans])
        self.tracks = np.stack([trajectory(plan) for plan in train_plans])

    def predict(self, plan):
        distance = ((self.vectors - np.asarray(plan["knobs"])) ** 2).sum(axis=1)
        return self.tracks[int(np.argmin(distance))]

    def mean_error(self, plans):
        return round(float(np.mean([
            np.abs(self.predict(plan) - trajectory(plan)).mean() for plan in plans
        ])), 3)


def constant_baseline_error(train_plans, plans):
    mean_track = np.mean(np.stack([trajectory(plan) for plan in train_plans]), axis=0)
    return round(float(np.mean([
        np.abs(mean_track - trajectory(plan)).mean() for plan in plans
    ])), 3)


def motion_quality(points, fps=24):
    """Frame-level comfort measurements used by the production gate."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=2)
    jerk = np.abs(np.diff(points, n=3, axis=0)).mean()
    margin = 24.0
    offscreen = float(np.sum((points[..., 0] < margin) | (points[..., 0] > CANVAS[0] - margin)
                             | (points[..., 1] < margin) | (points[..., 1] > CANVAS[1] - margin)))
    return {
        "max_joint_step_px": round(float(steps.max()), 3),
        "mean_joint_step_px": round(float(steps.mean()), 3),
        "mean_jerk_px": round(float(jerk), 4),
        "safe_zone_violations": int(offscreen),
        "frames": int(len(points)),
        "fps": fps,
    }


DEFECTS = ("jitter", "joint_gap", "offscreen", "flash", "flicker", "abrupt")
TRAIN_DEFECTS = DEFECTS[:4]
HELD_OUT_DEFECTS = DEFECTS[4:]
GATE_FEATURES = ("max_step_px", "jerk_px", "joint_gap_px", "offscreen_px",
                 "flash_delta", "switches_per_second", "subtitle_words", "easing_smooth")


def _gate_row(defect, index, split):
    seed = int(hashlib.sha256(f"{split}:{defect}:{index}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    values = {
        "max_step_px": rng.uniform(2.0, 5.0),
        "jerk_px": rng.uniform(0.2, 1.2),
        "joint_gap_px": rng.uniform(0.0, 1.5),
        "offscreen_px": rng.uniform(0.0, 0.5),
        "flash_delta": rng.uniform(0.0, 0.05),
        "switches_per_second": rng.uniform(0.0, 0.25),
        "subtitle_words": rng.integers(1, 4),
        "easing_smooth": 1.0,
    }
    if defect is not None:
        boost = {
            "jitter": ("max_step_px", 22.0), "joint_gap": ("joint_gap_px", 12.0),
            "offscreen": ("offscreen_px", 40.0), "flash": ("flash_delta", 0.35),
            "flicker": ("switches_per_second", 1.4), "abrupt": ("easing_smooth", -1.0),
        }[defect]
        values[boost[0]] += boost[1]
        if defect == "jitter":
            values["jerk_px"] += 9.0
    return {"id": f"{split}-{defect or 'clean'}-{index:04d}", "defect": defect,
            "split": split, "features": {key: round(float(value), 4)
                                         for key, value in values.items()}}


class MotionQualityGate:
    """Thresholds fitted from clean train rows; tested on held-out defect kinds."""

    def __init__(self):
        self.rules = {}

    def fit(self, rows_):
        clean = [row["features"] for row in rows_ if row["defect"] is None]
        bad = [row["features"] for row in rows_ if row["defect"] is not None]
        for name in GATE_FEATURES:
            values = sorted(float(row[name]) for row in clean)
            ceiling = max(values) + 0.001
            floor = min(values) - 0.001
            violations_high = sum(float(row[name]) > ceiling for row in bad)
            violations_low = sum(float(row[name]) < floor for row in bad)
            if violations_high or violations_low:
                self.rules[name] = {"min": round(floor, 4), "max": round(ceiling, 4)}
        return self

    def violations(self, row):
        problems = []
        for name, rule in self.rules.items():
            value = float(row["features"][name])
            if value < rule["min"] or value > rule["max"]:
                problems.append(name)
        return problems

    def accuracy(self, rows_):
        correct = sum(
            (not self.violations(row)) == (row["defect"] is None) for row in rows_
        )
        return round(correct / len(rows_), 4)


def build_gate_corpus(per_class=120):
    train = [_gate_row(None, i, "train") for i in range(per_class * 2)]
    train += [_gate_row(defect, i, "train") for defect in TRAIN_DEFECTS
              for i in range(per_class)]
    validation = [_gate_row(None, i, "validation") for i in range(per_class)]
    validation += [_gate_row(defect, i, "validation") for defect in TRAIN_DEFECTS
                   for i in range(per_class // 2)]
    test = [_gate_row(None, i, "test") for i in range(per_class)]
    test += [_gate_row(defect, i, "test") for defect in HELD_OUT_DEFECTS
             for i in range(per_class)]
    return {"train": train, "validation": validation, "test": test}


def run_benchmark(train_per_family=120, held_out_per_family=80, epochs=60):
    """Real CPU measurements for the report. Returns a JSON-safe dictionary."""
    splits = build_curriculum(train_per_family, held_out_per_family)
    learner = MotionLearner(epochs=epochs).fit(splits["train"], splits["validation"])
    memory = FamilyMemoryBaseline(splits["train"])
    gate_corpus = build_gate_corpus()
    gate = MotionQualityGate().fit(gate_corpus["train"])
    result = {
        "split_counts": {name: len(plans) for name, plans in splits.items()},
        "train_families": list(TRAIN_FAMILIES),
        "validation_family": VALIDATION_FAMILY,
        "test_family": TEST_FAMILY,
        "held_out_family_overlap": [],
        "epochs": epochs,
        "loss_curve": learner.history,
        "final_train_mse": learner.history[-1]["train_mse"],
        "mae_px": {
            name: {
                "compositional_learner": learner.mean_error(plans),
                "nearest_train_plan": memory.mean_error(plans),
                "constant_mean_pose": constant_baseline_error(splits["train"], plans),
            }
            for name, plans in splits.items()
        },
        "max_error_px_test": learner.max_error(splits["test"]),
        "generalization_gain_vs_nearest": round(
            memory.mean_error(splits["test"]) / max(1e-6, learner.mean_error(splits["test"])), 2
        ),
        "motion_quality_validation_sample": motion_quality(
            learner.predict(splits["validation"][0])
        ),
        "defect_gate": {
            "train_defects": list(TRAIN_DEFECTS),
            "held_out_defects": list(HELD_OUT_DEFECTS),
            "fitted_rules": len(gate.rules),
            "accuracy": {name: gate.accuracy(rows_)
                         for name, rows_ in gate_corpus.items()},
        },
        "notes": "Synthetic primitive curriculum; held-out families, not held-out rows.",
    }
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-per-family", type=int, default=120)
    parser.add_argument("--held-out-per-family", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.train_per_family, args.held_out_per_family,
                                   args.epochs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
