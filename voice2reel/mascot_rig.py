"""Validated SVG puppet rig (trusted authored asset, never model-generated code)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "mascot_v1"


def load_rig(asset_dir: Path | str = ASSET_DIR) -> tuple[dict, dict, str]:
    root_dir = Path(asset_dir)
    rig = json.loads((root_dir / "rig.json").read_text(encoding="utf-8"))
    poses = json.loads((root_dir / "poses.json").read_text(encoding="utf-8"))
    svg = (root_dir / rig["asset"]).read_text(encoding="utf-8")
    root = ET.fromstring(svg)
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("Mascot SVG must use the SVG namespace")
    if rig.get("canvas") != {"width": 512, "height": 768}:
        raise ValueError("Unexpected mascot canvas")
    elements = {elem.attrib.get("id"): elem for elem in root.iter() if elem.attrib.get("id")}
    if not rig.get("parts"):
        raise ValueError("Mascot rig requires parts")
    for part, config in rig["parts"].items():
        if part not in elements or not elements[part].tag.endswith("}g"):
            raise ValueError(f"Mascot rig part is missing: {part}")
        if config["min_deg"] >= config["max_deg"]:
            raise ValueError(f"Invalid rotation range for {part}")
        if config["parent"] != "character":
            parent_elem = elements.get(config["parent"])
            if parent_elem is None or elements[part] not in list(parent_elem):
                raise ValueError(f"Incorrect nesting for {part}")
        px, py = config["pivot"]
        if not 0 <= px <= 512 or not 0 <= py <= 768:
            raise ValueError(f"Pivot outside canvas for {part}")
        expected = f"rotate(0 {px} {py})"
        if elements[part].get("transform") != expected:
            raise ValueError(f"SVG pivot mismatch for {part}")
    if poses.get("schema") != "voice2reel.mascot_poses.v1":
        raise ValueError("Unexpected pose schema")
    for gesture, angles in poses.items():
        if gesture == "schema":
            continue
        for part, angle in angles.items():
            if part not in rig["parts"]:
                raise ValueError(f"Unknown pose joint {part} in {gesture}")
            part_cfg = rig["parts"][part]
            if not part_cfg["min_deg"] <= angle <= part_cfg["max_deg"]:
                raise ValueError(f"Pose angle outside limits for {part} in {gesture}")
    return rig, poses, svg


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def animation_angles(rig: dict, poses: dict, gesture: str,
                     elapsed: float, duration: float) -> dict[str, float]:
    """Natural-looking arm timing with an authored, relaxed rest position.

    Each joint follows a bounded angle curve with a short lag for forearms and
    hands. Poses ease back to *idle*, not the previous stiff zero-angle T pose.
    Transform hierarchy and pivots never change.
    """
    if gesture not in poses or gesture == "schema":
        raise ValueError(f"Unsupported mascot gesture: {gesture}")
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if not math.isfinite(elapsed):
        raise ValueError("Elapsed time must be finite")

    neutral = poses["idle"]
    requested = poses[gesture]
    elapsed = min(duration, max(0.0, elapsed))
    angles = {}
    for part, config in rig["parts"].items():
        rest = float(neutral.get(part, 0))
        target = float(requested.get(part, rest))
        # Sequential joint articulation: shoulders move first, elbow follows.
        lag = (0.12 if part.startswith("hand_") else
               0.075 if part.endswith("_lower") else 0.0)
        lag = min(lag, duration * 0.12)
        active_duration = max(0.001, duration - lag)
        t = max(0.0, elapsed - lag)
        entry = ease(min(1.0, t / max(0.18, active_duration * 0.25)))
        leave = ease(min(1.0, max(0.0, (active_duration - t) /
                                  max(0.18, active_duration * 0.25))))
        weight = entry * leave
        angle = rest + (target - rest) * weight
        if gesture == "wave" and part in ("arm_right_lower", "hand_right"):
            angle += 7.0 * math.sin(2 * math.pi * elapsed * 1.7) * weight
        if gesture == "step" and part.startswith("leg_"):
            angle = rest + (target-rest) * weight * math.sin(
                2 * math.pi * elapsed * 1.2)
        if part == "head" and gesture != "idle":
            angle += 0.8 * math.sin(2 * math.pi * elapsed * 0.6) * weight
        angles[part] = min(config["max_deg"], max(config["min_deg"], angle))
    return angles
