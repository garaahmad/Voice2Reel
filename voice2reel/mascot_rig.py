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
    """Smoothly enter/exit a gesture; keep angles within rig constraints."""
    if gesture not in poses or gesture == "schema":
        raise ValueError(f"Unsupported mascot gesture: {gesture}")
    if duration <= 0:
        raise ValueError("Duration must be positive")
    blend = ease(min(1.0, elapsed / max(.2, duration * .22))) * ease(
        min(1.0, max(0.0, (duration - elapsed) / max(.2, duration * .22))))
    angles = {}
    for part, config in rig["parts"].items():
        target = float(poses[gesture].get(part, 0))
        angle = target * blend
        if part == "head":
            angle += 1.8 * math.sin(2 * math.pi * elapsed * .55) * blend
        if gesture == "wave" and part in ("arm_right_lower", "hand_right"):
            angle += 9.0 * math.sin(2 * math.pi * elapsed * 2.0) * blend
        if gesture == "step" and part.startswith("leg_"):
            angle *= math.sin(2 * math.pi * elapsed * 1.25)
        angles[part] = min(config["max_deg"], max(config["min_deg"], angle))
    return angles
