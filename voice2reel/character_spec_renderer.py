"""Apply bounded character styling to trusted mascot_v1 SVG artwork.

Generated specs NEVER control free-form XML, geometry, scripts or rig pivots.
The authored joint hierarchy is reused without modification.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from voice2reel.character_spec import CharacterSpec, PALETTES, parse_spec

SVG_NS = "http://www.w3.org/2000/svg"
Q = "{" + SVG_NS + "}"
ET.register_namespace("", SVG_NS)

COLOR_SLOTS = {
    "#20232B": "ink",
    "#EF7559": "accent",
    "#FFFCF4": "face",
    "#41454D": "detail",
    "#4A4E57": "detail",
    "#252B33": "shade",
}
DEFAULT_MOUTH = "M234 242 Q256 267 278 242"
MOUTHS = {
    "happy": (DEFAULT_MOUTH, "none"),
    "surprised": ("M256 234 C240 234 240 269 256 269 C272 269 272 234 256 234 Z", "ink"),
    "curious": ("M235 249 Q257 240 277 251", "none"),
    "thoughtful": ("M239 255 Q257 249 275 253", "none"),
    "confident": ("M232 244 Q256 271 282 241", "none"),
}
EYEBROWS = {
    "soft": ("M203 169 Q216 157 233 168", "M278 168 Q296 157 309 169"),
    "raised": ("M201 160 Q217 146 234 158", "M277 157 Q297 146 311 160"),
    "focused": ("M201 161 Q217 168 233 174", "M278 174 Q295 168 311 161"),
}


def apply_spec(svg: str, spec: CharacterSpec | dict) -> str:
    style = parse_spec(spec)
    palette = PALETTES[style.palette]
    root = ET.fromstring(svg)

    # Recolor only a closed, trusted set of palette slots.
    for element in root.iter():
        for attribute in ("fill", "stroke"):
            old = element.get(attribute)
            if old in COLOR_SLOTS:
                slot = COLOR_SLOTS[old]
                element.set(attribute, palette[slot])

    elements = {e.get("id"): e for e in root.iter() if e.get("id")}
    if any(part not in elements for part in ("character", "head", "face", "torso", "mouth_happy")):
        raise ValueError("Trusted mascot artwork has missing part IDs")

    face = elements["face"]
    brows = [e for e in face.findall(Q + "path") if e.get("id") is None]
    if len(brows) != 2:
        raise ValueError("Expected exactly two authored brows")
    for brow, path in zip(brows, EYEBROWS[style.brows]):
        brow.set("d", path)
    mouth = elements["mouth_happy"]
    mouth_path, mouth_fill = MOUTHS[style.expression]
    mouth.set("d", mouth_path)
    mouth.set("fill", palette["ink"] if mouth_fill == "ink" else "none")

    # Controlled eye expression: no coordinates outside the original face.
    eyes = [e for e in face.findall(Q + "ellipse")
            if e.get("cx") in {"220", "291"} and e.get("fill") == "#FFFFFF"]
    # White sclera stays white regardless of palette.
    if len(eyes) != 2:
        raise ValueError("Expected two authored eyes")
    dims = {"friendly": ("18", "23"), "wide": ("20", "26"), "relaxed": ("18", "19")}[style.eyes]
    for e in eyes:
        e.set("rx", dims[0])
        e.set("ry", dims[1])
    if style.expression == "surprised":
        for e in eyes:
            e.set("rx", "22")
            e.set("ry", "28")

    for e in face.findall(Q + "ellipse"):
        if e.get("cx") in {"193", "320"}:
            e.set("opacity", "0" if style.cheeks == "none" else "0.23")

    # Accent detail is a designed flourish only, never a body joint or bone.
    for e in elements["torso"].findall(Q + "path"):
        if e.get("d") == "M243 309 Q256 320 269 309":
            if style.accent == "none":
                e.set("opacity", "0")
            elif style.accent == "simple":
                e.set("stroke", palette["detail"])
            else:
                e.set("stroke", palette["accent"])
            break

    result = ET.tostring(root, encoding="unicode")
    return result
