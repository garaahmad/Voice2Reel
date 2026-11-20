"""Strict text-to-SVG validation and optional PNG conversion.

Never execute Qwen output. Only a small SVG subset is allowed.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

TAGS = {"svg", "g", "path", "circle", "ellipse", "rect", "line"}
ATTRS = {"xmlns", "width", "height", "viewBox", "cx", "cy", "r", "rx", "ry",
         "x", "y", "x1", "y1", "x2", "y2", "d", "fill", "stroke",
         "stroke-width", "stroke-linecap", "stroke-linejoin", "opacity"}
COLORS = re.compile(r"^(#[0-9a-fA-F]{3,8}|none|white|black)$")
NUM = re.compile(r"^[0-9.]+$")
PATH = re.compile(r"^[MmLlHhVvCcSsQqTtZz0-9.,+\- eE]+$")
def validate_svg(svg):
    if not isinstance(svg, str) or len(svg) > 24000:
        raise ValueError("SVG exceeds length limit")
    if re.search(r"<!|&|<\s*(?:script|image|foreignObject|use)\b", svg, re.I):
        raise ValueError("External resources, entities and scripts are forbidden")
    root = ET.fromstring(svg)
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("Expected SVG root with SVG namespace")
    if root.attrib.get("viewBox") != "0 0 512 768":
        raise ValueError("Expected viewBox 0 0 512 768")
    if root.attrib.get("width") != "512" or root.attrib.get("height") != "768":
        raise ValueError("Expected 512x768 SVG")
    elements = list(root.iter())
    if len(elements) > 100:
        raise ValueError("Too many SVG elements")
    for elem in elements:
        tag = elem.tag.split("}")[-1]
        if tag not in TAGS:
            raise ValueError("Unsupported SVG element: "+tag)
        if elem.text and elem.text.strip():
            raise ValueError("Text nodes are forbidden")
        for attr, value in elem.attrib.items():
            if attr not in ATTRS or len(value) > 2000:
                raise ValueError("Unsupported SVG attribute: "+attr)
            if attr in ("fill", "stroke") and not COLORS.fullmatch(value):
                raise ValueError("Unsupported SVG color")
            if attr == "d" and not PATH.fullmatch(value):
                raise ValueError("Unsupported SVG path")
            if attr in ("cx","cy","r","rx","ry","x","y","x1","y1","x2","y2","stroke-width","opacity") and not NUM.fullmatch(value):
                raise ValueError("Unsupported SVG number")
            if attr in ("stroke-linecap","stroke-linejoin") and value not in ("round","butt","square","miter","bevel"):
                raise ValueError("Unsupported stroke style")
    return svg

def to_png(svg, path):
    validate_svg(svg)
    import cairosvg
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"),write_to=str(path),
                    output_width=512,output_height=768)
    return path
