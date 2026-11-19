"""Animate the jointed mascot SVG using nested vector transforms.

No strokes or disconnected joint positions are regenerated per frame;
existing authored vector artwork rotates around declared pivots.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image
from voice2reel.mascot_rig import ASSET_DIR, animation_angles, load_rig


class MascotAnimator:
    def __init__(self, asset_dir: Path | str = ASSET_DIR, spec=None):
        self.rig, self.poses, self.base_svg = load_rig(asset_dir)
        if spec is not None:
            from voice2reel.character_spec_renderer import apply_spec
            self.base_svg = apply_spec(self.base_svg, spec)
        self._tokens = {}
        self.template = self.base_svg
        for part, config in self.rig["parts"].items():
            x, y = config["pivot"]
            source = f'id="{part}" transform="rotate(0 {x} {y})"'
            placeholder = f'id="{part}" transform="rotate(@@{part}@@ {x} {y})"'
            if self.template.count(source) != 1:
                raise ValueError(f"Expected exactly one transform for {part}")
            self.template = self.template.replace(source, placeholder)
            self._tokens[part] = f"@@{part}@@"

    def frame_svg(self, gesture="idle", elapsed=0.0, duration=3.0):
        angles = animation_angles(self.rig, self.poses, gesture, elapsed, duration)
        svg = self.template
        for part, angle in angles.items():
            svg = svg.replace(self._tokens[part], f"{angle:.3f}")
        bob = 2 * math.sin(2 * math.pi * elapsed * 0.7)
        svg = svg.replace('id="character" transform="translate(0 0)"',
                          f'id="character" transform="translate(0 {bob:.2f})"')
        return svg

    def frame_image(self, gesture="idle", elapsed=0.0, duration=3.0,
                    width=640):
        import cairosvg
        svg = self.frame_svg(gesture, elapsed, duration)
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                              output_width=width,
                              output_height=round(width * 768 / 512))
        return Image.open(io.BytesIO(png)).convert("RGBA")


def render_pose_sheet(output: str | Path, gestures=("idle", "present", "point", "wave"), spec=None):
    """Render a low-cost visual acceptance sheet before video production."""
    from PIL import ImageDraw
    animator = MascotAnimator(spec=spec)
    cell = (300, 510)
    sheet = Image.new("RGB", (cell[0] * len(gestures), cell[1]), "#f6f6f5")
    draw = ImageDraw.Draw(sheet)
    for i, pose in enumerate(gestures):
        frame = animator.frame_image(pose, 1.25, 3, width=300)
        sheet.paste(frame, (i * cell[0], -5), frame)
        draw.text((i*cell[0]+100, cell[1]-38), pose, fill="#141923")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output
