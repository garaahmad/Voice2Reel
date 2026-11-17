"""Composite a jointed, authored SVG character into a production PIL video frame."""
from __future__ import annotations
from PIL import Image
from voice2reel.mascot_animator import MascotAnimator


def draw_rig_character(image: Image.Image, animator: MascotAnimator, actor_box,
                       gesture: str, elapsed: float, duration: float):
    x0, y0, x1, y1 = actor_box
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid actor box")
    width = max(128, round(min(x1-x0, (y1-y0)*2/3)))
    frame = animator.frame_image(gesture, elapsed, duration, width=width)
    x = round((x0+x1-frame.width)/2)
    y = round((y0+y1-frame.height)/2)
    image.paste(frame, (x, y), frame)
    return (x, y, frame.width, frame.height)
