"""Regression tests for the original SVG mascot and nested joint animation."""
from xml.etree import ElementTree as ET

from PIL import Image
from voice2reel.mascot_animator import MascotAnimator
from voice2reel.mascot_rig import animation_angles, load_rig
from voice2reel.mascot_renderer import draw_rig_character


def test_rig_integrity():
    rig, poses, svg = load_rig()
    assert len(rig["parts"]) == 14
    assert 'id="hat"' not in svg and 'id="beret"' not in svg
    assert all(p in poses for p in (
        "idle", "present", "point", "wave", "think",
        "explain", "celebrate", "step"))


def test_all_gestures_generate_valid_svg():
    animator = MascotAnimator()
    for gesture in animator.poses:
        if gesture == "schema":
            continue
        for elapsed in (0, .2, 1.5, 2.8, 3):
            doc = animator.frame_svg(gesture, elapsed, 3)
            assert "@@" not in doc
            root = ET.fromstring(doc)
            assert root.tag.endswith("svg")
            angles = animation_angles(animator.rig, animator.poses, gesture, elapsed, 3)
            for part, angle in angles.items():
                limits = animator.rig["parts"][part]
                assert limits["min_deg"] <= angle <= limits["max_deg"]
    assert all(angle == 0 for angle in
               animation_angles(animator.rig, animator.poses, "point", 3, 3).values())


def test_composite_reels_frame():
    animator = MascotAnimator()
    frame = Image.new("RGB", (1080, 1920), "#f8f6f0")
    bbox = draw_rig_character(frame, animator, (130, 240, 900, 1510),
                              "present", 1.3, 3)
    assert bbox[2] >= 300
    assert frame.getpixel((500, 500)) != (248, 246, 240)
    assert frame.size == (1080, 1920)


def test_cairo_png():
    image = MascotAnimator().frame_image("wave", 1.25, 3, width=256)
    assert image.size == (256, 384)
    assert image.mode == "RGBA"
    assert image.getbbox() is not None
