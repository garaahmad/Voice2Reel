from PIL import Image, ImageDraw, ImageFont

import voice2reel.text_layout as layout


def test_visual_text_is_correct_for_both_pillow_backends(monkeypatch):
    text = "مَرْحَبًا اليوم راح نشرح الـ for loop"

    # Pillow/RAQM handles bidi itself, so the fallback helper must not
    # pre-reorder the string. Latin stays readable and tashkeel survives.
    monkeypatch.setattr(layout, "has_raqm", lambda: True)
    visual = layout.visual_text(text, "rtl")
    assert visual == text
    assert "for loop" in visual

    # Without RAQM the fallback reverses run order for visual RTL drawing.
    monkeypatch.setattr(layout, "has_raqm", lambda: False)
    visual = layout.visual_text(text, "rtl")
    assert "loop for" in visual
    assert visual != text
    assert "\u064e" in visual

    assert layout.visual_text("plain Latin", "ltr") == "plain Latin"


def test_draw_text_without_raqm_does_not_raise(monkeypatch):
    monkeypatch.setattr(layout, "has_raqm", lambda: False)
    font = ImageFont.truetype("voice2reel/assets/Cairo.ttf", 26)
    image = Image.new("RGB", (540, 100), "white")
    draw = ImageDraw.Draw(image)
    layout.draw_text(draw, (270, 10), "مرحبا بالعالم", font, "rtl", fill="black", anchor="mt")
    assert layout.text_length("مرحبا بالعالم", font, "rtl") > 0
    assert image.getbbox() is not None
