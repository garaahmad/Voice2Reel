from PIL import Image, ImageDraw, ImageFont

from voice2reel.text_layout import draw_text, text_length, visual_text


def test_visual_text_keeps_latin_readable_and_reshapes_arabic():
    text = "مَرْحَبًا اليوم راح نشرح الـ for loop"
    visual = visual_text(text, "rtl")
    # Latin tokens keep their letters but appear in visual RTL order.
    assert "loop for" in visual
    assert visual != text
    assert "\u064e" in visual  # fatha survives the fallback reshaper
    assert visual_text("plain Latin", "ltr") == "plain Latin"


def test_draw_text_without_raqm_does_not_raise():
    font_path = "voice2reel/assets/Cairo.ttf"
    font = ImageFont.truetype(font_path, 26)
    image = Image.new("RGB", (540, 100), "white")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (270, 10), "مرحبا بالعالم", font, "rtl", fill="black", anchor="mt")
    assert text_length("مرحبا بالعالم", font, "rtl") > 0
    assert image.getbbox() is not None
