"""Arabic text layout fallback for Pillow builds without libraqm.

Pillow wheels on many machines (including Colab and this sandbox) do not bundle
libraqm. Passing ``direction="rtl"`` to ``ImageDraw.text`` then raises a KeyError.
When RAQM is available we use Pillow's own shaping. Otherwise this module applies
an MIT-licensed Arabic reshaper and a small visual-order pass, so captions stay
connected and ordered. Latin runs are preserved left-to-right.
"""

import re
from functools import lru_cache

from PIL import features

# Arabic blocks, presentation forms, and Arabic punctuation.
ARABIC_RUN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]+")
# Split into Arabic / other runs but keep whitespace as its own token.
TOKENS = re.compile(r"(\s+|[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]+|[^\s\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]+)")


@lru_cache(maxsize=1)
def has_raqm():
    try:
        return bool(features.check_feature("raqm"))
    except (AttributeError, TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _reshaper():
    try:
        import arabic_reshaper
    except ImportError:
        return None
    # Keep full tashkeel visible and avoid presentation-form ligatures that many
    # Arabic fonts (including Cairo.ttf) do not contain.
    return arabic_reshaper.ArabicReshaper({
        "delete_harakat": False,
        "delete_tatweel": False,
        "support_ligatures": False,
        "use_unshaped_instead_of_isolated": True,
    })


def _reshape(text):
    reshaper = _reshaper()
    if reshaper is None:
        return text
    try:
        return reshaper.reshape(text)
    except (TypeError, ValueError):
        return text


def visual_text(text, direction="ltr"):
    """Return the string to draw when Pillow cannot shape Arabic itself.

    This is a *simple* visual-order pass for captions and labels, not a full
    Unicode Bidi implementation. It keeps Arabic words connected and in RTL
    order while leaving technical Latin tokens readable left-to-right.
    """
    if direction != "rtl" or has_raqm() or not isinstance(text, str):
        return text
    tokens = [t for t in TOKENS.findall(text) if t]
    shaped = [
        _reshape(t)[::-1] if not t.isspace() and ARABIC_RUN.fullmatch(t) else t
        for t in tokens
    ]
    return "".join(reversed(shaped))


def text_length(text, font, direction="ltr"):
    """Width of ``text`` in pixels, respecting the active layout backend."""
    if has_raqm():
        return font.getlength(text, direction=direction)
    return font.getlength(visual_text(text, direction))


def draw_text(draw, xy, text, font, direction="ltr", **kwargs):
    """Draw ``text`` with RAQM when present, otherwise the reshaping fallback."""
    if has_raqm():
        return draw.text(xy, text, font=font, direction=direction, **kwargs)
    return draw.text(xy, visual_text(text, direction), font=font, **kwargs)
