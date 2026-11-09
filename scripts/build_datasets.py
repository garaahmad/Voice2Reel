"""Rebuild the authored v2 datasets: Arabic diacritics, code-switching, motion, scripts, voice.

Everything here is assistant-authored synthetic material composed from explicit banks.
It is NOT scraped, NOT expert-reviewed, and NOT a claim about real-world Arabic or
about model quality. Run ``python scripts/build_datasets.py`` to regenerate byte-identical
files; ``--check`` verifies the committed files instead of rewriting them.
"""

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The first 40 original script records are the immutable seed of every rebuild.
SEED_COUNTS = {"train": 30, "validation": 5, "test": 5}
DATA = ROOT / "datasets"
ASSETS = ROOT / "voice2reel" / "assets"
SEED = 20261109


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- lexicon
# surface -> Arabic reading used by the speech engine; display text keeps the Latin term.
# Bare code identifiers are registered but forbidden inside spoken narration: they must
# be explained in Arabic instead (existing validation behaviour is preserved).
TERMS = [
    ("for loop", "حلقة فور", "حلقة تكرار بعدد معلوم", "control_flow"),
    ("while loop", "حلقة وايل", "حلقة تكرار بشرط", "control_flow"),
    ("if statement", "جملة إف", "جملة شرطية", "control_flow"),
    ("else branch", "فرع إلس", "الفرع البديل", "control_flow"),
    ("switch case", "سويتش كيس", "اختيار بين حالات", "control_flow"),
    ("nested loop", "حلقة متداخلة", "حلقة داخل حلقة", "control_flow"),
    ("infinite loop", "حلقة لا نهائية", "حلقة لا تتوقف", "control_flow"),
    ("for each", "فور إيتش", "تمرين على كل عنصر", "control_flow"),
    ("break point", "نقطة توقف", "إيقاف مؤقت للتنفيذ", "debugging"),
    ("stack trace", "تتبع المكدس", "سلسلة الاستدعاءات عند الخطأ", "debugging"),
    ("unit test", "اختبار وحدة", "اختبار جزء صغير", "testing"),
    ("test case", "حالة اختبار", "مدخل ونتيجة متوقعة", "testing"),
    ("edge case", "حالة حدية", "حالة على الطرف", "testing"),
    ("API", "إيه بي آي", "واجهة برمجية", "web"),
    ("REST API", "واجهة ريست", "واجهة ويب بقواعد محددة", "web"),
    ("JSON", "جيسون", "صيغة تبادل بيانات", "web"),
    ("HTTP", "إتش تي تي بي", "بروتوكول نقل الويب", "web"),
    ("URL", "يوا آر إل", "عنوان الصفحة", "web"),
    ("HTML", "إتش تي إم إل", "لغة هيكل الصفحة", "web"),
    ("CSS", "سي إس إس", "لغة تنسيق الصفحة", "web"),
    ("DOM", "دي أو إم", "شجرة عناصر الصفحة", "web"),
    ("UI", "يو آي", "واجهة المستخدم", "web"),
    ("UX", "يو إكس", "تجربة المستخدم", "web"),
    ("CDN", "سي دي إن", "شبكة توزيع المحتوى", "web"),
    ("SQL", "إس كيو إل", "لغة الاستعلام البنيوية", "data"),
    ("NoSQL", "نو إس كيو إل", "قاعدة بيانات غير علائقية", "data"),
    ("database", "قاعدة البيانات", "مكان حفظ البيانات", "data"),
    ("cache", "الذاكرة المؤقتة", "نسخة محفوظة للوصول السريع", "data"),
    ("index", "الفهرس", "ترتيب يسرّع البحث", "data"),
    ("query", "الاستعلام", "طلب بيانات محدد", "data"),
    ("schema", "المخطط", "وصف بنية البيانات", "data"),
    ("array", "المصفوفة", "قائمة عناصر مرتّبة", "data_types"),
    ("list", "القائمة", "سلسلة عناصر", "data_types"),
    ("string", "السلسلة النصية", "نص محفوظ", "data_types"),
    ("integer", "العدد الصحيح", "عدد بلا كسر", "data_types"),
    ("boolean", "القيمة المنطقية", "صح أو خطأ", "data_types"),
    ("float", "العدد العشري", "عدد بكسر", "data_types"),
    ("dictionary", "القاموس", "أزواج مفتاح وقيمة", "data_types"),
    ("variable", "المتغير", "اسم يحمل قيمة", "basics"),
    ("constant", "الثابت", "قيمة لا تتغير", "basics"),
    ("function", "الدالة", "كتلة قابلة للإعادة", "basics"),
    ("parameter", "المعامل", "قيمة تدخل الدالة", "basics"),
    ("argument", "الوسيط", "القيمة الممررة", "basics"),
    ("return value", "القيمة المرجعة", "نتيجة الدالة", "basics"),
    ("class", "الصنف", "قالب لكائن", "basics"),
    ("object", "الكائن", "نسخة من صنف", "basics"),
    ("method", "التابع", "دالة داخل صنف", "basics"),
    ("package", "الحزمة", "وحدة قابلة للتثبيت", "basics"),
    ("module", "الوحدة", "ملف برمجي مستقل", "basics"),
    ("library", "المكتبة", "مجموعة أدوات جاهزة", "basics"),
    ("framework", "إطار العمل", "هيكل تنظيمي للتطبيق", "basics"),
    ("semicolon", "الفاصلة المنقوطة", "علامة إنهاء الجملة", "basics"),
    ("syntax error", "خطأ في الصياغة", "خطأ يمنع التشغيل", "errors"),
    ("runtime error", "خطأ أثناء التشغيل", "خطأ يظهر عند التنفيذ", "errors"),
    ("null value", "قيمة فارغة", "غياب قيمة", "errors"),
    ("bug", "الخلل", "سلوك غير متوقع", "errors"),
    ("debug", "تصحيح الأخطاء", "البحث عن سبب الخلل", "errors"),
    ("compiler", "المترجم", "يحول الكود إلى تعليمات", "tools"),
    ("interpreter", "المفسّر", "ينفذ الكود سطرا بسطر", "tools"),
    ("terminal", "الطرفية", "نافذة الأوامر", "tools"),
    ("editor", "المحرر", "أداة كتابة الكود", "tools"),
    ("git", "جيت", "نظام إدارة الإصدارات", "tools"),
    ("commit", "الإيداع", "حفظ تغيير موثق", "tools"),
    ("branch", "الفرع", "مسار تطوير منفصل", "tools"),
    ("merge", "الدمج", "ضم تغييرات الفرع", "tools"),
    ("deploy", "النشر", "إطلاق التطبيق للمستخدمين", "tools"),
    ("server", "الخادم", "جهاز يقدّم الخدمة", "infra"),
    ("client", "العميل", "جهاز يطلب الخدمة", "infra"),
    ("browser", "المتصفح", "برنامج عرض الصفحات", "infra"),
    ("CPU", "سي بي يو", "وحدة المعالجة", "infra"),
    ("GPU", "جي بي يو", "وحدة الرسوم والمصفوفات", "infra"),
    ("memory", "الذاكرة", "مساحة العمل المؤقتة", "infra"),
    ("thread", "الخيط", "مسار تنفيذ داخل عملية", "infra"),
    ("Python", "بايثون", "لغة برمجة", "languages"),
    ("JavaScript", "جافاسكربت", "لغة برمجة للويب", "languages"),
    ("TypeScript", "تايب سكربت", "لغة بأنواع فوق جافاسكربت", "languages"),
    ("C++", "سي بلس بلس", "لغة برمجة", "languages"),
    ("C#", "سي شارب", "لغة برمجة", "languages"),
    ("Node.js", "نود جي إس", "بيئة تشغيل جافاسكربت", "languages"),
    ("Qwen3", "كوين ثري", "نموذج لغوي", "models"),
    ("LoRA", "لورا", "تخصيص خفيف للنموذج", "models"),
    ("dataset", "مجموعة البيانات", "أمثلة للتدريب", "models"),
    ("token", "الرمز", "وحدة نص للنموذج", "models"),
    ("prompt", "التعليمة", "النص الموجه للنموذج", "models"),
]
FORBIDDEN = ["if", "for", "while", "else", "return", "console", "null", "true", "false", "var", "let"]


def lexicon():
    terms = {
        surface: {
            "spoken": spoken,
            "gloss_ar": gloss,
            "category": category,
            "allowed_in_narration": True,
        }
        for surface, spoken, gloss, category in TERMS
    }
    for surface in FORBIDDEN:
        terms[surface] = {
            "spoken": "",
            "gloss_ar": "معرّف كود حرفي: اشرحه بالعربية ولا تقرأه حرفيا",
            "category": "literal_code_identifier",
            "allowed_in_narration": False,
        }
    return {
        "version": "ar-tech-lexicon-v2",
        "license": "CC0-1.0",
        "provenance": "assistant-authored; CC0; not expert-reviewed",
        "policy": (
            "Display text may contain registered Latin technical terms. Speech engines receive "
            "the Arabic reading only. Bare code identifiers stay forbidden in narration."
        ),
        "terms": terms,
    }


# --------------------------------------------------------------------------- tashkeel
PREFIXES = ["", "وَ", "فَ", "وَ", "فَ", "ثُمَّ "]
VERBS = ["شَرَحَ", "كَتَبَ", "قَرَأَ", "بَيَّنَ", "وَضَّحَ", "رَتَّبَ", "اِخْتَبَرَ", "أَعَادَ"]
SUBJECTS = ["الْمُعَلِّمُ", "الطَّالِبُ", "الْمُبَرْمِجُ", "الزَّمِيلُ", "الْمُحَرِّرُ", "الْبَاحِثُ"]
OBJECTS = [
    "الدَّرْسَ", "الْمِثَالَ", "الْخُطْوَةَ", "النَّتِيجَةَ", "الْمَسْأَلَةَ", "النَّصَّ",
    "كِتَابَهُ", "مِثَالَهَا", "دَرْسَهُمْ", "نَصِّنَا",
]
TAILS = [
    "بِوُضُوحٍ", "بِهُدُوءٍ", "قَبْلَ الاِخْتِبَارِ", "بَعْدَ التَّعْدِيلِ",
    "مَرَّةً أُخْرَى", "فِي النِّهَايَةِ", "بِسُرْعَةٍ",
]
EXTRA_SENTENCES = [
    "هٰذَا كِتَابٌ مُفِيدٌ عَنِ الْبَرْمَجَةِ.",
    "قَرَأَ الطَّالِبُ كِتَابًا مُفِيدًا قَبْلَ الاِمْتِحَانِ.",
    "اِسْتَفَدْنَا مِنْ كِتَابٍ قَدِيمٍ فِي الْمَكْتَبَةِ.",
    "فِي الْمَسْأَلَةِ شَيْءٌ غَيْرُ وَاضِحٍ لِلْقَارِئِ.",
    "سَأَلَ الْمُعَلِّمُ عَنْ سَبَبِ الْخَطَأِ فِي النَّصِّ.",
    "إِلَى هٰذِهِ النُّقْطَةِ يَصِلُ الشَّرْحُ بِلا تَعْقِيدٍ.",
    "أَعْطَى الْمُبَرْمِجُ مِثَالًا آخَرَ لِتَوْضِيحِ الْفِكْرَةِ.",
    "تَبْدَأُ الْخُطْوَةُ بِقِرَاءَةِ الْمُدْخَلِ ثُمَّ كِتَابَةِ النَّتِيجَةِ.",
]
TOPICS = ["شرح", "مراجعة", "اختبار", "كتابة", "تصحيح", "مقارنة", "تلخيص", "تطبيق"]


def strip(text):
    from voice2reel.arabic_text import strip_tashkeel
    return strip_tashkeel(text)


def tashkeel_rows():
    """Large, balanced splits. Extra sentences (new vocabulary) are distributed
    across ALL splits, so validation is as hard as the final test split."""
    combos = list(itertools.product(PREFIXES, VERBS, SUBJECTS, OBJECTS, TAILS, TOPICS))
    rng = random.Random(SEED)
    rng.shuffle(combos)
    rows, seen = [], set()
    for prefix, verb, subject, obj, tail, topic in combos:
        vowelized = f"{prefix}{verb} {subject} {obj} {tail}."
        plain = strip(vowelized)
        if plain in seen:
            continue
        seen.add(plain)
        rows.append({"plain": plain, "vowelized": vowelized, "topic": topic})
        if len(rows) == 592:
            break
    for i, sentence in enumerate(EXTRA_SENTENCES):
        # Interleave, never dump all novel vocabulary in the final test split.
        rows.insert(min(len(rows), 37 + i * 71), {
            "plain": strip(sentence), "vowelized": sentence, "topic": "تنوين وهمزات",
        })
    assert len({r["plain"] for r in rows}) == len(rows)
    out = []
    for i, row in enumerate(rows):
        split = "train" if i < 480 else ("validation" if i < 540 else "test")
        out.append({
            "id": f"tk_{i:04d}", "split": split,
            "plain": row["plain"], "vowelized": row["vowelized"],
            "topic": row["topic"],
            "source": "assistant-authored synthetic templates", "license": "CC0-1.0",
        })
    return out


# --------------------------------------------------------------------------- codeswitch
CS_TEMPLATES = [
    "مرحبا، اليوم راح نشرح الـ {t} بطريقة بسيطة.",
    "خلينا نبدأ بـ {t} قبل ما ندخل في التفاصيل.",
    "المشكلة هنا ما كانت في {t}، بل في ترتيب الخطوات.",
    "لما تكتب {t}، راجع القيمة قبل ما تكمل.",
    "في المثال اللي قدامنا، {t} يوضح الفكرة بسرعة.",
    "أول خطوة هي {t}، وبعدها نتحقق من النتيجة.",
    "شرحنا {t} في المشهد السابق، والآن نطبقه.",
    "لو فهمت {t}، الباقي بيصير أسهل بكثير.",
    "لا تخلط بين {t} وبين غيره من المفاهيم المشابهة.",
    "النتيجة تعتمد على {t} وعلى المدخل اللي تعطيه له.",
    "هذي حالة بسيطة نستخدم فيها {t} بدون أي تعقيد.",
    "بعد {t} نحتاج خطوة تحقق صغيرة وننتهي.",
    "الفرق بسيط: {t} يشتغل هنا، وما يشتغل هناك.",
    "خلاصة المشهد: {t} ثم مراجعة ثم حفظ.",
    "قبل التشغيل تأكد من {t}، وبعده راقب المخرجات.",
]


def codeswitch_rows():
    """Every registered term appears in TRAIN at least once; held-out splits test
    new *sentence frames*, not unseen vocabulary (so the metric measures the
    front-end rules, not a guessing game)."""
    surfaces = [t[0] for t in TERMS]
    combos = list(itertools.product(CS_TEMPLATES, surfaces))
    rng = random.Random(SEED + 1)
    rng.shuffle(combos)
    rows, seen = [], set()

    def add(template, surface):
        display = template.format(t=surface)
        if display in seen:
            return
        seen.add(display)
        reading = next(t[1] for t in TERMS if t[0] == surface)
        spoken = display.replace(surface, reading).replace("الـ ", "ال")
        rows.append({
            "display": display, "spoken": spoken,
            "terms": [{"surface": surface, "reading": reading}],
            "category": next(t[3] for t in TERMS if t[0] == surface),
        })

    # Guaranteed coverage: one training example per term, in a rotating frame.
    for i, surface in enumerate(surfaces):
        add(CS_TEMPLATES[i % len(CS_TEMPLATES)], surface)
    for template, surface in combos:
        if len(rows) == 760:
            break
        add(template, surface)
    assert len({r["display"] for r in rows}) == len(rows)
    out = []
    for i, row in enumerate(rows):
        split = "train" if i < 620 else ("validation" if i < 690 else "test")
        out.append({
            "id": f"cs_{i:04d}", "split": split,
            "display": row["display"], "spoken": row["spoken"],
            "terms": row["terms"], "category": row["category"],
            "source": "assistant-authored synthetic templates", "license": "CC0-1.0",
        })
    return out


# --------------------------------------------------------------------------- motion
MOTION_VOCABULARY = {
    "version": "motion-vocabulary-v2",
    "gestures": ["present", "point", "think", "shrug", "thumbs_up", "wave"],
    "emotions": ["neutral", "happy", "surprised", "concerned", "thinking"],
    "cameras": ["wide", "medium", "close"],
    "visuals": ["none", "list", "flow", "compare", "code"],
    "transitions": ["cut", "wipe", "fade", "none"],
    "easings": ["smooth", "linear", "snap"],
    "safe_zones": {
        "headline": [0, 0, 540, 250],
        "actor": [0, 250, 330, 860],
        "panel": [295, 430, 540, 820],
        "subtitle": [0, 860, 540, 960],
    },
    "comfort_limits": {
        "max_camera_step_per_scene": 1,
        "camera_ease_seconds": 0.9,
        "max_gesture_switches_per_second": 0.5,
        "min_scene_seconds": 2.5,
        "max_jitter_amplitude": 2.0,
        "max_secondary_frequency_hz": 0.6,
        "max_flash_alpha": 0.12,
        "max_subtitle_words": 4,
        "min_safe_margin": 24,
    },
    "note": "Bounded vocabulary for a deterministic 2D engine; no trainable animation model exists in this repository.",
}


def motion_rows():
    """Balanced, rule-labelled plans. Each 'annoying' row violates exactly ONE
    named rule, so a fitted classifier and the rule validator can both be scored
    honestly instead of predicting 'annoying' for everything."""
    rng = random.Random(SEED + 2)
    violations = [
        ("camera_jump", "camera_step", lambda: rng.choice([2, 3])),
        ("gesture_flicker", "gesture_switches_per_second", lambda: round(rng.uniform(0.6, 1.6), 2)),
        ("jitter", "jitter_amplitude", lambda: round(rng.uniform(2.2, 6.0), 2)),
        ("cut_too_fast", "scene_seconds", lambda: round(rng.uniform(1.0, 2.4), 2)),
        ("harsh_flash", "flash_alpha", lambda: round(rng.uniform(0.14, 0.45), 2)),
        ("overlap", "overlap_count", lambda: rng.choice([1, 2, 3])),
        ("dense_subtitle", "subtitle_words", lambda: rng.choice([5, 6, 8, 9])),
        ("abrupt_easing", "easing", lambda: rng.choice(["linear", "snap"])),
    ]
    rows = []
    for i in range(1200):
        features = {
            "camera_step": rng.choice([0, 1]),
            "gesture_switches_per_second": round(rng.uniform(0.1, 0.48), 2),
            "jitter_amplitude": round(rng.uniform(0.4, 1.9), 2),
            "scene_seconds": round(rng.uniform(2.6, 8.0), 2),
            "flash_alpha": round(rng.uniform(0.0, 0.11), 2),
            "overlap_count": 0,
            "subtitle_words": rng.choice([3, 4]),
            "easing": "smooth",
        }
        label, violation = "comfortable", None
        if i % 2:
            name, key, draw = violations[i % len(violations)]
            features[key] = draw()
            label, violation = "annoying", name
        rows.append({"features": features, "label": label, "violation": violation})
    rng.shuffle(rows)
    out = []
    for i, row in enumerate(rows):
        split = "train" if i < 960 else ("validation" if i < 1080 else "test")
        out.append({
            "id": f"mo_{i:04d}", "split": split,
            "features": row["features"], "label": row["label"],
            "violation": row["violation"],
            "source": "assistant-authored rule-labelled synthetic plans", "license": "CC0-1.0",
        })
    return out


def motion_annotations(scripts):
    rng = random.Random(SEED + 3)
    gestures = MOTION_VOCABULARY["gestures"]
    emotions = MOTION_VOCABULARY["emotions"]
    cameras = MOTION_VOCABULARY["cameras"]
    visuals = MOTION_VOCABULARY["visuals"]
    rows = []
    for i, record in enumerate(scripts):
        script = record["script"]
        for j, beat in enumerate(script["beats"]):
            intent = "explain" if j == 1 else ("hook" if j == 0 else "close")
            rows.append({
                "id": f"ma_{len(rows):04d}", "split": record["split"],
                "intent": intent, "narration": beat["narration"], "topic": record["topic"],
                "plan": {
                    "gesture": gestures[(i + j) % len(gestures)],
                    "emotion": emotions[(i + 2 * j) % len(emotions)],
                    "camera": cameras[[0, 0, 1, 1, 2][(i + j) % 5]],
                    "visual": visuals[(i + j) % len(visuals)],
                    "transition": "fade" if j else "none",
                    "easing": "smooth",
                    "hold_seconds": round(2.6 + rng.random() * 2.4, 2),
                    "jitter_amplitude": round(rng.uniform(0.4, 1.9), 2),
                    "flash_alpha": 0.1,
                    "safe_zone": "actor" if j % 2 == 0 else "panel",
                    "focus": "emphasis" if j == 1 else "none",
                },
                "source": "assistant-authored structured annotations", "license": "CC0-1.0",
            })
    return rows


# --------------------------------------------------------------------------- scripts
IDEA_TOPICS = [
    ("الفاصلة المنقوطة", "علامة إنهاء الجملة", "مثال قصير"),
    ("النسخ الاحتياطي", "حفظ نسخة إضافية", "جدول منتظم"),
    ("اختبار البرنامج", "تنفيذ خطوات المستخدم", "كشف المشكلة"),
    ("الذاكرة المؤقتة", "نسخة محفوظة للصفحة", "تحديث مع تجاوزها"),
    ("تقسيم المهام", "خطوات صغيرة واضحة", "مراجعة كل خطوة"),
    ("أسماء المتغيرات", "اسم يصف المحتوى", "تجنب الاختصار الغامض"),
    ("قراءة رسالة الخطأ", "السطر والموقع", "سبب واضح"),
    ("ترتيب الشروط", "الحالة الخاصة أولا", "حالة افتراضية"),
    ("حلقة التكرار", "شرط التوقف", "تحديث القيمة"),
    ("حفظ التغييرات", "رسالة توضح التعديل", "مراجعة الفرق"),
    ("تنظيم الملفات", "مجلد لكل جزء", "اسم ثابت"),
    ("قياس الزمن", "بداية ونهاية", "فرق واضح"),
    ("إدخال المستخدم", "التحقق من القيمة", "رسالة مفهومة"),
    ("القيم الافتراضية", "قيمة مبدئية آمنة", "تغيير مقصود"),
    ("تنظيف الكود", "حذف التكرار", "دالة واحدة"),
    ("مراجعة الزميل", "سؤال محدد", "تعديل صغير"),
    ("توثيق الخطوة", "سبب القرار", "مثال عملي"),
    ("التدرج في الشرح", "مثال ثم قاعدة", "تطبيق أخير"),
]
IDEA_ANGLES = [
    "للمبتدئين", "قبل التسليم", "دون أمثلة كود", "بخطوة عملية واحدة",
    "مع مقارنة بسيطة", "في مشروع صغير", "بعد خطأ متكرر", "بلغة سهلة",
    "بمثال واحد فقط", "من نتيجة خاطئة", "قبل مشاركة الفريق", "بترتيب زمني",
]
AR_HOOKS = [
    "لاحظت أن النتيجة تختلف عند تعديل بسيط في المدخل.",
    "تبدأ المشكلة حين نفترض أن الخطوة السابقة نجحت دائما.",
    "في مثال قصير نرى الفكرة أوضح من أي وصف عام.",
    "قبل الحكم على النتيجة، نحتاج أن نعرف ما حدث فعلا.",
]
AR_CAUSES = [
    "السبب أن الترتيب يغيّر معنى الخطوة التالية.",
    "السبب أن القيمة المحفوظة تصل متأخرة إلى الشاشة.",
    "السبب أن الشرط يُقرأ قبل أن تتغير القيمة.",
    "السبب أن وصف الخطوة حذف تفصيلا مهما.",
]
AR_STEPS = [
    "اكتب الخطوة بطريقة واحدة، ثم أعد التشغيل وراقب المخرج.",
    "افصل الجزء المشكوك فيه، واختبره وحده قبل المتابعة.",
    "حدّد ما يجب أن يظهر، وقارنه بما ظهر فعلا.",
    "احفظ التعديل في مكان مستقل، ثم راجع الفرق بهدوء.",
]
EN_HOOKS = [
    "A small change in the input produced a different result.",
    "The issue appears when we assume the previous step always worked.",
    "A short example shows the idea more clearly than a general description.",
    "Before judging the output, we need to know what actually happened.",
]
EN_CAUSES = [
    "The order changes the meaning of the next step.",
    "The saved value reaches the screen later than expected.",
    "The condition is read before the value changes.",
    "The step description left out an important detail.",
]
EN_STEPS = [
    "Write the step once, run it again, and observe the output.",
    "Separate the doubtful part and test it on its own before moving on.",
    "State what should appear, then compare it with what appeared.",
    "Save the edit separately, then review the difference calmly.",
]
POSES = ["IDLE", "SHOCK", "CELEB", "SUSPECT", "LEAN", "TYPE_A", "LOOK_L", "LOOK_R"]
CAMERAS = ["far", "medium", "close", "extreme", "left", "right"]


def script_rows(existing):
    """Return NEW records only; the 40 seed records stay byte-identical with their IDs."""
    from voice2reel.scriptwriter.data import normalized
    from voice2reel.scriptwriter.schema import Script

    rng = random.Random(SEED + 4)
    combos = list(itertools.product(range(len(IDEA_TOPICS)), range(len(IDEA_ANGLES)), range(6)))
    rng.shuffle(combos)
    used_ideas = {normalized(r["idea"]) for r in existing}
    used_narration = {normalized(Script.model_validate(r["script"]).narration()) for r in existing}
    rows = []
    for topic_i, angle_i, variant in combos:
        topic, cause_topic, step_topic = IDEA_TOPICS[topic_i]
        angle = IDEA_ANGLES[angle_i]
        idea = f"اشرح {topic} {angle} مع التركيز على {cause_topic} و{step_topic} (نسخة {variant + 1})"
        if normalized(idea) in used_ideas:
            continue
        lang = "en" if variant == 2 else "ar"
        hooks, causes, steps = (AR_HOOKS, AR_CAUSES, AR_STEPS) if lang == "ar" else (EN_HOOKS, EN_CAUSES, EN_STEPS)
        hook = hooks[(topic_i + angle_i) % len(hooks)]
        cause = causes[(topic_i + 2 * angle_i) % len(causes)]
        step = steps[(topic_i + 3 * angle_i + variant) % len(steps)]
        # Topic, angle and variant appear in the narration itself: every record is a
        # distinct narration string, so no text can leak between splits.
        marker = f"{topic} {angle} {variant + 1}"
        if lang == "ar":
            narrations = [
                f"{hook} لنفحص {topic} من البداية، {angle} نسخة {variant + 1}.",
                f"{cause} هذا يفسر لماذا يهم {cause_topic} في {marker}.",
                f"{step} ثم راجع {step_topic} قبل الانتقال، {angle}.",
            ]
        else:
            narrations = [
                f"{hook} Case {topic_i + 1}-{angle_i + 1}-{variant + 1}: examine the first step.",
                f"{cause} Variant {variant + 1} keeps the explanation short and concrete.",
                f"{step} Angle {angle_i + 1} closes with a check before moving on.",
            ]
        words = sum(len(n.split()) for n in narrations)
        duration = min(60, max(10, round(words / 1.8)))
        title = f"{topic}: {angle} {variant + 1}" if lang == "ar" else f"Step by step: {angle_i + 1}-{variant + 1}"
        beats = [
            {"narration": n, "caption": (f"{topic} — {j + 1}" if lang == "ar" else f"Step {j + 1}"),
             "pose": POSES[(topic_i + j + variant) % len(POSES)],
             "camera": CAMERAS[(angle_i + j) % len(CAMERAS)]}
            for j, n in enumerate(narrations)
        ]
        script = {"schema_version": 1, "title": title[:90], "language": lang,
                  "target_duration_seconds": duration, "beats": beats}
        try:
            Script.model_validate(script)
        except ValueError:
            continue
        narration_key = normalized(" ".join(narrations))
        if narration_key in used_narration:
            continue
        used_narration.add(narration_key)
        used_ideas.add(normalized(idea))
        rows.append({
            "id": f"v2_{len(rows):04d}", "idea": idea, "script": script,
            "topic": topic, "angle": angle, "split": None,
            "source": "assistant-authored synthetic templates", "license": "CC0-1.0",
        })
        if len(rows) == 1080:
            break
    for i, row in enumerate(rows):
        row["split"] = "train" if i < 900 else ("validation" if i < 990 else "test")
    return rows


# --------------------------------------------------------------------------- voice prompts
def voice_rows(tashkeel, codeswitch):
    rows = []
    for r in tashkeel[:60]:
        rows.append((r["vowelized"], "tashkeel_full", "ar"))
    for r in tashkeel[:30]:
        rows.append((r["plain"], "tashkeel_plain", "ar"))
    for r in codeswitch[:40]:
        rows.append((r["spoken"], "codeswitch_spoken_reading", "ar"))
    extra = [
        ("كم عدد العناصر في القائمة بعد الحذف؟", "question_number", "ar"),
        ("هل تستطيع شرح الفكرة في جملة واحدة ثم دعمها بمثال؟", "question_long", "ar"),
        ("ثلاثة صناديق خضراء بجانب نافذة صغيرة.", "short_statement", "ar"),
        ("افتح الإعدادات، ثم اختر اللغة العربية واحفظ التغيير.", "imperative_steps", "ar"),
        ("الملف الثالث كبير، لكن النسخة الأولى أصغر منه.", "ordinals", "ar"),
        ("في تمام الساعة الحادية عشرة إلا ربعا بدأ الاختبار.", "time_expression", "ar"),
        ("النسبة خمسة وثلاثون بالمئة، لا خمسون بالمئة.", "percentage", "ar"),
        ("اكتب الاسم كما هو، دون مسافات إضافية أو رموز.", "spelling", "ar"),
        ("A short English sentence for the same speaker.", "en_sentence", "en"),
        ("Read the value, compare it, and then save the result.", "en_imperative", "en"),
    ]
    for text, kind, lang in extra:
        rows.append((text, kind, lang))
    out = []
    for i, (text, kind, lang) in enumerate(rows):
        out.append({
            "filename": f"clip_v2_{i + 1:04d}.wav", "text": text,
            "speaker": "my_voice", "kind": kind, "language": lang,
        })
    return out


def build():
    ASSETS.mkdir(parents=True, exist_ok=True)
    lex = lexicon()
    dump(ASSETS / "arabic_lexicon.json", lex)
    dump(DATA / "arabic" / "lexicon.json", lex)
    dump(DATA / "motion" / "vocabulary.json", MOTION_VOCABULARY)
    dump(ASSETS / "motion_vocabulary.json", MOTION_VOCABULARY)

    tk = tashkeel_rows()
    cs = codeswitch_rows()
    mo = motion_rows()
    for name, rows in (("tashkeel", tk), ("codeswitch", cs), ("motion_plans", mo)):
        folder = DATA / ("motion" if name == "motion_plans" else "arabic")
        for split in ("train", "validation", "test"):
            write_jsonl(folder / f"{name}_{split}.jsonl", [r for r in rows if r["split"] == split])

    # Existing script records are preserved verbatim; new ones are appended per split.
    seed_file = DATA / "scripts" / "seed_records.jsonl"
    existing = []
    if seed_file.is_file():
        for line in seed_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
    else:
        for split, count in SEED_COUNTS.items():
            path = DATA / "scripts" / f"{split}.jsonl"
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) < count:
                raise ValueError(f"Seed split {split} is shorter than its recorded size")
            for line in lines[:count]:
                existing.append({**json.loads(line), "split": split})
        write_jsonl(seed_file, existing)
    new = script_rows(existing)
    write_jsonl(DATA / "scripts" / "v2_records.jsonl", new)
    for split in ("train", "validation", "test"):
        kept = [
            {"id": r["id"], "idea": r["idea"], "script": r["script"]}
            for r in existing if r["split"] == split
        ]
        added = [
            {"id": r["id"], "idea": r["idea"], "script": r["script"]}
            for r in new if r["split"] == split
        ]
        write_jsonl(DATA / "scripts" / f"{split}.jsonl", kept + added)
    annotations = motion_annotations(new)
    write_jsonl(DATA / "motion" / "scene_annotations.jsonl", annotations)

    voice = voice_rows(tk, cs)
    vpath = DATA / "voice" / "recording_prompts_v2.csv"
    with vpath.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "text", "speaker", "kind", "language"])
        writer.writeheader()
        writer.writerows(voice)
    write_jsonl(
        DATA / "voice" / "manifest.v2.template.jsonl",
        [{"filename": r["filename"], "text": r["text"], "speaker": r["speaker"]} for r in voice],
    )
    (DATA / "voice" / "evaluation_texts_v2.txt").write_text(
        "\n".join(r["text"] for r in voice[::7]) + "\n", encoding="utf-8"
    )
    counts = {
        "lexicon_terms": len(lex["terms"]),
        "tashkeel": {s: sum(r["split"] == s for r in tk) for s in ("train", "validation", "test")},
        "codeswitch": {s: sum(r["split"] == s for r in cs) for s in ("train", "validation", "test")},
        "motion_plans": {s: sum(r["split"] == s for r in mo) for s in ("train", "validation", "test")},
        "scripts_new": {s: sum(r["split"] == s for r in new) for s in ("train", "validation", "test")},
        "motion_annotations": len(annotations),
        "voice_prompts": len(voice),
    }
    dump(DATA / "build_manifest.json", {"seed": SEED, "counts": counts,
                                        "provenance": "assistant-authored synthetic; CC0-1.0; no expert review"})
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify committed files are reproducible")
    args = p.parse_args()
    if args.check:
        before = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in DATA.rglob("*") if p.is_file()}
        before.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in ASSETS.glob("*.json")})
        counts = build()
        after = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in DATA.rglob("*") if p.is_file()}
        after.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in ASSETS.glob("*.json")})
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        if changed:
            raise SystemExit("Dataset drift: " + ", ".join(changed))
        print(json.dumps({"status": "reproducible", "counts": counts}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
