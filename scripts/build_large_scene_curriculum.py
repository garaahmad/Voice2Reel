"""Deterministic 200k-scene curriculum with strict schema and topic-held-out splits.

Generate locally, never commit multi-GB datasets. The generator ensures structural
validity and exact deduplication, NOT human-verified cinematographic quality.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from voice2reel.python_scene_dsl import parse_python_scene, PYTHON_SYSTEM_PROMPT

TOPICS = [
    "Python basics", "Git branches", "Flutter widgets", "REST APIs",
    "database indexes", "unit tests", "debugging", "security",
    "data structures", "algorithms", "responsive design", "accessibility",
    "cloud deployment", "networking", "productivity", "machine learning",
    "user experience", "authentication", "caching", "performance",
    "error handling", "version control", "state management", "code reviews",
    "CI pipelines", "Docker", "Linux", "API pagination", "SQL joins",
    "memory management", "web animation", "UI typography", "testing strategy",
    "dependency injection", "file uploads", "app architecture",
]
AR = [
    "خلينا نفهم {topic} خطوة بخطوة",
    "ابدأ بالمشكلة، وبعدها شوف الحل في {topic}",
    "جرّب هذا المثال العملي عن {topic}",
    "تجنب الخطأ الشائع في {topic}",
    "قارن بين الحل البسيط والحل الأفضل في {topic}",
    "راجع النتيجة وطبّق الفكرة على مشروعك",
]
EN = [
    "Let's explore {topic} step by step",
    "Start with the problem before the solution to {topic}",
    "Try this practical example of {topic}",
    "Avoid this common mistake in {topic}",
    "Compare a simple approach with a better {topic} solution",
    "Review the result and apply it in your project",
]
GESTURES = ("idle", "present", "point", "wave", "think", "explain", "celebrate", "step")
CAMERAS = ("wide", "medium", "close")
TRANSITIONS = ("cut", "fade")
FRAMINGS = ((540, 960), (1080, 1920))
LANGUAGES = ("ar", "en")

def split_for(topic):
    # Entire topics are held out. No topic appears in more than one split.
    value = int(hashlib.sha256(topic.encode()).hexdigest()[:8], 16) % 20
    return "test" if value == 0 else "validation" if value == 1 else "train"

STORY_AR = {
    "hook": ("تخيّل إن مشروعك توقف فجأة بسبب {topic}.", "think", "close"),
    "problem": ("المشكلة الشائعة في {topic} هي تجاهل الخطوات الأساسية.", "point", "medium"),
    "explain": ("الفكرة الأساسية: قسّم {topic} إلى خطوات واضحة.", "explain", "wide"),
    "example": ("خلينا نجرّب مثال عملي صغير في {topic}.", "present", "medium"),
    "compare": ("قارن الحل العشوائي مع طريقة منظمة في {topic}.", "point", "wide"),
    "warning": ("انتبه: التسرع في {topic} ممكن يسبب أخطاء.", "think", "close"),
    "result": ("هيك بتوصل لنتيجة أوضح باستخدام {topic}.", "celebrate", "medium"),
    "cta": ("جرّب الفكرة بنفسك وشارك شو تعلمت.", "wave", "medium"),
}
STORY_EN = {
    "hook": ("Imagine your project suddenly failing because of {topic}.", "think", "close"),
    "problem": ("A common {topic} problem is skipping the fundamentals.", "point", "medium"),
    "explain": ("The key idea is to break {topic} into clear steps.", "explain", "wide"),
    "example": ("Let's walk through a small practical {topic} example.", "present", "medium"),
    "compare": ("Compare a rushed approach with a structured {topic} solution.", "point", "wide"),
    "warning": ("Watch out: rushing {topic} can introduce mistakes.", "think", "close"),
    "result": ("Now you can get clearer results with {topic}.", "celebrate", "medium"),
    "cta": ("Try it in your own project and share what you learned.", "wave", "medium"),
}
STORY_AR_CAPTIONS = {
    "hook": "ليش الموضوع مهم؟", "problem": "وين المشكلة؟",
    "explain": "الفكرة ببساطة", "example": "مثال عملي",
    "compare": "قارن الحلول", "warning": "انتبه لهذا الخطأ",
    "result": "النتيجة", "cta": "دورك تجرّب",
}
STORY_EN_CAPTIONS = {
    "hook": "Why it matters", "problem": "The problem",
    "explain": "Core idea", "example": "Practical example",
    "compare": "Compare approaches", "warning": "Common pitfall",
    "result": "The result", "cta": "Your turn",
}
STORY_STRUCTURES = (
    ("hook", "problem", "explain", "example", "result", "cta"),
    ("hook", "warning", "explain", "example", "result"),
    ("problem", "compare", "example", "result", "cta"),
    ("hook", "explain", "example", "warning", "result"),
    ("problem", "explain", "compare", "example", "result", "cta"),
)

def generate(index, seed):
    rng = random.Random(f"{seed}:{index}")
    topic = TOPICS[index % len(TOPICS)]
    lang = LANGUAGES[(index // len(TOPICS)) % 2]
    phrases = AR if lang == "ar" else EN
    structure = rng.choice(STORY_STRUCTURES)
    shots = len(structure)
    width, height = rng.choice(FRAMINGS)
    idea = (
        f"Create a {shots}-shot {'Arabic' if lang == 'ar' else 'English'} "
        f"animated explainer about {topic}. Emphasize "
        f"{rng.choice(('clarity', 'surprise', 'step-by-step reasoning', 'comparison', 'mistakes', 'practical advice'))}. "
        f"Use the locked character without redesigning it. "
        f"Variation ID {index}."
    )
    code = [f"scene(title={json.dumps(topic, ensure_ascii=False)}, fps={rng.choice((24, 30))}, width={width}, height={height})"]
    story = STORY_AR if lang == "ar" else STORY_EN
    captions = STORY_AR_CAPTIONS if lang == "ar" else STORY_EN_CAPTIONS
    for j, beat in enumerate(structure):
        text, gesture, camera = story[beat]
        caption = captions[beat]
        narration = text.format(topic=topic)
        word_count = len(narration.split())
        seconds = round(min(12.0, max(2.5, word_count / (2.5 if lang == "en" else 2.2) + 0.8)), 2)
        code.append("shot(" + ", ".join([
            "caption=" + json.dumps(caption, ensure_ascii=False),
            "narration=" + json.dumps(narration, ensure_ascii=False),
            f"seconds={seconds}",
            "gesture=" + json.dumps(gesture),
            "camera=" + json.dumps(camera),
            "transition=" + json.dumps(rng.choice(TRANSITIONS)),
        ]) + ")")
    response = "\n".join(code)
    parse_python_scene(response)
    return {
        "id": f"scene-{index:07d}",
        "messages": [
            {"role": "system", "content": PYTHON_SYSTEM_PROMPT},
            {"role": "user", "content": idea},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "topic": topic, "language": lang, "shot_count": shots,
            "generator_seed": seed, "story_structure": structure,
            "quality": "synthetic_semantic_choreography_not_artist_reviewed",
            "character_policy": "external_locked_template",
        },
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=200000)
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--output", default="datasets/python_scene_large")
    p.add_argument("--check-every", type=int, default=10000)
    args = p.parse_args()
    if args.count < 1 or args.count > 2000000:
        p.error("count must be 1..2,000,000")
    if args.check_every < 1:
        p.error("--check-every must be positive")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {split: directory / f"{split}.jsonl" for split in ("train", "validation", "test")}
    handles = {split: path.open("w", encoding="utf-8") for split, path in paths.items()}
    counts = {split: 0 for split in paths}
    try:
        for i in range(args.count):
            row = generate(i, args.seed)
            split = split_for(row["metadata"]["topic"])
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            if (i + 1) % args.check_every == 0:
                print(f"Validated and written {i + 1:,}/{args.count:,} scenes", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "seed": args.seed, "count": args.count, "splits": counts,
        "topic_assignment": {t: split_for(t) for t in TOPICS},
        "quality_limitations": [
            "Synthetic template data, not human-reviewed",
            "Repeated phrasing and limited semantic depth",
            "Scene DSL only; no arbitrary Python",
            "No video renders or audio timing quality validated",
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
