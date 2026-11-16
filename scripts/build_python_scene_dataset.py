"""Create executable-by-parser Python scene DSL supervision with held-out topics."""
import argparse
import json
from pathlib import Path
from voice2reel.python_scene_dsl import parse_python_scene, PYTHON_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {
    "train": ["git", "python", "debugging", "algorithms", "databases", "security",
              "productivity", "learning", "web design", "mobile apps"],
    "validation": ["networking", "accessibility"],
    "test": ["testing", "cloud deployment"],
}
GESTURES = ["present", "point", "think", "explain", "wave", "celebrate", "step"]
CAMERAS = ["medium", "close", "wide"]

def build():
    output = {}
    for split, topics in TOPICS.items():
        rows = []
        for topic in topics:
            for variant in range(24):
                duration = 3 + variant % 3
                idea = f"Explain {topic} for beginners, version {variant + 1}, in a short animated video"
                lines = [f'scene(title={json.dumps(topic.title())}, fps=30, width=540, height=960)']
                for j in range(4):
                    lines.append("shot(" + ", ".join([
                        "caption=" + json.dumps(f"{topic.title()}: step {j + 1}"),
                        "narration=" + json.dumps(
                            f"Here is practical {topic} guidance, part {j + 1}. "
                            f"Try this step in your next project."
                        ),
                        f"seconds={duration}.0",
                        "gesture=" + json.dumps(GESTURES[(variant + j) % len(GESTURES)]),
                        "camera=" + json.dumps(CAMERAS[(variant + j) % len(CAMERAS)]),
                        'transition="fade"',
                    ]) + ")")
                response = "\n".join(lines)
                parse_python_scene(response)
                rows.append({
                    "id": f"{split}-{topic.replace(' ', '-')}-{variant:02d}",
                    "messages": [
                        {"role": "system", "content": PYTHON_SYSTEM_PROMPT},
                        {"role": "user", "content": idea},
                        {"role": "assistant", "content": response},
                    ],
                    "provenance": "synthetic, template-authored; not artist-reviewed",
                })
        output[split] = rows
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT / "datasets/python_scene"))
    args = p.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in build().items():
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
        print(split, len(rows))

if __name__ == "__main__":
    main()
