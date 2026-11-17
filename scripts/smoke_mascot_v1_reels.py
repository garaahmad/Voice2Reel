"""GPU-free 1080x1920 Reels integration smoke test for Mascot V1."""
import argparse
from pathlib import Path

from voice2reel.character_templates import lock_template
from voice2reel.production_pipeline import render, verify_reels_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/mascot_v1_smoke.mp4")
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(".lock.json")
    lock_template("classic", lock_path)
    plan = {
        "version": 1, "title": "Rig integration",
        "width": 1080, "height": 1920, "fps": 24,
        "shots": [{
            "caption": "Mascot V1",
            "narration": "Mascot V1",
            "seconds": 1.0, "gesture": "wave",
            "camera": "medium", "transition": "fade",
            "objects": []
        }]
    }
    render(plan, lock_path, output, character_engine="rig")
    verify_reels_video(output)
    print("✅ 1080x1920 (9:16) Mascot V1 rig video:", output)


if __name__ == "__main__":
    main()
