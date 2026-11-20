"""Preview the authored rigged mascot without running Qwen training."""
import argparse
from pathlib import Path
from voice2reel.mascot_animator import MascotAnimator, render_pose_sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/mascot_v1_pose_sheet.png")
    p.add_argument("--gesture", default=None)
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.gesture:
        MascotAnimator().frame_image(args.gesture, 1.4, 3, width=512).save(output)
    else:
        render_pose_sheet(output)
    print("✅ Preview:", output)


if __name__ == "__main__":
    main()
