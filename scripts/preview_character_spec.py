"""Preview safe CharacterSpec styles on the existing animated mascot (GPU-free)."""
import argparse
from pathlib import Path
from voice2reel.character_spec import DEFAULT_SPEC, load_spec
from voice2reel.mascot_animator import MascotAnimator, render_pose_sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default=None, help="Previously validated JSON CharacterSpec")
    p.add_argument("--output", default="outputs/character_spec_preview.png")
    p.add_argument("--sheet", default="outputs/character_spec_pose_sheet.png")
    args = p.parse_args()
    spec = load_spec(args.spec) if args.spec else DEFAULT_SPEC
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    MascotAnimator(spec=spec).frame_image("idle", 1.25, 3, width=512).save(output)
    render_pose_sheet(args.sheet, spec=spec, gestures=("idle", "present", "point", "wave"))
    print("✅ Rigged CharacterSpec preview:", output)
    print("✅ Pose sheet:", args.sheet)
    print("✅ Validated style:", spec.model_dump_json())


if __name__ == "__main__":
    main()
