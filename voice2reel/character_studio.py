"""Locked-character silent clip production and observable script review."""
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from voice2reel.character_templates import JOINT_ORDER, draw_character, get_template, load_lock
from voice2reel.common import write_json
from voice2reel.motion_curriculum import motion_quality


def review_script(script):
    """Observable editorial checks, not a measure of hidden reasoning or truth."""
    captions = [beat.caption for beat in script.beats]
    words = len(script.narration().split())
    return {
        "schema_valid": True,
        "word_count": words,
        "words_per_second": round(words / script.target_duration_seconds, 2),
        "unique_captions": len(set(captions)) == len(captions),
        "short_captions": all(len(text.split()) <= 7 for text in captions),
        "pose_variety": len({beat.pose for beat in script.beats}),
        "human_review_required": ["factual accuracy", "hook", "coherence", "usefulness"],
    }


def render_clip(lock_path, learner, plan, output, seconds=4, fps=24):
    """Render only a confirmed character, using the trained learner. No code exec."""
    if not 1 <= seconds <= 60 or not 12 <= fps <= 60:
        raise ValueError("seconds 1..60; fps 12..60")
    import imageio_ffmpeg

    locked = load_lock(lock_path)
    template_id = locked["template_id"]
    template = get_template(template_id)
    points = learner.predict(plan, frames=seconds * fps)
    quality = motion_quality(points, fps)
    if quality["safe_zone_violations"] or quality["max_joint_step_px"] > 40:
        raise ValueError(f"Motion safety gate failed: {quality}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "540x960", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        try:
            for index, frame in enumerate(points):
                image = Image.new("RGB", (540, 960), template["paper"])
                draw_character(ImageDraw.Draw(image), dict(zip(JOINT_ORDER, frame)),
                               template_id, phase=index / len(points))
                proc.stdin.write(np.asarray(image).tobytes())
            proc.stdin.close()
            error = proc.stderr.read().decode()
            if proc.wait():
                raise RuntimeError(error[-4000:])
        except BaseException:
            proc.kill()
            raise
    write_json(output.with_suffix(".json"), {
        "character": locked, "plan": plan, "quality": quality,
        "seconds": seconds, "silent": True,
        "note": "Joint safety checks do not certify visual aesthetics or anatomy.",
    })
    return output
