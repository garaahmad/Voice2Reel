"""Locked-character video renderer and optional licensed music/audio mux.

Inputs are validated JSON plans, never executable model-produced Python.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from voice2reel.character_templates import FONT_PATH, draw_character, get_template, load_lock, pose_points
from voice2reel.production_contract import ProductionPlan
from voice2reel.scene_objects import draw_objects

def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def render(plan, lock_path, output, voice=None, music=None, music_volume=0.12):
    import imageio_ffmpeg
    plan = ProductionPlan.model_validate(plan)
    lock = load_lock(lock_path)
    template_id = lock["template_id"]
    template = get_template(template_id)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_name(output.stem + ".silent.mp4")
    fps = plan.fps
    width, height = plan.width, plan.height
    if (width, height) != (1080, 1920):
        raise ValueError(f"Instagram Reels export requires 1080x1920, got {width}x{height}")
    scale = width / 540
    writer = imageio_ffmpeg.write_frames(str(silent), (width, height), fps=fps,
                                         codec="libx264", quality=8,
                                         macro_block_size=1)
    writer.send(None)
    from voice2reel.text_layout import draw_text
    font = (ImageFont.truetype(str(FONT_PATH), round(26 * scale))
            if FONT_PATH.is_file() else ImageFont.load_default())
    try:
        for shot in plan.shots:
            frames = round(shot.seconds * fps)
            for index in range(frames):
                phase = index / max(frames, 1)
                im = Image.new("RGB", (width, height), template["paper"])
                draw = ImageDraw.Draw(im)
                # Subtle animated background lines and graphic accents.
                for band in range(4):
                    yy = int((160 + band * 220 + 12 * math.sin(phase * 2 * math.pi + band)) * scale)
                    draw.line((int(24*scale), yy, int(516*scale), yy),
                              fill=template["panel"], width=max(1, round(2*scale)))
                # Motion is smooth in absolute time; template stays locked.
                # Eased, continuous gesture motion with subtle breathing and weight shift.
                motion = ease(phase / 0.22) if phase < 0.22 else (
                    1.0 if phase < 0.78 else ease((1.0 - phase) / 0.22))
                camera_scale = {"wide": 0.88, "medium": 1.0, "close": 1.18}[shot.camera]
                drift = math.sin(2 * math.pi * phase) * 5 * scale
                actor_box = (
                    (270 - 205 * camera_scale) * scale + drift,
                    (165 - (camera_scale - 1) * 140) * scale,
                    (270 + 205 * camera_scale) * scale + drift,
                    (780 + (camera_scale - 1) * 60) * scale,
                )
                if shot.objects:
                    # Reserve right-side space for props; keep the character legible.
                    actor_box = (actor_box[0] - 85 * scale, actor_box[1] + 65 * scale,
                                 actor_box[2] - 85 * scale, actor_box[3] - 35 * scale)
                points = pose_points(shot.gesture, phase, width, height,
                                     actor_box=actor_box)
                # Blend first/last pose toward idle to avoid gesture popping.
                blend = motion
                idle = pose_points("idle", phase, width, height,
                                   actor_box=actor_box)
                points = {key: (idle[key][0] * (1 - blend) + pt[0] * blend,
                                idle[key][1] * (1 - blend) + pt[1] * blend)
                          for key, pt in points.items()}
                # Point at a visible prop using the character's actual arm joints.
                # The character stays in the left visual column to avoid covering props.
                if shot.objects and shot.gesture in ("point", "present", "explain"):
                    visible = [obj for obj in shot.objects
                               if obj.start <= phase * shot.seconds < obj.end]
                    if visible:
                        target = visible[0]
                        shoulder = points["shoulder_right"]
                        wrist = (target.x * width - 45 * scale,
                                 target.y * height)
                        # Keep the reach plausible and avoid dislocating the arm.
                        vx, vy = wrist[0] - shoulder[0], wrist[1] - shoulder[1]
                        distance = math.hypot(vx, vy)
                        reach = min(160 * scale, distance)
                        if distance > 0:
                            wrist = (shoulder[0] + vx / distance * reach,
                                     shoulder[1] + vy / distance * reach)
                            elbow = (shoulder[0] + (wrist[0] - shoulder[0]) * 0.55,
                                     shoulder[1] + (wrist[1] - shoulder[1]) * 0.45
                                     + 18 * scale)
                            amount = ease(min(1.0, max(0.0, phase * 5)))
                            for joint, desired in (("elbow_right", elbow), ("wrist_right", wrist)):
                                old = points[joint]
                                points[joint] = (old[0] * (1 - amount) + desired[0] * amount,
                                                 old[1] * (1 - amount) + desired[1] * amount)
                draw_character(draw, points, template_id, phase=phase)
                draw_objects(im, shot.objects, shot.seconds, phase, template["accent"], template["ink"])
                # Captions are authored and validated; render with repo's Arabic-aware helper.
                draw.rounded_rectangle(tuple(round(v * scale) for v in (25, 790, 515, 905)), radius=round(20 * scale),
                                       fill=template["panel"])
                draw_text(draw, (round(270 * scale), round(840 * scale)), shot.caption, font,
                          direction="rtl" if any("\u0600" <= ch <= "\u06ff" for ch in shot.caption) else "ltr",
                          fill=template["ink"], anchor="mm")
                writer.send(im.tobytes())
    finally:
        writer.close()
    if voice is None and music is None:
        silent.replace(output)
        verify_reels_video(output)
        return output
    if not 0 <= music_volume <= 0.5:
        raise ValueError("music_volume must be between 0 and 0.5")
    inputs = ["-i", str(silent)]
    filters = []
    maps = ["-map", "0:v:0"]
    if voice:
        voice = Path(voice)
        if not voice.is_file():
            raise FileNotFoundError(voice)
        inputs += ["-i", str(voice)]
        filters.append("[1:a]aresample=48000,volume=1.0[v]")
    if music:
        music = Path(music)
        if not music.is_file():
            raise FileNotFoundError(music)
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        idx = 2 if voice else 1
        filters.append(f"[{idx}:a]aresample=48000,volume={music_volume}[m]")
    if voice and music:
        filters.append("[v][m]amix=inputs=2:duration=first:normalize=0[a]")
        maps += ["-map", "[a]"]
    elif voice:
        maps += ["-map", "[v]"]
    else:
        maps += ["-map", "[m]"]
    duration = sum(shot.seconds for shot in plan.shots)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               *inputs, "-filter_complex", ";".join(filters), *maps,
               "-t", str(duration), "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True, timeout=300)
    silent.unlink(missing_ok=True)
    verify_reels_video(output)
    return output

def verify_reels_video(path):
    """Check encoded MP4 dimensions, not notebook preview size."""
    import imageio_ffmpeg
    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner",
                           "-i", str(path)], capture_output=True, text=True)
    import re
    dimensions = re.search(r"Video:.*?\\b(\\d{3,5})x(\\d{3,5})\\b", proc.stderr)
    if not dimensions:
        raise RuntimeError("Could not verify encoded video dimensions")
    actual = tuple(map(int, dimensions.groups()))
    if actual != (1080, 1920):
        raise RuntimeError(f"Incorrect Reels output dimensions: {actual}, expected 1080x1920")
    print(f"✅ Verified Instagram Reels output: {actual[0]}x{actual[1]} (9:16)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--voice")
    p.add_argument("--music")
    p.add_argument("--music-volume", type=float, default=0.12)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    print(render(plan, args.lock, args.output, args.voice, args.music, args.music_volume))

if __name__ == "__main__":
    main()
