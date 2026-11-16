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
                points = pose_points(shot.gesture, phase, width, height,
                                     actor_box=actor_box)
                # Blend first/last pose toward idle to avoid gesture popping.
                blend = motion
                idle = pose_points("idle", phase, width, height,
                                   actor_box=actor_box)
                points = {key: (idle[key][0] * (1 - blend) + pt[0] * blend,
                                idle[key][1] * (1 - blend) + pt[1] * blend)
                          for key, pt in points.items()}
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
    return output

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
