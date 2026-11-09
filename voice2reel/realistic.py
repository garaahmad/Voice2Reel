"""Experimental LTX-Video generation, resumable clips, and synchronized MP4 assembly."""

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf
import numpy as np

from voice2reel.common import digest_file, write_json
from voice2reel.render import find_font, wrap_text
from voice2reel.scriptwriter.schema import Script
from voice2reel.storyboard import Brief, Plan, build_timeline

VIDEO_MODEL = "Lightricks/LTX-Video-0.9.5"
VIDEO_REVISION = "e58e28c39631af4d1468ee57a853764e11c1d37e"
NEGATIVE = "animation, cartoon, illustration, distorted anatomy, blurry, jitter, watermark, text, subtitles"


def command(*args):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-2000:]}")
    return result.stdout


def probe(path):
    return json.loads(command("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path))


def extract_frame(video, destination, width, height, last=False):
    args = ["ffmpeg", "-v", "error", "-y"]
    if last:
        args += ["-sseof", "-0.05"]
    args += ["-i", video, "-frames:v", "1", "-vf",
             f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}", destination]
    command(*args)
    if not Path(destination).is_file():
        raise ValueError("Reference video has no decodable frame")


def validate_settings(width, height, steps):
    if (type(width) is not int or type(height) is not int or width % 32 or height % 32
            or not 256 <= width <= 1024 or not 256 <= height <= 1024):
        raise ValueError("Width/height must be multiples of 32 between 256 and 1024")
    if type(steps) is not int or not 20 <= steps <= 60:
        raise ValueError("Use 20–60 denoising steps for this non-distilled checkpoint")


def encode_prompts(prompts):
    """Quantize T5 and release it before loading the video transformer (important on T4)."""
    import torch
    from diffusers import LTXConditionPipeline
    from transformers import BitsAndBytesConfig, T5EncoderModel

    encoder = T5EncoderModel.from_pretrained(
        VIDEO_MODEL, revision=VIDEO_REVISION, subfolder="text_encoder",
        torch_dtype=torch.float32, device_map={"": "cuda"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32,
        ),
    )
    pipe = LTXConditionPipeline.from_pretrained(
        VIDEO_MODEL, revision=VIDEO_REVISION, text_encoder=encoder, transformer=None, vae=None,
        torch_dtype=torch.float32,
    )
    encoded = []
    with torch.inference_mode():
        for prompt in prompts:
            if len(pipe.tokenizer(prompt).input_ids) > 256:
                raise ValueError("Visual prompt exceeds 256 T5 tokens; shorten plan.json before generating video")
            values = pipe.encode_prompt(
                prompt, negative_prompt=NEGATIVE, do_classifier_free_guidance=True,
                device=torch.device("cuda"), max_sequence_length=256,
            )
            if any(not torch.isfinite(value).all() for value in values):
                raise ValueError("Non-finite text embeddings; aborting instead of generating invalid video")
            encoded.append(tuple(value.cpu() for value in values))
    del pipe, encoder
    gc.collect()
    torch.cuda.empty_cache()
    return encoded


def caption_png(text, destination, width, height, language):
    from PIL import Image, ImageDraw, ImageFont

    from voice2reel.text_layout import draw_text

    direction = "rtl" if language == "ar" else "ltr"
    size = max(14, width // 20)
    while True:
        font = ImageFont.truetype(str(find_font()), size)
        lines = wrap_text(text, font, width * .88, direction)
        if len(lines) <= 3 or size <= 10:
            break
        size -= 1
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    line_height = size * 1.6
    y = height * .88 - len(lines) * line_height
    draw.rounded_rectangle((width * .03, y - 12, width * .97, height * .9), radius=8, fill=(0, 0, 0, 175))
    for line in lines:
        draw_text(draw, (width / 2, y), line, font, direction, anchor="mt", fill="white")
        y += line_height
    image.save(destination)


def assemble(project, timeline, script, audio, output, width, height):
    """Real ffmpeg encode; exact frame bounds, real audio duration, no shell interpolation."""
    project, output = Path(project), Path(output)
    info = sf.info(audio)
    duration = info.frames / info.samplerate
    if abs(duration - timeline["total_frames"] / timeline["fps"]) > 1 / timeline["fps"]:
        raise ValueError("Audio duration does not match the validated timeline")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        work = Path(temporary)
        entries = []
        for clip in timeline["clips"]:
            source = project / "clips" / f"{clip['index']:03d}.mp4"
            required = clip["end_frame"] - clip["start_frame"]
            stream = next(s for s in probe(source)["streams"] if s["codec_type"] == "video")
            if float(stream["duration"]) + .001 < required / timeline["fps"]:
                raise ValueError(f"Generated clip too short: {source}")
            overlay = work / f"caption-{clip['index']}.png"
            caption_png(script.beats[clip["shot_index"]].caption, overlay, width, height, script.language)
            trimmed = work / f"part-{clip['index']:03d}.mp4"
            command("ffmpeg", "-v", "error", "-y", "-i", source, "-i", overlay,
                    "-filter_complex",
                    f"[0:v]fps={timeline['fps']},scale={width}:{height},setsar=1[v];[v][1:v]overlay=0:0",
                    "-frames:v", required, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-threads", "2", trimmed)
            entries.append(f"file '{trimmed.name}'")
        listing = work / "concat.txt"
        listing.write_text("\n".join(entries) + "\n", encoding="utf-8")
        final = work / "final.mp4"
        command("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "1", "-i", listing,
                "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "128k", "-t", duration, "-movflags", "+faststart", final)
        command("ffmpeg", "-v", "error", "-i", final, "-f", "null", "-")
        final.replace(output)
    return output


def generate(project, width=288, height=512, steps=40, seed=42, max_clips=None):
    import torch

    validate_settings(width, height, steps)
    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Realistic video needs a CUDA GPU. Select GPU in Colab; no weights were downloaded.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Install ffmpeg before generating video")
    project = Path(project)
    status_path = project / "generation_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") != "ready_for_speech":
        raise ValueError("Director/script generation is incomplete; do not reuse files from an older run")
    brief = Brief.model_validate_json((project / "brief.json").read_text())
    plan = Plan.model_validate_json((project / "plan.json").read_text())
    script = Script.model_validate_json((project / "script.json").read_text())
    metadata = json.loads((project / "narration.json").read_text())
    timeline = build_timeline(plan, script, metadata)
    audio = project / "narration.wav"
    audio_info = sf.info(audio)
    if abs(audio_info.frames / audio_info.samplerate - metadata["duration"]) > .001:
        raise ValueError("WAV does not match narration metadata")
    write_json(project / "timeline.json", timeline)
    clip_dir = project / "clips"
    clip_dir.mkdir(exist_ok=True)
    reference = None
    if brief.reference_video:
        if not Path(brief.reference_video).is_file():
            raise ValueError("Reference must be an existing local video file")
        reference = clip_dir / "reference.png"
        extract_frame(brief.reference_video, reference, width, height)
    settings = {"model": VIDEO_MODEL, "revision": VIDEO_REVISION, "width": width, "height": height,
                "steps": steps, "seed": seed, "fps": timeline["fps"], "negative_prompt": NEGATIVE,
                "reference_sha256": digest_file(reference) if reference else None}
    signature = hashlib.sha256(json.dumps({"settings": settings, "timeline": timeline}, sort_keys=True).encode()).hexdigest()
    manifest_path = project / "video_generation.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = previous if previous.get("signature") == signature else {
        "signature": signature, "settings": settings, "clips": {}, "synthetic_video": True,
    }
    pending = []
    selected = timeline["clips"][:max_clips]
    for clip in selected:
        index = str(clip["index"])
        path = clip_dir / f"{clip['index']:03d}.mp4"
        dependency_changed = clip["part"] > 0 and pending and pending[-1]["index"] == clip["index"] - 1
        if dependency_changed or not (
            path.exists() and manifest["clips"].get(index, {}).get("sha256") == digest_file(path)
        ):
            pending.append(clip)
    if pending:
        from diffusers import LTXConditionPipeline
        from diffusers.utils import export_to_video
        from PIL import Image

        embeddings = encode_prompts([clip["prompt"] for clip in pending])
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        pipe = LTXConditionPipeline.from_pretrained(
            VIDEO_MODEL, revision=VIDEO_REVISION, text_encoder=None, torch_dtype=dtype,
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        for clip, values in zip(pending, embeddings):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            path = clip_dir / f"{clip['index']:03d}.mp4"
            image_path = reference if clip["index"] == 0 else None
            # Continue a long narration beat using the previous clip's final frame.
            if clip["part"] > 0:
                image_path = clip_dir / "continuation.png"
                extract_frame(clip_dir / f"{clip['index'] - 1:03d}.mp4", image_path, width, height, last=True)
            prompt, mask, negative, negative_mask = values
            frames = pipe(
                prompt_embeds=prompt.to(device="cuda", dtype=dtype),
                negative_prompt_embeds=negative.to(device="cuda", dtype=dtype),
                prompt_attention_mask=mask.to("cuda"), negative_prompt_attention_mask=negative_mask.to("cuda"),
                image=Image.open(image_path).convert("RGB") if image_path else None,
                width=width, height=height, num_frames=clip["num_frames"], frame_rate=24,
                num_inference_steps=steps, guidance_scale=3.0,
                decode_timestep=.05, decode_noise_scale=.025, image_cond_noise_scale=.025,
                generator=torch.Generator(device="cpu").manual_seed(seed + clip["index"]),
                output_type="np",
            ).frames[0]
            if len(frames) != clip["num_frames"] or not np.isfinite(frames).all():
                raise ValueError("Video model returned invalid frames")
            temporary = path.with_suffix(".tmp.mp4")
            export_to_video(frames, str(temporary), fps=24, macro_block_size=1)
            command("ffmpeg", "-v", "error", "-i", temporary, "-f", "null", "-")
            temporary.replace(path)
            manifest["clips"][str(clip["index"])] = {
                "sha256": digest_file(path), "seed": seed + clip["index"],
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "gpu": torch.cuda.get_device_name(),
            }
            write_json(manifest_path, manifest)
            print(f"Generated clip {clip['index'] + 1}/{len(timeline['clips'])}", flush=True)
        del pipe, embeddings
        gc.collect()
        torch.cuda.empty_cache()
    if len(selected) < len(timeline["clips"]):
        print("Preview clips saved. Rerun without --max-clips to resume and assemble the full video.")
        return None
    output = assemble(project, timeline, script, audio, project / "video.mp4", width, height)
    manifest["output_sha256"] = digest_file(output)
    manifest["duration_seconds"] = probe(output)["format"]["duration"]
    write_json(manifest_path, manifest)
    return output


def main():
    p = argparse.ArgumentParser(description="Generate realistic LTX scenes after the director and narration stages")
    p.add_argument("--project", required=True)
    p.add_argument("--width", type=int, default=288)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-clips", type=int, help="Generate only the first N clips for a GPU smoke test")
    args = p.parse_args()
    result = generate(args.project, args.width, args.height, args.steps, args.seed, args.max_clips)
    if result:
        print(f"Saved {result}")


if __name__ == "__main__":
    main()
