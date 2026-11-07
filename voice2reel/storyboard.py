"""Five-question brief and a director plan for realistic video generation."""

import math
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from voice2reel.render import validate_timeline
from voice2reel.scriptwriter.schema import StrictModel

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]
Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)]
English = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=1, max_length=1200,
    pattern=r"^[^\u0600-\u06ff]*[A-Za-z][^\u0600-\u06ff]*$",
)]
QUESTIONS = (
    ("audience", "لمن الفيديو؟"),
    ("target_duration_seconds", "كم ثانية تريد؟ من 10 إلى 40"),
    ("language", "لغة التعليق الصوتي: ar للعربية أو en للإنجليزية؟"),
    ("setting", "ما المكان والشخصيات والجو البصري؟"),
    ("message", "ما الرسالة الأساسية أو الحقائق التي يجب الالتزام بها؟"),
)


class Brief(StrictModel):
    idea: Text
    audience: Short
    target_duration_seconds: Annotated[int, Field(ge=10, le=40)] = 20
    language: Literal["ar", "en"] = "ar"
    setting: Short
    message: Text
    reference_video: str | None = None

    @property
    def shot_count(self):
        return max(3, math.ceil(self.target_duration_seconds / 5))


class Shot(StrictModel):
    purpose: Short
    visual_prompt: English
    opening_action: English
    closing_action: English


class Plan(StrictModel):
    title: Annotated[str, StringConstraints(min_length=1, max_length=90)]
    continuity: English
    shots: Annotated[list[Shot], Field(min_length=3, max_length=8)]


def video_prompt(plan, shot):
    return (
        f"Realistic live-action footage. {plan.continuity} {shot.visual_prompt} "
        f"At the beginning: {shot.opening_action} "
        f"By the end: {shot.closing_action} "
        "One continuous shot, natural human movement, consistent lighting, no cuts, no visible text."
    )


def build_timeline(plan, script, metadata, fps=24, max_clip_seconds=5.0):
    """Split long speech segments into bounded clips; never silently freeze/loop a short clip."""
    if fps != 24 or not 1 <= max_clip_seconds <= 6:
        raise ValueError("Use 24 fps and clips of 1–6 seconds")
    if len(plan.shots) != len(script.beats):
        raise ValueError("Director shots and script beats must correspond one-to-one")
    segments = validate_timeline(script, metadata)
    clips = []
    cursor = 0
    for index, (shot, segment) in enumerate(zip(plan.shots, segments)):
        end_frame = round(segment["end"] * fps)
        count = end_frame - cursor
        if count < 1:
            raise ValueError("Speech segment shorter than one video frame")
        parts = math.ceil(count / (max_clip_seconds * fps))
        for part in range(parts):
            frames = count // parts + (part < count % parts)
            # LTX temporal compression requires 8n+1 generated frames.
            generated_frames = 8 * math.ceil((frames - 1) / 8) + 1
            clips.append({
                "index": len(clips), "shot_index": index, "part": part,
                "start_frame": cursor, "end_frame": cursor + frames,
                "num_frames": max(9, generated_frames),
                "prompt": video_prompt(plan, shot),
                "keyframes": [
                    {"frame": cursor, "description": shot.opening_action},
                    {"frame": cursor + frames - 1, "description": shot.closing_action},
                ],
            })
            cursor += frames
    return {"fps": fps, "total_frames": cursor, "clips": clips}
