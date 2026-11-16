"""Strict Qwen-directed production plan. Qwen never edits character artwork.

This is a data-only contract: never exec/eval generated Python on the host.
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from voice2reel.character_templates import TEMPLATE_IDS, POSES

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class Shot(Strict):
    caption: str = Field(min_length=1, max_length=100)
    narration: str = Field(min_length=1, max_length=600)
    seconds: float = Field(ge=1, le=15, strict=False)
    gesture: Literal["idle", "present", "point", "wave", "think", "explain", "celebrate", "step"]
    camera: Literal["wide", "medium", "close"] = "medium"
    transition: Literal["cut", "fade"] = "fade"

class ProductionPlan(Strict):
    version: Literal[1] = 1
    title: str = Field(min_length=1, max_length=100)
    shots: list[Shot] = Field(min_length=1, max_length=20)
    fps: Literal[24, 30] = 30
    width: Literal[540, 1080] = 540
    height: Literal[960, 1920] = 960

    @model_validator(mode="after")
    def limits(self):
        if self.width * 16 != self.height * 9:
            raise ValueError("Video aspect ratio must be 9:16")
        if sum(s.seconds for s in self.shots) > 60:
            raise ValueError("Total duration must be <= 60 seconds")
        return self

def validate_character(plan, locked_character_id):
    if locked_character_id not in TEMPLATE_IDS:
        raise ValueError("Unknown character")
    if not isinstance(plan, ProductionPlan):
        plan = ProductionPlan.model_validate(plan)
    return plan

def qwen_instruction(idea, character_id):
    if character_id not in TEMPLATE_IDS:
        raise ValueError("Unknown character")
    if not idea or len(idea) > 2000:
        raise ValueError("Idea must contain 1..2000 characters")
    return (
        "You are a short-video director. Return ONLY a JSON object conforming "
        "to the ProductionPlan schema below. Do not output Python, imports, "
        "file paths, or arbitrary commands. The renderer owns all Python code. "
        "The user's character is locked to " + character_id + ". "
        "Never change its shape, colors, face or accessories. "
        "Use supported gestures and camera shots. Captions must be concise. "
        "Prefer 3-6 shots and total duration 10-60 seconds. "
        "JSON schema: " + __import__("json").dumps(
            ProductionPlan.model_json_schema(), ensure_ascii=False
        ) + "\nIdea: " + idea
    )
