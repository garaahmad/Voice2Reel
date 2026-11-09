"""Compact, validated storyboard for the fast 2D presenter engine."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from voice2reel.arabic_text import resolve_scene_texts
from voice2reel.scriptwriter.schema import Script, ShortText, StrictModel, Text

# Display narration may contain registered technical Latin terms or full tashkeel.
# ``spoken`` is the Arabic-only text actually sent to the speech engine.
SpokenText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=600)]


class Visual(StrictModel):
    kind: Literal["none", "list", "flow", "compare", "code"] = "none"
    items: Annotated[list[ShortText], Field(max_length=4)] = Field(default_factory=list)
    code: Annotated[str, Field(max_length=600)] = ""
    focus: Annotated[str, Field(max_length=32)] = ""
    status: Literal["neutral", "warning", "success"] = "neutral"


class Scene(StrictModel):
    narration: Text
    spoken: SpokenText = ""
    caption: ShortText
    gesture: Literal["present", "point", "think", "shrug", "thumbs_up", "wave"] = "present"
    emotion: Literal["neutral", "happy", "surprised", "concerned", "thinking"] = "neutral"
    camera: Literal["wide", "medium", "close"] = "wide"
    visual: Visual = Field(default_factory=Visual)


class PresenterScript(StrictModel):
    title: ShortText
    language: Literal["ar", "en"] = "ar"
    scenes: Annotated[list[Scene], Field(min_length=3, max_length=6)]

    @model_validator(mode="after")
    def short_reel(self):
        words = sum(len(scene.narration.split()) for scene in self.scenes)
        if not 6 <= words <= 180:
            raise ValueError("Use 6–180 spoken words for a short reel")
        return self


def parse_script(value):
    # Existing reviewed script.json files remain useful without another AI pass.
    if "beats" in value:
        value = Script.model_validate(value).model_dump()
        value = {
            "title": value["title"],
            "language": value["language"],
            "scenes": [
                {
                    "narration": b["narration"],
                    "caption": b["caption"],
                    "gesture": {"SUSPECT": "think", "SHOCK": "shrug", "CELEB": "thumbs_up"}.get(
                        b["pose"], "point"
                    ),
                    "emotion": {"SHOCK": "surprised", "CELEB": "happy", "SUSPECT": "thinking"}.get(
                        b["pose"], "neutral"
                    ),
                }
                for b in value["beats"]
            ],
        }
    # Compact/user scripts may contain registered technical terms in display text.
    # Fill the TTS-safe Arabic reading and reject unregistered code identifiers.
    return PresenterScript.model_validate(resolve_scene_texts(value))


class ReelRequest(StrictModel):
    idea: Annotated[str, Field(min_length=1, max_length=2000)]
    facts: Annotated[str, Field(max_length=2000)] = ""
    audience: Annotated[str, Field(min_length=1, max_length=400)] = "General audience"
    language: Literal["ar", "en"] = "ar"
    duration: Annotated[int, Field(ge=10, le=60)] = 20
    visual_focus: Annotated[str, Field(max_length=600)] = "Clear cartoon presenter with explanatory labels"

    @model_validator(mode="after")
    def meaningful_input(self):
        if not self.idea.strip() or not self.audience.strip():
            raise ValueError("Idea and audience must not be blank")
        return self
