import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=90)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Beat(StrictModel):
    narration: Text
    caption: ShortText
    pose: Literal["IDLE", "SHOCK", "CELEB", "SUSPECT", "LEAN", "TYPE_A", "LOOK_L", "LOOK_R"]
    camera: Literal["far", "medium", "close", "extreme", "left", "right"]


class Script(StrictModel):
    schema_version: Literal[1]
    title: ShortText
    language: Literal["ar"]
    target_duration_seconds: Annotated[int, Field(ge=10, le=60)]
    beats: Annotated[list[Beat], Field(min_length=3, max_length=8)]

    @model_validator(mode="after")
    def reasonable_length(self):
        words = sum(len(beat.narration.split()) for beat in self.beats)
        # Broad guard against empty/rambling output, not a speech-duration guarantee.
        if not 0.6 * self.target_duration_seconds <= words <= 3.5 * self.target_duration_seconds:
            raise ValueError("Narration word count is implausible for the target duration")
        return self

    def narration(self):
        return " ".join(beat.narration for beat in self.beats)


def parse_script(raw, expected_duration=None):
    value = json.loads(raw)
    script = Script.model_validate(value)
    if expected_duration is not None and script.target_duration_seconds != expected_duration:
        raise ValueError("Generated target duration differs from requested duration")
    return script


SYSTEM_PROMPT = """أنت كاتب فيديوهات قصيرة بالعربية. حوّل الفكرة إلى سكربت واضح ومفيد، بجمل سهلة النطق.
اكتب JSON فقط بلا Markdown. لا تخترع إحصاءات أو وعودا أو مصادر. التزم بالمدة المطلوبة.
قسّم النص إلى ثلاثة حتى ثمانية مقاطع. النص المنطوق في narration، والعنوان القصير في caption.
هذه مقاطع سردية، والتوقيت الفعلي يضاف بعد توليد الصوت. اختر pose وcamera من القيم المسموحة.
المخطط المطلوب:
""" + json.dumps(Script.model_json_schema(), ensure_ascii=False, separators=(",", ":"))


def messages_for(idea, duration=30):
    if not isinstance(idea, str) or not idea.strip() or len(idea) > 2000:
        raise ValueError("Idea must contain 1–2000 characters")
    if type(duration) is not int or not 10 <= duration <= 60:
        raise ValueError("Duration must be an integer from 10 to 60 seconds")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"الفكرة: {idea.strip()}\nالمدة المستهدفة: {duration} ثانية"},
    ]
