"""Safe restricted Python scene language emitted by Qwen.

No exec(), eval(), imports, attributes, loops, functions, or file access.
Only one top-level scene() call and 1..20 shot() calls are accepted.
Character identity is supplied exclusively by the external lock.
"""
import ast
import json
from voice2reel.production_contract import ProductionPlan

ALLOWED = {"scene", "shot"}

def parse_python_scene(source):
    if not isinstance(source, str) or len(source) > 12000:
        raise ValueError("Python scene source exceeds limits")
    tree = ast.parse(source, mode="exec")
    if not 2 <= len(tree.body) <= 21:
        raise ValueError("Expected one scene() and 1..20 shot() calls")
    data = {"version": 1, "shots": []}
    for index, statement in enumerate(tree.body):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError("Only function-call statements are supported")
        call = statement.value
        name = call.func.id if isinstance(call.func, ast.Name) else None
        if name != ("scene" if index == 0 else "shot") or call.args:
            raise ValueError("Only scene(...) followed by shot(...) allowed")
        values = {}
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in values:
                raise ValueError("Unpacking or duplicate keyword forbidden")
            if not isinstance(keyword.value, ast.Constant) or type(keyword.value.value) not in (str, int, float):
                raise ValueError("Only literal strings and numbers allowed")
            values[keyword.arg] = keyword.value.value
        if index == 0:
            data.update(values)
        else:
            if "objects" in values:
                if not isinstance(values["objects"], str) or len(values["objects"]) > 2000:
                    raise ValueError("objects must be a short JSON string")
                values["objects"] = json.loads(values["objects"])
            data["shots"].append(values)
    return ProductionPlan.model_validate(data)

PYTHON_SYSTEM_PROMPT = """Generate ONLY a Python scene DSL script, never raw Python.
Allowed statements: scene(title="...", fps=30, width=1080, height=1920)
and shot(caption="...", narration="...", seconds=3.0, gesture="wave",
camera="medium", transition="fade", objects='[{"kind":"router","animation":"pop_in","x":0.72,"y":0.42}]').
Objects are an optional JSON-encoded STRING literal, maximum 6 per shot.
Kinds: router, ethernet_cable, wifi_signal, laptop, phone, server, cloud.
Animations: pop_in, draw, pulse, slide_in, static.
Coordinates x/y are normalized 0..1; start/end are seconds within the shot.
Exactly one scene() then 1..20 shot() calls. No imports, loops, expressions,
attributes, function definitions, variables, files or arbitrary Python.
Gestures: idle, present, point, wave, think, explain, celebrate, step.
Camera: wide, medium, close. Transition: cut, fade.
Total duration <=60 seconds. The character is locked externally; never draw
or choose another character. Audio is generated separately from narration.
Produce 3-6 varied shots with meaningful captions and natural narration."""
