"""Safe restricted Python scene language emitted by Qwen.

No exec(), eval(), imports, attributes, loops, functions, or file access.
Only one top-level scene() call and 1..20 shot() calls are accepted.
Character identity is supplied exclusively by the external lock.
"""
import ast
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
            data["shots"].append(values)
    return ProductionPlan.model_validate(data)

PYTHON_SYSTEM_PROMPT = """Generate ONLY a Python scene DSL script, never raw Python.
Allowed statements: scene(title="...", fps=30, width=540, height=960)
and shot(caption="...", narration="...", seconds=3.0, gesture="wave",
camera="medium", transition="fade").
Exactly one scene() then 1..20 shot() calls. No imports, loops, expressions,
attributes, function definitions, variables, files or arbitrary Python.
Gestures: idle, present, point, wave, think, explain, celebrate, step.
Camera: wide, medium, close. Transition: cut, fade.
Total duration <=60 seconds. The character is locked externally; never draw
or choose another character. Audio is generated separately from narration.
Produce 3-6 varied shots with meaningful captions and natural narration."""
