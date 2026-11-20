import pytest
from voice2reel.python_scene_dsl import parse_python_scene
from voice2reel.production_contract import qwen_instruction

VALID = '''scene(title="Hello", fps=30, width=540, height=960)
shot(caption="Hi", narration="A simple introduction to the subject.", seconds=3.0, gesture="wave", camera="medium", transition="fade")
'''

def test_safe_python_scene():
    plan = parse_python_scene(VALID)
    assert plan.shots[0].gesture == "wave"
    assert plan.fps == 30

@pytest.mark.parametrize("source", [
    "import os\n" + VALID,
    "scene(title='A')\n__import__('os').system('id')",
    "scene(title='A')\nshot(caption='X', narration='Y', seconds=2, gesture='invalid')",
    "scene(title='A')\nshot(caption='X', narration='Y', seconds=2, gesture='wave', extra='oops')",
    "scene(title='A')\nshot(caption='X', narration='Y', seconds=2, gesture='wave', seconds=3)",
])
def test_reject_unsafe_or_invalid(source):
    with pytest.raises((ValueError, SyntaxError)):
        parse_python_scene(source)

def test_lock_character_prompt():
    assert "minimal" in qwen_instruction("Explain testing", "minimal")
    with pytest.raises(ValueError):
        qwen_instruction("Explain testing", "nonexistent")
