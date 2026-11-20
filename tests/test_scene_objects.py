"""Smoke tests for safe timed props and the router demonstration."""
import json
import pytest
from PIL import Image
from voice2reel.production_contract import ProductionPlan
from voice2reel.python_scene_dsl import parse_python_scene
from voice2reel.scene_objects import draw_objects
from scripts.demo_router_scene import build

def test_router_demo():
    plan = build()
    assert (plan.width, plan.height) == (1080, 1920)
    assert len(plan.shots) == 3
    assert all(shot.objects for shot in plan.shots)

def test_safe_dsl_objects():
    src = """scene(title="Router", width=1080, height=1920)
shot(caption="Router", narration="This is a router", seconds=3, gesture="point", objects='[{"kind":"router","start":0,"end":3}]')
"""
    assert parse_python_scene(src).shots[0].objects[0].kind == "router"

def test_reject_unknown_object_and_bad_time():
    with pytest.raises(ValueError):
        ProductionPlan.model_validate({"title":"x","shots":[{"caption":"x","narration":"x","seconds":3,"gesture":"idle","objects":[{"kind":"malicious"}]}]})
    with pytest.raises(ValueError):
        ProductionPlan.model_validate({"title":"x","shots":[{"caption":"x","narration":"x","seconds":3,"gesture":"idle","objects":[{"kind":"router","start":2,"end":1}]}]})

def test_draw_object_changes_pixels():
    image = Image.new("RGB", (540,960), "white")
    original = image.tobytes()
    draw_objects(image, build().shots[0].objects, 5, 0.4)
    assert image.tobytes() != original
