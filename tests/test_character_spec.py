"""CPU-only regression tests for safe Qwen -> CharacterSpec -> rig output."""
import json
from xml.etree import ElementTree as ET

import pytest

from voice2reel.character_spec import DEFAULT_SPEC, PALETTE_IDS, parse_spec, save_spec, load_spec
from voice2reel.character_spec_renderer import apply_spec
from voice2reel.generate_character_spec import _extract_json
from voice2reel.mascot_animator import MascotAnimator
from voice2reel.mascot_rig import load_rig
from scripts.build_character_spec_dataset import create_rows
from voice2reel.train_character_spec import load as load_training_rows


def variant(**changes):
    data = DEFAULT_SPEC.model_dump()
    data.update(changes)
    return parse_spec(data)


def test_strict_contract_rejects_unsafe_or_invalid_styles():
    with pytest.raises(Exception):
        variant(arm_length=9999)
    with pytest.raises(Exception):
        variant(palette="<script>alert(1)</script>")
    with pytest.raises(Exception):
        parse_spec('{"palette":"forest"}')
    with pytest.raises(Exception):
        variant(version=2)
    with pytest.raises(Exception):
        parse_spec("<svg></svg>")


def test_json_roundtrip_and_generation_parser(tmp_path):
    spec = variant(palette="forest", expression="curious")
    filename = save_spec(spec, tmp_path / "model.json")
    assert load_spec(filename) == spec
    assert json.loads(_extract_json("Here is the answer:\n" + spec.model_dump_json())) == spec.model_dump()
    with pytest.raises(ValueError):
        _extract_json("not a JSON object")


def test_all_styles_keep_14_joint_groups_and_constant_pivots():
    rig, _, original_svg = load_rig()
    for palette in PALETTE_IDS:
        for expression in ("happy", "surprised", "curious", "thoughtful", "confident"):
            spec = variant(palette=palette, expression=expression, eyes="wide",
                           brows="focused", cheeks="none", accent="simple")
            restyled = apply_spec(original_svg, spec)
            root = ET.fromstring(restyled)
            ids = {e.attrib.get("id"): e for e in root.iter() if e.attrib.get("id")}
            for name, config in rig["parts"].items():
                assert ids[name].get("transform") == (
                    f"rotate(0 {config['pivot'][0]} {config['pivot'][1]})"
                )
            assert len(rig["parts"]) == 14
            assert "<script" not in restyled
            assert restyled != original_svg


def test_rendered_cartoon_face_and_limb_motion():
    spec = variant(palette="plum", expression="surprised", eyes="wide")
    animator = MascotAnimator(spec=spec)
    for pose in ("idle", "present", "point", "wave", "think",
                 "explain", "celebrate", "step"):
        svg = animator.frame_svg(pose, 1.25, 3)
        ET.fromstring(svg)
        assert "@@" not in svg
    image = animator.frame_image("wave", 1.25, 3, width=256)
    assert image.size == (256, 384) and image.getbbox() is not None


def test_dataset_validates_and_train_validation_prompts_do_not_overlap(tmp_path):
    rows = create_rows(140, seed=42)
    assert len(rows) == 140
    assert len({r["messages"][1]["content"] for r in rows}) == 140
    for row in rows:
        assert parse_spec(row["messages"][2]["content"]).version == 1
        assert "<svg" not in row["messages"][2]["content"]
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "validation.jsonl"
    for filename, batch in ((train, rows[:125]), (valid, rows[125:])):
        filename.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in batch),
                            encoding="utf-8")
    assert len(load_training_rows(train)) == 125
    assert len(load_training_rows(valid)) == 15
