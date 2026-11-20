import json

import numpy as np
import pytest

from voice2reel.character_templates import (
    TEMPLATE_IDS, load_lock, lock_template, render_preview, template_diversity,
)
from voice2reel.motion_curriculum import (
    FamilyMemoryBaseline, MotionLearner, build_curriculum, motion_quality,
)


def test_catalog_previews_and_explicit_lock(tmp_path):
    assert len(TEMPLATE_IDS) == 12
    assert template_diversity()["unique_render_sha256"] == 12
    for name in TEMPLATE_IDS:
        assert render_preview(name).size == (360, 520)
    with pytest.raises(ValueError):
        render_preview("not-a-character")
    path = tmp_path / "lock.json"
    with pytest.raises(FileNotFoundError):
        load_lock(path)
    lock_template("robot", path)
    assert load_lock(path)["template_id"] == "robot"
    record = json.loads(path.read_text())
    record["catalog_sha256"] = "invalid"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changed"):
        load_lock(path)


def test_family_splits_and_training_generalization():
    splits = build_curriculum(train_per_family=20, held_out_per_family=5)
    train_families = {r["family"] for r in splits["train"]}
    for name in ("validation", "test", "extrapolation"):
        assert not train_families & {r["family"] for r in splits[name]}
    learner = MotionLearner(epochs=35, batch_size=128).fit(
        splits["train"], splits["validation"])
    nearest = FamilyMemoryBaseline(splits["train"])
    assert learner.mean_error(splits["test"]) < nearest.mean_error(splits["test"])
    points = learner.predict(splits["test"][0], frames=48)
    assert points.shape == (48, 18, 2)
    assert np.isfinite(points).all()
    assert motion_quality(points)["safe_zone_violations"] == 0


def test_render_requires_approval(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from voice2reel.character_studio import render_clip
    with pytest.raises(FileNotFoundError):
        render_clip(tmp_path / "missing.json", None, {}, tmp_path / "clip.mp4")
