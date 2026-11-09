"""Measured checks for the large synthetic stickman drawing corpus."""

import math
from pathlib import Path

import nbformat
import pytest

from voice2reel.stickman_artist import (
    DefectDetector,
    StickmanArtist,
    draw_frame,
    feature_dict,
    is_professional,
    points_dict,
    rows,
    run_rounds,
    stickman_integrity,
    violations_for,
    vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": 50_000, "validation": 5_000, "test": 5_000}


def test_large_corpus_is_complete_disjoint_and_label_consistent():
    vocab = vocabulary()
    checks = stickman_integrity()
    assert all(checks.values()), [key for key, value in checks.items() if value is False]
    assert vocab["version"] == "stickman-drawing-v2"
    assert len(vocab["keypoint_order"]) == 18
    assert len(vocab["feature_names"]) == 14
    assert len(vocab["defects"]) == 13

    ids = []
    for split, expected in SPLITS.items():
        part = rows(split)
        assert len(part) == expected
        assert all(row["split"] == split for row in part)
        ids.extend(row["id"] for row in part)
        professional = [row for row in part if is_professional(row)]
        needs_fix = [row for row in part if not is_professional(row)]
        assert professional and needs_fix
        assert all(not violations_for(row) for row in professional)
        assert all(violations_for(row) for row in needs_fix)
        assert {row["defect"] for row in needs_fix} == set(vocab["defects"])
        assert all(len(row["points"]) == 18 for row in part)
        assert all(len(row["features"]) == 14 for row in part)
        assert all(len(row["style"]) == 4 for row in part)
    assert len(ids) == 60_000
    assert len(set(ids)) == 60_000


def test_every_named_defect_is_reconstructed_from_features():
    part = rows("test")
    for defect in vocabulary()["defects"]:
        row = next(item for item in part if item["defect"] == defect)
        assert violations_for(row) == [defect]


def test_defect_detector_learns_thresholds_not_hardcoded_labels():
    train = rows("train")[:4_000]
    test = rows("test")[:1_000]
    small = DefectDetector(features=vocabulary()["feature_names"][:5]).fit(train)
    full = DefectDetector().fit(train)
    assert small.rules and full.rules
    assert all(rule["train_false_alarms"] == 0 for rule in full.rules.values())
    assert full.accuracy(test) > small.accuracy(test)
    assert full.accuracy(test) > 0.95


def test_stickman_artist_predicts_eighteen_finite_joints():
    artist = StickmanArtist().fit(rows("train")[:2_000])
    sample = rows("test")[:200]
    professional = next(row for row in sample if is_professional(row))
    predicted = artist.predict(professional)
    assert len(predicted) == 18
    assert all(len(point) == 2 for point in predicted)
    assert all(math.isfinite(value) for point in predicted for value in point)
    assert artist.mean_joint_error(sample) < 30


def test_quality_loop_is_measured_and_improves_with_more_data():
    report = run_rounds(max_rounds=3)
    scores = [item["score"] for item in report["rounds"]]
    assert len(scores) == 3
    assert scores[0] < scores[1] < scores[2]
    assert report["rounds"][0]["train_rows"] < report["rounds"][-1]["train_rows"]
    assert report["rounds"][0]["features"] < report["rounds"][-1]["features"]
    assert report["final_test_on_best_round"]["score"] > 0.90
    assert all(report["integrity"].values())
    committed = (ROOT / "examples" / "stickman_drawing" / "artist_rounds.json").read_text(encoding="utf-8")
    assert report == __import__("json").loads(committed)


def test_single_stickman_notebook_covers_data_training_and_video():
    path = ROOT / "notebooks" / "06_stickman_generator.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "60,000" in source
    assert "scripts/check_data.py" in source
    assert "torch" in source and "MODEL" in source
    assert "stickman_demo.mp4" in source
    assert "stickman_pipeline_report.json" in source
    assert "05_quality_loop.ipynb" in source
    assert "no expert review" in source


def test_vector_renderer_draws_connected_neck_and_png(tmp_path):
    pytest.importorskip("PIL.Image")
    pytest.importorskip("arabic_reshaper")
    part = rows("validation")
    professional = next(row for row in part if is_professional(row))
    points = points_dict(professional)
    assert math.dist(points["head_bottom"], points["neck"]) < 20
    assert math.dist(points["neck"], points["shoulder_left"]) < 40
    assert feature_dict(professional)["joint_gap"] < 5

    output = tmp_path / "frame.png"
    image = draw_frame(professional, output=output)
    assert image.size == (540, 960)
    assert output.is_file() and output.stat().st_size > 1_000

    broken = next(row for row in part if row["defect"] == "disconnected_joint")
    defect_output = tmp_path / "broken.png"
    draw_frame(broken, output=defect_output)
    assert defect_output.is_file()
