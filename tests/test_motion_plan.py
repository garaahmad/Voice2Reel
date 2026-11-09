from voice2reel.motion_plan import (
    ComfortClassifier, camera_step, clamp_camera, comfort_score, fit_actor,
    limits, overlaps, safe_actor_box, scene_plan, validate_plan, vocabulary,
)


def test_vocabulary_and_limits_are_authored_and_bounded():
    vocab = vocabulary()
    assert vocab["version"] == "motion-vocabulary-v2"
    assert limits()["max_camera_step_per_scene"] == 1
    assert "smooth" in vocab["easings"]
    assert camera_step("wide", "close") == 2
    assert clamp_camera("wide", "close") == "medium"


def test_comfort_rules_detect_each_violation():
    good = {
        "camera_step": 1, "gesture_switches_per_second": 0.25,
        "jitter_amplitude": 1.2, "scene_seconds": 4.0, "flash_alpha": 0.05,
        "overlap_count": 0, "subtitle_words": 4, "easing": "smooth",
    }
    assert validate_plan({"features": good}) == []
    assert comfort_score({"features": good}) == 1.0
    bad = dict(good, camera_step=3, jitter_amplitude=5.0, easing="snap")
    assert set(validate_plan({"features": bad})) == {"camera_jump", "jitter", "abrupt_easing"}
    assert comfort_score({"features": bad}) < 0.75


def test_safe_boxes_and_actor_fitting():
    box = safe_actor_box("wide", True)
    assert box[0] >= 24 and box[2] <= 330
    assert not overlaps((0, 0, 10, 10), (20, 0, 30, 10))
    scale, x, y = fit_actor((400, 800), (0, 0), (24, 250, 271, 836))
    assert 0 < scale <= 1
    assert x >= 24 and y >= 250


def test_scene_plan_clamps_camera_and_reports_features():
    plan = scene_plan({"camera": "close", "gesture": "point"}, "wide", 4.0, True)
    assert plan["camera"] == "medium"
    assert plan["features"]["camera_step"] == 1
    assert plan["violations"] == []
    assert plan["comfort_score"] == 1.0


def test_classifier_is_fitted_from_rows_and_improves_with_features():
    rows = []
    for i in range(120):
        comfortable = i % 2 == 0
        rows.append({
            "features": {
                "camera_step": 0 if comfortable else 2,
                "gesture_switches_per_second": 0.2 if comfortable else 1.2,
                "jitter_amplitude": 1.0,
                "scene_seconds": 4.0,
                "flash_alpha": 0.05,
                "overlap_count": 0,
                "subtitle_words": 4,
                "easing": "smooth",
            },
            "label": "comfortable" if comfortable else "annoying",
        })
    full = ComfortClassifier().fit(rows)
    partial = ComfortClassifier(features=("camera_step",)).fit(rows)
    assert full.accuracy(rows) == 1.0
    assert partial.accuracy(rows) == 1.0
    assert full.predict({"camera_step": 2, "easing": "smooth"}) == "annoying"
