from voice2reel.quality_gate import (
    CodeSwitchReader, ComfortClassifier, Diacritizer, dataset_integrity, rows, run_rounds,
)


def test_integrity_checks_all_pass_on_committed_data():
    checks = dataset_integrity()
    failing = [key for key, value in checks.items() if value is False]
    assert failing == []
    assert checks["voice_prompt_count"] == 140
    assert checks["motion_annotations"] > 3000
    assert checks["tashkeel_mark_classes"] >= 8


def test_front_end_models_are_fitted_and_meet_documented_target():
    train_tk, train_cs, train_mo = rows("tashkeel", "train"), rows("codeswitch", "train"), rows("motion_plans", "train")
    diacritizer = Diacritizer().fit(train_tk)
    reader = CodeSwitchReader().fit(train_cs)
    classifier = ComfortClassifier().fit(train_mo)
    assert diacritizer.metrics(rows("tashkeel", "test"))["word_accuracy"] > 0.9
    assert reader.metrics(rows("codeswitch", "test"))["exact_rate"] > 0.9
    assert classifier.accuracy(rows("motion_plans", "test")) > 0.9
    report = run_rounds(max_rounds=3)
    assert report["rounds"]
    assert report["final_test_on_best_round"]["score"] >= 0.90
    assert report["best_round"] == max(report["rounds"], key=lambda r: r["score"])["round"]
