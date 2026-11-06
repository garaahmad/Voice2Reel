import pytest

from voice2reel.evaluate import edit_distance, normalize_arabic, speech_metrics


def test_arabic_normalization_and_error_counts():
    assert normalize_arabic("أَهْلًا، بالعالم!") == "اهلا بالعالم"
    result = speech_metrics([{"reference": "مرحبا يا صديق", "hypothesis": "مرحبا صديق"}])
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["cer"] > 0
    assert edit_distance([], [1, 2]) == 2
    assert speech_metrics([{"reference": "نص", "hypothesis": "نص"}])["cer"] == 0
    with pytest.raises(ValueError):
        speech_metrics([{"reference": "", "hypothesis": "نص"}])
