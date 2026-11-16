"""Quality gates for semantic choreography and topic-held-out curriculum."""
from scripts.build_large_scene_curriculum import TOPICS, generate, split_for
from voice2reel.python_scene_dsl import parse_python_scene

def test_topic_holdout():
    assert set(split_for(t) for t in TOPICS) == {"train", "validation", "test"}
    assert len(set(TOPICS)) == len(TOPICS)

def test_bilingual_semantic_story():
    for i in range(72):
        row = generate(i, 20261116)
        plan = parse_python_scene(row["messages"][-1]["content"])
        assert len(plan.shots) == len(row["metadata"]["story_structure"])
        assert 1 <= sum(shot.seconds for shot in plan.shots) <= 60
        assert all(shot.narration and shot.caption for shot in plan.shots)
        assert row["metadata"]["language"] in {"ar", "en"}
        for beat, shot in zip(row["metadata"]["story_structure"], plan.shots):
            if beat == "warning":
                assert shot.gesture == "think"
            if beat == "cta":
                assert shot.gesture == "wave"

def test_determinism():
    assert generate(17, 42) == generate(17, 42)
    assert generate(17, 42) != generate(17, 43)
