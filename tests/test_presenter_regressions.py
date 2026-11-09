"""Regressions for jointed arms and the reviewed JavaScript explanation."""

import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import voice2reel.presenter_render as renderer
from voice2reel.presenter_schema import PresenterScript


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/presenter_v2/semicolon_ar.json'


@pytest.mark.parametrize('gesture', renderer.GESTURES)
@pytest.mark.parametrize('local', [0.0, 0.24, 1.2, 3.1])
def test_visible_arm_passes_through_solved_elbow(monkeypatch, gesture, local):
    script = PresenterScript.model_validate_json(EXAMPLE.read_text())
    script.scenes[1].gesture = gesture
    segments = [dict(start=i * 4, end=(i + 1) * 4, text=s.narration) for i, s in enumerate(script.scenes)]
    actor = renderer.PresenterRenderer(script, segments, np.zeros(24000 * 12), 24000)
    solved, drawn = [], []
    original_joint, original_line = renderer.joint, renderer.rounded_line

    def joint(root, target, upper, lower, bend):
        elbow, wrist = original_joint(root, target, upper, lower, bend)
        assert math.dist(root, elbow) == pytest.approx(upper)
        assert math.dist(elbow, wrist) == pytest.approx(lower)
        solved.append([root, elbow, wrist])
        return elbow, wrist

    def line(draw, points, width, color=renderer.INK):
        drawn.append(points)
        return original_line(draw, points, width, color)

    monkeypatch.setattr(renderer, 'joint', joint)
    monkeypatch.setattr(renderer, 'rounded_line', line)
    actor.character(4 + local, 1, local)
    assert len(solved) == 2
    # The relaxed left arm must bend away from the torso, including while
    # switching poses. The old inward solution made it cross the body.
    assert solved[0][1][0] < solved[0][0][0]
    for bones in solved:
        pixels = [(round(x * 2), round(y * 2) - 500) for x, y in bones]
        assert pixels in drawn, 'The visible arm must follow both bones through the elbow'


def test_reviewed_example_code_exhibits_and_fixes_the_explained_bug():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute the JavaScript example')
    script = json.loads(EXAMPLE.read_text())
    for i, scene in enumerate(script['scenes']):
        code = scene['visual']['code']
        assert '\\n' not in code
        assert '\n' in code
        for ready in (False, True):
            program = (
                f'const ready = {json.dumps(ready)}; let calls = 0; '
                'function run() { calls++; }\n' + code + '\nconsole.log(calls);'
            )
            result = subprocess.run([node, '-e', program], capture_output=True, text=True, check=True)
            assert int(result.stdout) == (int(ready) if i == 2 else 1)


@pytest.mark.parametrize("mouth", [0, 1, 2, 3])
def test_mouth_has_no_red_tongue_dot(mouth):
    face = renderer.face("happy", mouth, False)
    pixels = np.asarray(face.crop((180, 340, 290, 420)))
    red = (pixels[:, :, 0].astype(float) > pixels[:, :, 1] * 1.5) & (pixels[:, :, 0] > 80)
    assert not np.any(red & (pixels[:, :, 3] > 0))
