import json
import math
import shutil
import subprocess

import pytest

from voice2reel.motion_preview import MotionStudy, joint, render_preview


@pytest.mark.parametrize('target', [(60, 40), (200, 0), (0, 0), (-40, 80)])
def test_two_bone_solver_preserves_lengths_even_outside_reach(target):
    elbow, end = joint((0, 0), target, 70, 60)
    assert math.dist((0, 0), elbow) == pytest.approx(70)
    assert math.dist(elbow, end) == pytest.approx(60)
    assert all(math.isfinite(x) for point in (elbow, end) for x in point)


def test_frames_are_seekable_and_capture_different_actions():
    rig = MotionStudy(270)
    assert rig.frame(3.5).tobytes() == rig.frame(3.5).tobytes()
    assert rig.frame(1).tobytes() != rig.frame(5.75).tobytes()
    assert rig.frame(5.75).size == (270, 480)
    with pytest.raises(ValueError):
        rig.frame(float('nan'))


def test_preview_is_a_real_silent_nine_second_video(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg unavailable')
    output = render_preview(tmp_path/'preview.mp4', width=270, fps=6)
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)
    ]))
    assert len(data['streams']) == 1
    stream = data['streams'][0]
    assert stream['codec_type'] == 'video'
    assert (stream['width'], stream['height'], int(stream['nb_frames'])) == (270, 480, 54)
    assert float(data['format']['duration']) == pytest.approx(9, abs=.05)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
