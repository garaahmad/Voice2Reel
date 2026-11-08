import ast
from pathlib import Path

import nbformat


def test_notebooks_are_valid_and_have_no_stored_outputs():
    paths = list((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert {path.name for path in paths} == {
        "00_ready_voice_ar_en.ipynb",
        "01_scriptwriter_qlora.ipynb",
        "02_arabic_voice_xtts.ipynb",
        "03_realistic_video.ipynb",
        "04_fast_stickman.ipynb",
    }
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.outputs == []
                assert cell.execution_count is None
