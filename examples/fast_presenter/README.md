# Fast presenter examples

`semicolon_ar.json` and `semicolon_en.json` are reviewed, hand-authored scripts for testing the renderer and actual AI speech. They are not claimed as raw AI-generated scripts. The technical point is supported by [MDN's empty-statement example](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/Empty).

`benchmark.local.json` records two real runs on a two-core CPU quota: a new voice/video from the reviewed Arabic script with local models, and a repeat using the existing outputs. CLI wall time includes process startup; model downloads/installation are excluded. A cache hit is not generation of a new story.

`actual_ai_run/` preserves a separate real Qwen3→Piper→MP4 run from a written idea. Its wording needs review before publishing, and it is not a quality benchmark. GPU/Colab timing is unmeasured.
