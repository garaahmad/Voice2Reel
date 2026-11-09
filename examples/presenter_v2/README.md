# Presenter v2 samples and evidence

`semicolon_ar.json` is a **manually written and reviewed** Arabic example. It uses simple Modern Standard Arabic, not the model's raw output. The code illustration is the JavaScript empty-statement pitfall documented by [MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/Empty): an empty statement can become the body of an `if`, leaving the subsequent block independent.

`failed_ai_attempts/` preserves an actual rejected generation containing a Latin identifier in Arabic narration. `writing_review/` preserves a two-attempt real Qwen3 run that passed structure/language validation but **failed manual editorial review**. The raw response, generated JSON and timings are retained. Neither is represented as successful Arabic writing. No hand-written replacement is silently attributed to the model.

The manually reviewed sample tests the new voice, drawing, music, effects and end-to-end encoding independently of the writer's limitations. The default voice in v2 is an external online Microsoft Edge neural voice, not locally trained weights. Background music and effects are original synthesized audio.

Measurements refer to a two-core CPU quota; online speech depends on service/network response. The old Piper benchmark in `examples/fast_presenter` is historical. Download/setup and GPU timings are not inferred from local warm-environment tests.


`plain_brief_run/` preserves a separate actual one-pass run using a fully Arabic brief. It reduced Latin leakage but used التمرين التالي where الكتلة التالية was intended; its review remains rejected. Changing the prompt alone has not established reliable Arabic writing.

`measurements.json` records the reviewed Arabic and English jobs including the new sound mix. The two renders overlapped; these are actual job observations, not isolated benchmark results.
