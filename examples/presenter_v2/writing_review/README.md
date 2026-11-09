# Actual Qwen3 run: rejected editorial quality

These are the unchanged recorded files from a real CPU run with online speech and original music/effects. `script.trace.json` records two model calls: a rejected first response and a correction. The old `text_model.json` field `passes: one compact storyboard` describes the pipeline type, not the call count; the trace is the source for actual attempts. That ambiguous field has been fixed in the pipeline for subsequent jobs.

Manual review rejected the final wording despite schema validation passing. See `review.json`. The generated video is deliberately not the primary quality demo. `../semicolon_ar.json` is separately marked as a manually reviewed script.
