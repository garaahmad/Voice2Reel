# Evaluate before choosing a model

## Scriptwriter

Use the same held-out ideas and settings for base and adapted models. The notebook reports first-pass JSON validity and validity after at most three attempts. Invalid generations are counted, never silently replaced by templates. Validation catches structure, unsupported visual values, requested duration, and rough text length; it cannot prove factual accuracy.

Review outputs blind where possible and score 1–5 for relevance, clear opening, natural Arabic, factual support, narration flow, and useful scene/caption choices. Count invented facts separately. Actual duration is measured after speech generation; word count is only a rough guard. Keep raw outputs and record the dataset/model revisions.

The bundled test set is only five original topics with three beats and a 20-second target. It is a smoke benchmark, not evidence of broad capability. Add held-out topics, dialects, durations, different beat counts, numerical claims, and mixed Arabic/English names before production. Never tune repeatedly against the final test set.

## Voice

Use identical reference audio, held-out text, seed, and synthesis settings for baseline and fine-tuned checkpoints. Listen to short and long sentences, questions, numbers written as spoken, and unfamiliar names. Record omitted/repeated words, intelligibility, rhythm, background artifacts, and perceived speaker similarity. Low validation loss alone does not establish a good voice.

For content error rates, transcribe generated audio manually or use an ASR system and manually check it. Save JSONL:

```json
{"reference":"مرحبا بكم في هذا المثال","hypothesis":"مرحبا بكم في المثال"}
```

Then run:

```bash
python -m voice2reel.evaluate speech --input transcripts.jsonl --output outputs/speech_metrics.json
```

Reports corpus-level WER and CER with documented Arabic normalization: NFKC, removal of Arabic diacritics/tatweel, normalization of alef variants, punctuation removal, and whitespace collapse. These measure content accuracy, not timbre similarity. ASR mistakes can inflate them; do not compare scores from different transcription pipelines.

Record timing, maximum GPU memory, and failures for real Colab runs. A syntactically valid notebook or CPU test is not a GPU training benchmark. Never report a metric without a completed measurement.
