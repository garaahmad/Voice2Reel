# Arabic narration quality / Voice2Reel

This improvement affects **only** the text-to-speech stage:
\`voice2reel/production_voice.py\` and the notebook voice cells.

No edits were made to the mascot, animations, scene frames, camera, or video
dimensions. Output remains \`voice.mp3\` for the existing video render pipeline.

## Quick audition (no Qwen training)

Open the Colab notebook at
\`notebooks/10_ai_story_director_all_in_one.ipynb\`, run Setup and generate a
storyboard (or use an existing \`scene.json\`). Enable
\`MAKE_VOICE_PREVIEWS=True\` and run the optional \`Compare Arabic voices\`
cell. Four comparable MP3 previews are generated from the first shot.
Pick the naturalness and accent you prefer by listening rather than by name.

Choices:
- ar-JO-TaimNeural — male, Jordan
- ar-EG-ShakirNeural — male, Egypt
- ar-SA-HamedNeural — male, Saudi Arabia
- ar-SA-ZariyahNeural — female, Saudi Arabia
- ar-EG-SalmaNeural — female, Egypt

The new default is \`ar-JO-TaimNeural\` at \`+0%\` speed and \`+0Hz\` pitch.
These settings **do not guarantee a natural-sounding result**; the underlying
Edge neural voices and quality of the written Arabic script set a limit.

## Keep the video unchanged while improving narration

In the \`Voiceover\` cell:
- Change \`VOICE_NAME\`, \`VOICE_RATE\`, or \`VOICE_PITCH\`.
- Set \`VOICE_OVERRIDES = {"1": "أهلًا وسهلًا! اليوم رح نعرف كيف يشتغل الراوتر."}\`
  to rewrite ONLY what is spoken in shot 1. Keys are 1-based.
- Set \`VOICE_TERMS = {"Flutter": "فلاتر", "setState": "سِت ستيت"}\` to
  override the pronunciation of specific terms, without changing captions.
- The final spoken text is written beside the MP3 as \`voice.speech.txt\` to
  make it easy to inspect and correct.

The generator preserves the original narration by default except for spaces
and sentence-ending punctuation; it does not claim to convert formal Arabic
into a Palestinian dialect automatically. For better delivery, review each
sentence and explicitly rewrite stilted or mistranscribed phrases.

**Warning:** The audio is currently assembled as one narration track. If it
runs longer than the sum of shot durations, the existing video mux may cut its
ending. Colab now reports audio and video lengths before rendering. To address
that warning without altering visuals, shorten or tighten \`VOICE_OVERRIDES\`.

## Command-line examples

\`\`\`bash
python -m voice2reel.production_voice \
  --preview-text "أهلًا فيكم! اليوم رح نشرح كيف يشتغل الإنترنت." \
  --voice ar-JO-TaimNeural --rate=+0% --pitch=+0Hz \
  --language ar --output /tmp/voice-preview.mp3

python -m voice2reel.production_voice \
  --plan /path/to/scene.json --output /path/to/voice.mp3 \
  --language ar --voice ar-JO-TaimNeural --rate=+0% --pitch=+0Hz
\`\`\`

Edge TTS requires an online connection. Its terms and available voices may
change independently of this repository. This stage does not clone voices,
train a speech model, or impersonate real people.

## Replace the sound in the existing video without changing the drawing

The notebook includes the optional cell **Replace ONLY the audio in an existing MP4**.

1. Generate and listen to the corrected \`voice.mp3\`.
2. Ensure the original \`final.mp4\` already exists.
3. Set \`REPLACE_AUDIO_ONLY = True\` and run the cell.

It produces \`final_better_arabic_voice.mp4\` beside the original video
by copying the existing encoded video stream (\`-c:v copy\`) and
re-encoding **only audio**. It does not rerender the mascot, frame geometry,
camera, animation or captions. It pads short audio with silence; audio that
exceeds the existing video duration will end with the video. Review any
duration warning in the preceding Voiceover cell.
