# Voice2Reel

Turn a written idea into a short video narrated in your own cloned voice.

**Status: training toolkit implemented; product pipeline still in development.** The repository now includes Colab notebooks, training/inference code for script generation and Arabic voice cloning, starter data, and tests. GPU training and voice quality have not been measured here. The renderer, backend, and user interface remain planned.

## Train and experiment on Colab

Start with the **[Arabic Colab guide](docs/COLAB_AR.md)**.

| Component | Notebook | Approach |
|---|---|---|
| Written idea → Arabic script and visual beats | [Scriptwriter QLoRA](notebooks/01_scriptwriter_qlora.ipynb) | Qwen2.5-1.5B-Instruct, NF4 + LoRA |
| Script + reference voice → narration | [Arabic voice XTTS](notebooks/02_arabic_voice_xtts.ipynb) | XTTS-v2 baseline inference and GPT fine-tuning |

The notebooks target separate free-Colab T4 sessions, save checkpoints to Drive, and support private-repository ZIP uploads. Base weights download automatically at pinned revisions; no newly trained weights are shipped. Voice training requires real recordings matching transcripts. The included 40 script examples and 120 recording prompts are a starter kit, not a production dataset.

See [dataset provenance](datasets/README.md), [research and model licenses](docs/RESEARCH.md), [evaluation](docs/EVALUATION.md), and [verified checks and remaining limits](docs/VALIDATION.md). XTTS weights and outputs are subject to the non-commercial CPML license.

The sections below describe the full product; planned API/rendering components are not provided by the training toolkit.

## The idea

The user provides:

1. A **written idea** describing the video they want.
2. A **recording of their own voice** as a reference for voice cloning.

An AI scriptwriter expands the idea into a narration script. A voice-cloning model generates new speech that reads the script in the user's voice. The server creates the visuals, synchronizes them with the generated narration, and exports the finished video.

The voice sample supplies the voice identity. Its words do not need to match the idea or the script, and its duration does not determine the final video duration.

## User flow

1. Write an idea.
2. Upload a voice sample.
3. Let AI generate the script.
4. Review and edit the script.
5. Start generation: synthesize the narration in the cloned voice, plan the scenes, and render the video on the server.
6. Preview and download the final video.

Example: a user writes “Explain why developers should write tests” and uploads a sample of their voice. AI writes a suitable script, the voice model reads that new script in the user's voice, and the server produces a video with matching scenes and captions.

## Pipeline

```text
Written idea ──▶ AI scriptwriter ──▶ Editable script
                                          │
Uploaded voice sample ──▶ Voice cloning ◀──┘
                                │
                        Generated narration
                                │
                         Speech timestamps
                                │
Script + timestamps ──▶ AI Director ──▶ Scene-plan JSON
                                              │
                                          Validator
                                              │
                                       Server renderer
                                              │
                   Visuals + narration + optional music/SFX
                                              │
                                            ffmpeg
                                              │
                                       Downloadable MP4
```

The planned server orchestrates the entire process. The current experiment uses local Qwen and XTTS models; production model/provider selection remains open.

## Responsibilities

| Component | Responsibility |
|---|---|
| Input | Accept the written idea and reference voice recording |
| Scriptwriter | Turn the written idea into an editable narration script |
| Voice cloning / TTS | Read the approved script using the reference voice |
| Alignment | Obtain segment or word timestamps from the generated narration |
| Director | Convert the script and speech timing into a structured scene plan |
| Validator | Check supported visual values, timeline continuity, and media references |
| Engine | Evaluate the validated scene plan into per-frame state |
| Rig | Draw characters, panels, Arabic text, backgrounds, and effects |
| Audio mixer | Combine narration with optional music and sound effects |
| Encoder | Export the synchronized video and audio as MP4 |
| API and worker | Store jobs, schedule stages, report progress, and serve results |

**Core rendering principle:** the story is data, the engine is fixed. A new video changes the validated scene plan rather than the rendering source code.

The engine decides what appears and when. The rig handles drawing. Scriptwriting and voice synthesis remain separate from rendering.

## Video and timing

The proposed initial visual style is vertical motion graphics at **1080 × 1920, 30 FPS**, using reusable characters, panels, captions, and effects. Other visual styles and output formats remain open decisions.

The final timeline follows the **generated narration**, using provider timestamps or a separate alignment step.

- Generate the script for the requested target duration.
- Measure the synthesized narration before finalizing scene boundaries.
- Synchronize scenes, captions, headlines, and sound cues to the speech.
- If speech exceeds the target duration, revise the script and regenerate it or let the user accept a different duration. Do not silently cut spoken words.

Voice-sample length, generated narration length, and target video length are separate concepts. Sample requirements depend on the selected voice-cloning model.

## Scene-plan format

The planned intermediate representation is versioned JSON validated with Pydantic.

| Data | Purpose |
|---|---|
| Script revision | Identify the approved narration text |
| Narration reference | Identify the generated audio artifact |
| Duration | Actual final timeline duration |
| Scenes | Contiguous character, camera, and prop states |
| Captions | Speech-synchronized subtitles |
| Headlines | Independent title overlays |
| Events | Timed effects and sound cues |
| Music settings | Optional background track configuration |

The Director selects only supported poses, expressions, camera modes, and effects. It does not generate executable code.

Planned validation rules:

- The first scene starts at zero, scenes have no gaps or overlaps, and the last scene ends at the video duration.
- Captions and headlines have positive durations and remain inside the timeline.
- Highlighted text is a substring of the raw caption text before Arabic shaping.
- Events stay inside the timeline.
- The narration and scene plan belong to the same script revision.
- The narration is not truncated by the video timeline.

## Proposed backend

**Python 3.11+ · FastAPI · Pydantic v2 · separate worker process · SQLite WAL · ffmpeg.**

| Concern | Proposed choice |
|---|---|
| HTTP API | FastAPI + Uvicorn |
| Validation | Shared Pydantic models |
| Job store | SQLite WAL for an initial single-machine deployment |
| Heavy stages | Worker with subprocess isolation and stage-specific timeouts |
| Rendering | Python, NumPy, Pillow, and ffmpeg |
| Arabic text | Centralized shaping and bidirectional text handling |
| Scriptwriting | Provider-agnostic LLM interface |
| Scene planning | Provider-agnostic Director with structured JSON output and bounded retries |
| Voice cloning | XTTS-v2 experimental toolkit; production provider selection pending |
| Alignment | Provider timestamps or a separate alignment tool |
| Storage | Local disk behind a Storage interface; object storage later if needed |
| Packaging | Docker image with pinned dependencies, ffmpeg, and fonts |

The API validates inputs, stores jobs, and returns promptly. It does not render frames inside HTTP requests.

Voice-model compute requirements must be evaluated separately from rendering. A local voice model may require a GPU; an external voice API has different cost and latency constraints.

## Job lifecycle and review

```text
received
  -> scripting
  -> awaiting_review
  -> queued
  -> cloning_voice
  -> aligning
  -> planning
  -> validating
  -> rendering
  -> mixing
  -> done
```

Jobs may also become failed or cancelled. The cloning stage covers reference-voice preparation and script synthesis; some providers combine these operations.

Review happens before voice generation and rendering. An explicit generate action starts the expensive stages.

Every generation uses a fixed script revision. Editing the script invalidates its narration, timestamps, scene plan, and video. Editing only the scene plan invalidates the video and must preserve narration timing constraints.

## Planned API

These endpoints describe a proposed interface; they are not implemented.

| Method | Path | Purpose |
|---|---|---|
| POST | /api/v1/jobs | Submit a written idea and voice sample as multipart data; return 202 with job ID |
| GET | /api/v1/jobs/{id} | Read status, stage progress, timings, and errors |
| GET | /api/v1/jobs/{id}/script | Retrieve the generated script |
| PUT | /api/v1/jobs/{id}/script | Save an edited script and invalidate dependent outputs |
| GET | /api/v1/jobs/{id}/story | Retrieve the generated scene plan |
| PUT | /api/v1/jobs/{id}/story | Validate and save an edited scene plan |
| POST | /api/v1/jobs/{id}/render | Generate narration and video for the current approved script revision; idempotent while active |
| GET | /api/v1/jobs/{id}/video | Download or stream the completed MP4 |
| DELETE | /api/v1/jobs/{id} | Cancel the job and purge its artifacts |
| GET | /health | Report worker availability, queue depth, and storage health |

## Proposed repository layout

Paths below are targets, not claims that these files already exist.

```text
voice2reel/
├── voice2reel/
│   ├── schema.py          # Script, scene-plan, and timeline validation
│   ├── scriptwriter/      # Written idea -> narration script
│   ├── voice/             # Voice-cloning provider and speech generation
│   ├── alignment.py       # Generated speech -> timestamps
│   ├── director/          # Script + timing -> scene-plan JSON
│   ├── engine.py          # Scene plan + time -> frame state
│   ├── rig/               # Reusable drawing primitives
│   ├── audio.py           # Narration, music, and SFX mixing
│   ├── render.py          # Validated plan + media -> MP4
│   ├── jobs.py            # Job persistence and worker orchestration
│   └── api/               # FastAPI endpoints
├── tests/
├── assets/fonts/          # Fonts and their licenses
├── docker/Dockerfile
└── data/                  # Runtime uploads and artifacts; gitignored
```

## Rendering and audio rules

- Validate the scene plan before drawing frames.
- Keep story-specific logic out of the reusable rig.
- Keep Arabic shaping in one module and test Arabic mixed with Latin text.
- Use resource-based font paths and avoid import-time rendering or exits.
- Check ffmpeg exit codes and surface failures in job status.
- Use the generated narration as the primary audio track.
- Duck optional music and effects under speech.
- Target approximately −14 LUFS for the final mix using two-pass loudness normalization.
- Seed synthesized SFX per event with a stable digest so adding one event does not change other sounds.
- Keep rendering deterministic for fixed scene plans, media, assets, and renderer versions. External AI outputs may vary and must be stored as artifacts.

Cache voice synthesis by script content/revision, reference-voice hash or version, provider/model version, and synthesis settings. Cache final renders by the validated scene plan, narration content hash, renderer version, font/assets hashes, and audio settings.

Validate uploaded audio using ffprobe and the selected model's format, size, and duration requirements. Use generated filenames and stage-specific timeouts. Recover or fail abandoned jobs after worker restarts.

## Legacy renderer context

Earlier design work references two scripts outside this repository: reel_v2.py and yellow_stripe_reel.py. They contain reusable motion-graphics techniques but hardcode each video's timeline.

The planned refactor extracts a reusable rig and data-driven engine. Before visual improvements, compare frames and synthetic audio against legacy fixtures to establish parity. Voice-cloned narration is a new component and needs separate evaluation.

Previously documented legacy issues to check when those scripts are imported:

- Hardcoded font paths and import-time side effects.
- Caption highlighting ignored by the drawing function.
- Duplicate or unused pose definitions and layout values.
- Unchecked ffmpeg failures.
- Full-size background copies on every frame.
- Global RNG order affecting synthesized audio.

## Status and roadmap

| Area | Current status |
|---|---|
| Product specification | Updated for written ideas and cloned-voice narration |
| Legacy scripts | Previously documented as outside this repository |
| Schema | Previously described as an untested draft; implementation not established here |
| Rig and engine | Not implemented in this repository |
| Scriptwriter | QLoRA training, validated generation, and Colab notebook implemented; GPU evaluation pending |
| Voice cloning and narration | XTTS-v2 training/inference toolkit implemented; real recordings and GPU evaluation required |
| Alignment and Director | Structured visual beats and per-segment synthesis timing implemented; word alignment and renderer scene plan remain planned |
| Backend and worker | Not started |
| User interface | Not started |

Planned implementation order:

1. Evaluate voice-cloning options for Arabic quality, voice similarity, sample requirements, compute needs, cost, and commercial terms.
2. Define script, narration-artifact, and scene-plan schemas.
3. Import and extract the reusable legacy rig, removing import-time side effects.
4. Implement the data-driven engine and establish legacy parity before visual polish.
5. Implement script generation and script review.
6. Integrate voice cloning, narration generation, and timestamps.
7. Generate and validate scene plans against actual narration timing.
8. Render visuals and mix narration, music, and effects.
9. Add the API, job worker, progress reporting, and user interface.
10. Validate the complete flow: written idea + voice sample -> reviewed script -> narrated video -> download.

## Open decisions

| Question | Impact |
|---|---|
| Which LLM should write scripts and plan scenes? | Quality, cost, latency, and structured-output reliability |
| Which voice-cloning model or provider? | Voice similarity, Arabic dialect support, hosting, and sample requirements |
| Which timestamp/alignment method? | Caption and scene synchronization |
| What video durations should be supported? | Script length and narration pacing |
| Which visual styles should v1 include? | Rig capabilities and scene vocabulary |
| One character or several? | Scene schema and layout |
| Vertical only or other aspect ratios? | Relative layout and rendering dimensions |
| Internal tool or public service? | Authentication, access isolation, storage, and deployment scope |

## Short description

Voice2Reel turns a written idea into a short video narrated in your own voice. AI writes the script, a voice-cloning model generates narration from your uploaded voice sample, and the server creates the visuals, synchronizes them with the speech, and exports the finished video.

