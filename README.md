# Voice2Reel

Turns a spoken idea into a 30-second vertical motion-graphics video (1080×1920, 30 FPS).

> **Core principle:** the story is data, the engine is fixed. A new video changes only the JSON — never the source code.

**Status: pre-implementation.** This repository currently contains design documentation only. There is no renderer, schema, engine, or backend yet — see [Status](#status). Do not treat any file path below as existing.

---

## 1. The idea

The existing scripts (`yellow_stripe_reel.py`, `reel_v2.py`) are hand-made animations: `if t < 2.5: … elif t < 4.2: …` chains that set poses, cameras, captions, and sound cues in dozens of places. Producing a new video means editing the timeline logic in every one of those places.

Voice2Reel splits that into layers with strict responsibilities, so that producing a new video is a **data** operation:

```
voice (10–30 s)
   │
   ├─▶ [1] Speech to text ──────────▶ transcript
   │
   ├─▶ [2] Director (LLM) ──────────▶ story JSON  (closed vocabulary only)
   │
   ├─▶ [3] Validator (Pydantic) ────▶ accept, or reject → back to [2] (max 2–3 retries)
   │
   ├─▶ [4] Engine ──────────────────▶ per-frame state  (scene, pose, camera, overlays)
   │
   ├─▶ [5] Rig ─────────────────────▶ pixels (stickman, panels, Arabic text, background)
   │
   ├─▶ [6] Audio synthesis ─────────▶ SFX + music bed
   │
   └─▶ [7] ffmpeg ──────────────────▶ MP4  (H.264 CRF 17, AAC 192 kbps, −14 LUFS)
```

The system is best understood as a **compiler**: an unconstrained input (human speech) is lowered into a verified intermediate representation (JSON), then compiled deterministically into pixels and audio. Every discipline in this project follows from that; every failure mode comes from letting story logic leak into a layer that should not own it.

## 2. Layers

| Layer | Responsibility | Changes per video? |
|---|---|---|
| Input | Record voice, transcribe to text | No |
| Director | LLM writes the story as JSON | **Yes** |
| Schema | Validate the JSON (structure + timeline) | No |
| Engine | Read validated JSON, evaluate the timeline per frame | No |
| Rig | Stickman, camera, panels, Arabic text, effects, audio synthesis | No |

The boundary that matters most is **Engine vs Rig**: the engine knows *when and what*, the rig knows *how*. Story logic inside the rig is the original problem returning.

## 3. Story format — four independent tracks

Captions, headlines, scenes, and sound cues do not share boundaries in the original scripts, so the format keeps them independent.

| Track | Purpose |
|---|---|
| `scenes` | Contiguous, gap-free character + camera states |
| `captions` | Subtitle bar at the bottom |
| `headlines` | Large title text at the top |
| `events` | Instant moments: shake, zoom punch, flash, glitch, sound cues |

```json
{
  "schema_version": 1,
  "title": "Silent Green Build",
  "duration": 30,
  "music": "lofi",
  "scenes": [
    {"start": 0.0, "end": 3.0, "pose": "SHOCK", "expr": "wide",
     "camera": "extreme", "badge": "Success"},
    {"start": 3.0, "end": 6.2, "pose": "TYPE_A", "pose_motion": "type",
     "camera": "medium", "zoom_drift": 0.03, "laptop": "code"}
  ],
  "captions": [
    {"start": 0.0, "end": 1.5, "text": "…", "highlight": "…"}
  ],
  "headlines": [],
  "events": [
    {"t": 0.55, "type": "impact", "strength": 0.85},
    {"t": 23.0, "type": "glitch", "strength": 1.0}
  ]
}
```

Closed vocabularies (the LLM may not invent values):

| Field | Allowed |
|---|---|
| `pose` | IDLE, SHOCK, CELEB, SUSPECT, FREEZE, LEAN, TYPE_A, TYPE_B, LOOK_L, LOOK_R, ERROR |
| `pose_motion` | none, type, look_sweep |
| `expr` | normal, happy, wide, squint |
| `camera` | far, medium, close, extreme, left, right |
| `laptop` | null, code, green, error |
| `card` | `code` or `print`, up to 8 lines |
| `lens` / `confetti` / `glow` | boolean |
| `badge` | ≤ 20 characters |
| `look` | −1.0 … 1.0 |

Timeline invariants enforced by the validator: first scene starts at 0 · scenes contiguous (no gaps, no overlaps) · last scene ends exactly at `duration` · every caption/headline has `end > start` and stays inside the video · a `highlight` must be a substring of its `text` · no event after the end.

## 4. Target repository layout

```
voice2reel/
├── voice2reel/
│   ├── schema.py          # Pydantic models + timeline validators (source of truth)
│   ├── engine.py          # (story, t) -> frame state. No drawing. No story content.
│   ├── rig/               # drawing primitives: stickman, camera, panels, text, fx
│   ├── audio.py           # SFX + music synthesis, deterministic per event
│   ├── render.py          # CLI: story.json -> mp4  (+ progress on stdout)
│   ├── director/          # prompt, JSON-mode client, retry loop
│   ├── asr.py             # speech to text
│   ├── jobs.py            # job store (SQLite) + worker loop
│   └── api/               # FastAPI app — thin, schedules only
├── tests/
│   ├── test_schema.py     # valid / invalid stories
│   ├── test_parity.py     # slow, marked; frames + audio vs. legacy output
│   └── fixtures/
├── assets/fonts/          # Cairo-Black.ttf + OFL license
├── docker/Dockerfile      # pinned ffmpeg + fonts
└── data/                  # runtime: uploads/, stories/, renders/  (gitignored)
```

## 5. Backend — technology decision

### The constraint that decides it

The renderer is Python, full stop: NumPy frame buffers, Pillow drawing, `arabic-reshaper` + `python-bidi` for Arabic shaping, Pydantic for validation, ffmpeg over a stdin pipe. That is not a replaceable component — it is the product. Every other runtime would end up shelling out to a Python child process anyway, adding a hop and duplicating the story schema on the other side.

### Decision

**Python 3.11+ · FastAPI + Uvicorn · one service · a separate worker process for heavy stages · SQLite job store · local disk behind a `Storage` interface · one Docker image with pinned ffmpeg and fonts.**

| Concern | Choice | Why |
|---|---|---|
| HTTP API | **FastAPI** | shares the Pydantic models already required by the schema; automatic OpenAPI; async uploads |
| ASGI server | Uvicorn (Granian as an optional swap) | boring and reliable |
| Validation | **Pydantic v2 — the same models as `schema.py`** | one definition validates LLM output, API payloads, and stored stories |
| Job store | **SQLite (WAL)** + `jobs.py` | one box, one worker; no Postgres/Redis to operate on day one |
| Heavy work | **subprocess** `python -m voice2reel.render …` | crash/OOM isolation, killable timeout, CPU limits, engine stays a pure CLI |
| Progress | render CLI prints `{"frame": n, "total": 900}` lines → worker → job row | no extra infra for a progress bar |
| Storage | local `data/` behind `Storage` (put/get/url) | swap to S3 without touching call sites |
| Speech to text | **faster-whisper** (CTranslate2, int8 on CPU) | runs in the same worker, no GPU required |
| LLM | one `Director` interface, provider-agnostic, JSON-schema output | don't lock in before pricing and commercial terms are checked |
| Container | Debian slim + pinned ffmpeg + font baked in | kills the "`/home/ahmad/…` font path" class of bug |
| Tests | pytest; parity suite marked `slow` | parity is the safety net for the refactor |
| Deploy | single VPS, Caddy in front, one image, two commands (`api`, `worker`) | cheapest correct shape for v1 |
| Auth | none in v1 | no accounts until there is a second user |

### Explicitly rejected

| Option | Why not |
|---|---|
| **Node / NestJS / Express** | the renderer is Python; every render becomes a cross-runtime proxy and the story schema gets duplicated |
| **Django + DRF** | admin and auth are genuine benefits, but v1 has no users or accounts; DRF serializers would duplicate Pydantic and drift from it |
| **Streamlit as the backend** | excellent for a demo, wrong as a backend: single-user, rerun-on-interaction, no API; a 3-minute job does not fit its execution model. Keep it only as an internal scratchpad |
| **Serverless / Lambda** | execution time limits, ffmpeg binary size, cold starts, ephemeral disk, no long-lived worker |
| **Celery + Redis on day one** | one machine, one worker: an atomic SQLite claim loop is enough. Add Redis + `arq` only when there is a second render node or per-task retry semantics |
| **Docker per render / K8s** | premature; a subprocess with a timeout buys most of the isolation for none of the operations cost |

### Why not render inside the request

A 30-second video is 900 frames drawn at 2× supersampling (2160×3840) and downscaled — minutes of single-core CPU per video. So:

- HTTP handlers never draw. They only validate, write a row, and return `202`.
- The worker runs the renderer as a **subprocess**, not a thread: it can be killed on timeout, memory-limited, and it cannot take the API down.
- Concurrency = one render per ~2 cores. Queue depth is visible in `/health`.

### Two-phase job: cheap first, expensive second

Transcription and direction take seconds; rendering takes minutes. So the job splits:

```
POST /jobs ─▶ transcribing ─▶ directing ─▶ validating ─▶ awaiting_review
                                                            │ (story JSON editable)
                                       PUT /jobs/{id}/story ┘  validate + store
                                       POST /jobs/{id}/render ─▶ queued ─▶ rendering ─▶ mixing ─▶ done
```

The user reviews and edits the story **before** paying for a render. Re-rendering after an edit is an explicit action, not a side effect of a keystroke.

### Planned API surface (v1)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | upload audio (multipart) → `202 {job_id}` |
| `GET` | `/api/v1/jobs/{id}` | stage, progress, timings, error |
| `GET` | `/api/v1/jobs/{id}/story` | story JSON for review/editing |
| `PUT` | `/api/v1/jobs/{id}/story` | validate + store → `200` or `422` with `[{path, message, allowed}]` |
| `POST` | `/api/v1/jobs/{id}/render` | enqueue render (idempotent while queued/rendering) |
| `GET` | `/api/v1/jobs/{id}/video` | stream or redirect to the MP4 |
| `DELETE` | `/api/v1/jobs/{id}` | cancel + purge artifacts |
| `GET` | `/health` | queue depth, disk free, ffmpeg + renderer version, cache hit rate |

Job states: `received → transcribing → directing → validating → needs_fix → awaiting_review → queued → rendering → mixing → done`, plus `failed` and `cancelled`.

### Operational rules

- **Deterministic cache:** key = `hash(story_json, renderer_version, font_hash)`. Same story → same file, no re-render. This is cheap and pays for itself immediately during schema and Director iteration.
- **Upload hygiene:** `ffprobe` for real MIME and duration; reject anything outside 10–30 s or over the size cap; UUID filenames only, never user input.
- **Stale-job reclaim:** on worker boot, any job stuck in `rendering` longer than the timeout is reset or failed — never left hanging.
- **No absolute paths, no import-time side effects:** the font is injected as a resource; importing `rig` must not build backgrounds or call `sys.exit`.
- **Structured logs** per stage with duration, so a slow render can be attributed to drawing vs. encoding.
- **`schema_version` in every stored story**, so old stories can be migrated rather than silently re-interpreted.

## 6. Status

| Item | State |
|---|---|
| Design document (this README) | done |
| `reel_v2.py` (rig + hardcoded story) | exists outside this repo |
| `yellow_stripe_reel.py` (second video) | exists outside this repo |
| `rig_core.py` (rig split out) | not started |
| `schema.py` | draft, **untested** |
| `engine.py` | not started |
| `story_reel_v2.json` | not started |
| Parity test | not started |
| Director (LLM) | not started |
| Speech to text | not started |
| Backend (FastAPI) | not started — stack decided above |
| UI | not started |

The two existing scripts live outside this repository today; importing `reel_v2` has top-level side effects (backgrounds built on import, `SystemExit` if no Arabic font is found), which is the first thing to remove.

## 7. Roadmap

| Phase | Task | Done when |
|---|---|---|
| 1 | Split the rig into `rig_core.py`; `reel_v2.py` re-exports it | `yellow_stripe_reel.py` still runs unchanged |
| 2 | Finish and test `schema.py` | valid stories pass, invalid ones fail with specific messages |
| 3 | `engine.py` — reads a story, calls the rig, no story content | engine renders a story file to frames |
| 4 | `story_reel_v2.json` | the JSON validates |
| 5 | Parity test (old vs. new, frames + audio) | frames match exactly or within tolerance; audio matches |
| 6 | Scene transitions (0.2–0.3 s blends) | visible improvement; parity no longer required past this point |
| 7 | Director prompt + API + retry loop | 10 test ideas produce valid stories |
| 8 | Backend: FastAPI + worker + job store; then a Streamlit or web UI | record → transcribe → edit → render → download on one machine |
| 9 | Convert `yellow_stripe_reel.py` to JSON | proves a second visual style runs on the same engine |

Parity must be locked **before** phase 6. After blending is introduced, parity loses its meaning — archive frame hashes at the end of phase 5.

## 8. Design rules

| Rule | Reason |
|---|---|
| The LLM never writes code | generated code is unpredictable and unverifiable |
| The LLM chooses only from closed lists | every output stays renderable and testable |
| Every story passes the validator before pixels are drawn | catches gaps, overlaps, bad values before wasting render time |
| Rendering is deterministic | same story → same video, which makes testing possible |
| The rig contains no story content | visuals stay reusable across all ideas |
| The API contains no drawing code | the renderer stays a CLI and stays testable without a server |
| No absolute paths; no import-time side effects | the project runs on any machine, and tests can import it |
| Parity before polish | separate the refactor from new features |
| Code in English, on-screen text in Arabic | maintainable code, Arabic content |

## 9. Known issues in the legacy scripts

These are the reasons for the refactor, and each one maps to a rule above.

- `draw_rtl_line` ignores its `highlight` argument — highlight words never color.
- `draw_app_panel(phase)` and `wrapped_rtl_lines`'s `widths` are computed and unused.
- Font paths hardcode `/home/ahmad/…` and fall back silently elsewhere.
- `IDLE` is defined twice; `BERET` / `IDLE_STATE` are dead code.
- Many magic numbers assume 1080×1920 (`540`, `1480`, `880`).
- `ff.wait()`'s return code is never checked — failed renders are silent.
- `BG2.copy()` at full supersampled size every frame is the render bottleneck.
- Import-time side effects make importing and testing slow.
- Global RNG order affects the audio: adding one sound changes the whole track.

## 10. Audio notes

Audio is synthesized with NumPy (no asset loading, except an optional voice file): a lo-fi chord loop, kick, hi-hats; SFX for whoosh, ding, glitch, impact; silence as a *range* with no music rather than a hardcoded array edit; sum, then normalize; loudness target −14 LUFS.

Two refinements worth doing before the audio is baked in:

1. **Seed per event, not per process.** Derive each event's randomness from `hash(type, t)` so adding a cue cannot change the sound of existing cues, and so audio parity can be checked per event instead of per track.
2. **`loudnorm` in two passes.** Single-pass loudnorm is approximate; the difference shows up as volume drift between videos, which is exactly what the −14 LUFS target exists to prevent.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Render time (minutes per video on CPU) | cache static layers; content-addressed render cache; measure before optimizing; more cores later |
| LLM picks poor pose/caption pairs | closed vocabularies, validation, retries, human review before render |
| One fixed visual style | acceptable for v1; templates later |
| Arabic + Latin bidi ordering | keep shaping in one module; test with mixed sentences; check `highlight` against raw text **before** reshapers/bidi |
| TTS licensing (`edge-tts` is not clearly commercial-safe) | decide before shipping narration |
| Font licensing | Cairo is OFL — ship the license file with the project |
| Scope creep | finish phases in order; don't start the UI before parity |

## 12. Open questions

| Question | What it changes |
|---|---|
| Which STT model, and what interface? | the input layer's shape (a `faster-whisper` default is assumed above) |
| One character or several? | `character` vs. `actors` in the schema — decides now, migrates later |
| 9:16 only, or other formats? | whether layout numbers must become relative |
| Which LLM (API or local)? | cost, quality, prompt length, commercial terms |
| Which TTS for narration? | licensing and the audio pipeline |
| Is `duration` always 30, or free within 10–30? | the validator's last-scene rule and what the engine does with the remainder |
| Does the last frame have to equal the first (seamless loop)? | if yes, it is a constraint on the *story*, not just the renderer |
| Product: service, SaaS, or both? | whether auth, multi-tenancy, and object storage are needed early |

## 13. Glossary

**Rig** — reusable drawing toolkit (stickman, camera, text, backgrounds).
**Pose** — named set of joint angles (`SHOCK`, `TYPE_A`, …).
**Scene** — contiguous time range with one character state, camera, and props.
**Caption / Headline** — bottom subtitle / top title track.
**Event (cue)** — instant moment: shake, punch, flash, glitch, sound.
**Validator** — schema + timeline checks (Pydantic).
**Engine** — turns a story into per-frame state; no drawing, no story content.
**Director** — the LLM step that turns a transcript into story JSON.
**Parity test** — proves two versions of the code produce identical output.
**Supersampling** — draw at 2×, downscale, for smooth edges.
**LUFS** — loudness measure; target −14.

---

### Description (320 characters, verified)

```
Voice2Reel turns a spoken idea into a short motion-graphics video. Record 10 to 30
seconds of speech, which is transcribed to text. An AI model then writes a scene plan
as validated JSON, and the engine draws the characters, UI panels, captions, and music
automatically. Stories are data. New ideas need no code changes.
```

`len(desc) == 320` — checked in Python. Any edit requires re-checking.
