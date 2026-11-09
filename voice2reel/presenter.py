"""One text-model pass → speech and sound design → 2D presenter MP4, with independent caches."""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from voice2reel.arabic_text import resolve_scene_texts, speech_text
from voice2reel.common import digest_file, write_json
from voice2reel.director_model import DIRECTOR_MODEL, DIRECTOR_REVISION, GGUF_REVISION
from voice2reel.fast_voice import REVISION as VOICE_REVISION
from voice2reel.voice.model import MODEL_REVISION as XTTS_REVISION
from voice2reel.presenter_schema import PresenterScript, ReelRequest, parse_script
from voice2reel.presenter_render import VERSION as RENDER_VERSION, render, validate_audio

PROMPT_VERSION = "presenter-4-mixed-arabic-tts"


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached(path, key, files):
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
        return manifest["key"] == key and all(
            p.is_file() and manifest["files"][p.name] == digest_file(p) for p in files
        )
    except (ValueError, KeyError, OSError, TypeError):
        return False


def manifest(path, key, files):
    write_json(path, {"key": key, "files": {p.name: digest_file(p) for p in files}})


def load_writer(device, gguf_model=None, threads=2):
    if device == "auto":
        try:
            import torch
        except ImportError:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        from voice2reel.director_model import GGUFDirector

        return GGUFDirector(gguf_model, threads)
    from voice2reel.scriptwriter.generate import ScriptGenerator

    return ScriptGenerator(device="cuda", model_id=DIRECTOR_MODEL, model_revision=DIRECTOR_REVISION)


def generate_script(writer, request, trace_path, attempts=1):
    from voice2reel.director import Narration, clean_response, validate_narration_language

    language = request["language"]
    instructions = (
        "اكتب تعليقًا لفيديو شرح بالعربية الفصحى البسيطة، بصوت إنسان يشرح لصديقه. "
        "لا لهجة عامية ولا تعابير مترجمة حرفيًا. ابدأ بمشكلة محددة من الفكرة، ثم اشرح سببها، ثم خطوة عملية. "
        "جمل قصيرة متصلة المعنى. لا تكتب مقدمة أو ترحيبًا أو دعوة للمتابعة. "
        "تجنب عبارات: في عالمنا اليوم، هيا بنا، اكتشف السر، تعزيز الفهم، ضمان تثبيت المعرفة. "
        "لا وعود مضمونة ولا تهويل. لا تكرر المعلومة في المشاهد الثلاثة. "
        "اجعل العناوين عربية قصيرة ومحددة، والنص مناسبًا للكلام بصوت عالٍ. "
        "المصطلحات التقنية المعروفة مثل for loop و API مسموحة في النص المعروض مع القراءة العربية لها في الكلام. "
        "ممنوع معرفات الكود الحرفية المنفردة مثل if أو console داخل الكلام؛ اشرحها بالعربية. "
        "تسميات اللوحات أشياء أو خطوات واضحة من الفكرة، لا صفات مبهمة مثل محدد أو إجباري. "
        "التزم بالحقائق المعطاة، ولا تصف حركة الشخصية في الكلام المنطوق. لا تقرأ كودًا حرفيًا. "
        if language == "ar"
        else "Write a concise English explainer. Follow the supplied facts; do not invent claims. "
        "Narration teaches the topic, not the character movements. Do not read code aloud. "
    )
    if request.get("facts", "").strip():
        instructions += (
            "أعد صياغة الحقائق المقدمة فقط؛ لا تضف أمثلة أو تفاصيل أو أرقامًا من عندك. "
            if language == "ar"
            else "Only rephrase the supplied facts. Do not add examples, details or statistics. "
        )
    instructions += (
        "Return one JSON object, with title and exactly 3 scenes. "
        "Each scene has ONLY narration (one or two short spoken sentences), caption (short heading), "
        "and labels (two short labels for a diagram). Do not write gesture, emotion or camera fields: "
        "the renderer compiles those. Do not generate code. "
        f"Aim for about {max(6, round(request.get('duration', 20) * 1.9 / 3))} spoken words per scene; "
        "write one or two short connected sentences per scene. The duration is a target, not a guarantee. "
        "Registered technical phrases such as 'for loop' are allowed in display narration; "
        "never output bare code identifiers such as if, for or console."
    )
    base = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
    ]
    if language == "ar":
        example_request = {
            "idea": "الصفحة تعرض نسخة قديمة",
            "facts": "التخزين المؤقت قد يعرض نسخة محفوظة. جرّب التحديث مع تجاوز الذاكرة المؤقتة.",
        }
        example_response = {
            "title": "لماذا تظهر الصفحة القديمة؟",
            "scenes": [
                {
                    "narration": "حدّثت الصفحة، لكنها ما زالت تعرض النسخة القديمة؟ قد تكون النسخة المحفوظة هي السبب.",
                    "caption": "لماذا لم تظهر التغييرات؟",
                    "labels": ["نسخة قديمة", "نسخة جديدة"],
                },
                {
                    "narration": "التخزين المؤقت يحتفظ بنسخة من الصفحة، فيعرضها دون تنزيلها من جديد.",
                    "caption": "التطبيق يعرض نسخة محفوظة",
                    "labels": ["حفظ الصفحة", "عرض النسخة"],
                },
                {
                    "narration": "جرّب تحديث الصفحة مع تجاوز النسخة المحفوظة، ثم تحقق من ظهور التغييرات.",
                    "caption": "حدّث الصفحة وتحقق",
                    "labels": ["تحديث الصفحة", "فحص النتيجة"],
                },
            ],
        }
        base = [
            base[0],
            {"role": "user", "content": json.dumps(example_request, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(example_response, ensure_ascii=False)},
            base[1],
        ]
    messages = base
    trace = []
    for attempt in range(attempts):
        started = time.perf_counter()
        raw = writer.complete(messages, max_tokens=650)
        row = {"attempt": attempt + 1, "raw_response": raw, "wall_seconds": time.perf_counter() - started}
        trace.append(row)
        try:
            clean, repaired = clean_response(raw)
            value = json.loads(clean)
            value["language"] = language
            if isinstance(value.get("scenes"), list):
                for i, scene in enumerate(value["scenes"]):
                    if not isinstance(scene, dict):
                        raise ValueError("Scenes must be objects")
                    labels = scene.pop("labels", [])
                    scene.setdefault("gesture", ("think", "point", "thumbs_up")[i % 3])
                    scene.setdefault("emotion", ("thinking", "concerned", "happy")[i % 3])
                    scene.setdefault("camera", "medium" if i == 2 else "wide")
                    scene.setdefault(
                        "visual",
                        {"kind": ("flow" if i == 1 else "list") if labels else "none", "items": labels},
                    )
            # Mixed *display* text is allowed for registered technical terms; the
            # speech engine only ever sees the Arabic-only ``spoken`` reading.
            value = resolve_scene_texts(value)
            script = PresenterScript.model_validate(value)
            if language == "ar" and any(
                any(c.isascii() and c.isalpha() for c in speech_text(s)) for s in script.scenes
            ):
                raise ValueError("الكلام المرسل لمحرك الصوت يجب أن يكون عربيًا كاملًا.")
            if len(script.scenes) != 3:
                raise ValueError("Exactly 3 scenes required")
            validate_narration_language(
                Narration(
                    title=script.title,
                    beats=[{"narration": s.narration, "caption": s.caption} for s in script.scenes],
                ),
                language,
            )
        except (ValueError, TypeError) as exc:
            row["error"] = str(exc)
            write_json(trace_path, trace)
            messages = base + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "صحح الأخطاء في الرد السابق. الكلام المنطوق فصحى طبيعية، "
                        "بجمل سليمة نحويًا. المصطلحات التقنية المسجلة مسموحة، "
                        "لكن لا تضع معرفات كود حرفية منفردة في الكلام. "
                        "حافظ على المصطلحات الدقيقة في الحقائق ولا تختصرها اختصارًا يغيّر المعنى. "
                        "أعد JSON المصحح فقط. "
                        if language == "ar"
                        else "Correct the previous response. Return JSON only. "
                    )
                    + str(exc)[:350],
                },
            ]
        else:
            row["json_repaired"] = repaired
            write_json(trace_path, trace)
            return script
    raise ValueError(f"Text model failed validation; see {trace_path}. Use --script with reviewed text.")


def _run(
    project,
    *,
    script_path=None,
    request=None,
    device="auto",
    gguf_model=None,
    threads=2,
    profile="fast",
    voice_engine="edge",
    voice_dir="models/piper",
    reference=None,
    speaker=None,
    xtts_base="models/xtts-v2",
    text_attempts=2,
    neural_voice=None,
    speech_rate=0,
    music=True,
    sfx=True,
    music_path=None,
):
    if profile not in ("fast", "quality") or not 1 <= threads <= 32 or text_attempts not in (1, 2):
        raise ValueError("Choose fast/quality, 1–32 threads, and 1–2 text attempts")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Choose auto, cpu or cuda")
    if not isinstance(speech_rate, int) or not -20 <= speech_rate <= 25:
        raise ValueError("Speech rate must be between -20 and +25 percent")
    if request is not None:
        request = ReelRequest.model_validate(request).model_dump()
    if reference and speaker:
        raise ValueError("Choose a preset speaker or reference voice, not both")
    if voice_engine not in ("edge", "piper", "xtts"):
        raise ValueError("Choose edge, piper or xtts")
    if voice_engine != "xtts" and (reference or speaker):
        raise ValueError("Reference cloning and --speaker require --voice-engine xtts")
    if voice_engine != "edge" and (neural_voice is not None or speech_rate != 0):
        raise ValueError("--neural-voice and --speech-rate require --voice-engine edge")
    if music_path and (not music or not Path(music_path).is_file()):
        raise ValueError("Custom music must be an existing file with music enabled")
    if (script_path is None) == (request is None):
        raise ValueError("Choose a script or a written idea")
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metrics = {"stages": {}, "cache_hits": {}}

    def status(stage):
        print(stage, flush=True)
        write_json(
            project / "progress.json", {"stage": stage, "elapsed_seconds": time.perf_counter() - started}
        )

    source = project / "script.json"
    audio = project / "narration.wav"
    timing = project / "narration.json"
    output = project / "video.mp4"
    try:
        status("Preparing script")
        stage = time.perf_counter()
        if script_path:
            script = parse_script(json.loads(Path(script_path).read_text(encoding="utf-8")))
            metrics["cache_hits"]["script"] = False
            metrics["script_source"] = "provided"
        else:
            write_json(project / "request.json", request)
            key = signature(
                {
                    "request": request,
                    "prompt": PROMPT_VERSION,
                    "pipeline_code": digest_file(Path(__file__)),
                    "writer_code": digest_file(Path(__file__).with_name("director_model.py")),
                    "schema_code": digest_file(Path(__file__).with_name("presenter_schema.py")),
                    "arabic_code": digest_file(Path(__file__).with_name("arabic_text.py")),
                    "lexicon": digest_file(Path(__file__).parent / "assets" / "arabic_lexicon.json"),
                    "base": DIRECTOR_REVISION,
                    "gguf": GGUF_REVISION,
                    "attempts": text_attempts,
                }
            )
            if cached(project / "script.cache.json", key, [source]):
                script = parse_script(json.loads(source.read_text()))
                metrics["cache_hits"]["script"] = True
                metrics["script_source"] = "cached_ai"
            else:
                status("Writing scenes in one AI pass")
                writer = load_writer(device, gguf_model, threads)
                try:
                    script = generate_script(writer, request, project / "script.trace.json", text_attempts)
                    write_json(
                        project / "text_model.json",
                        {
                            "model": writer.model_id,
                            "revision": writer.revision,
                            "backend": writer.backend,
                            "attempts": len(json.loads((project / "script.trace.json").read_text())),
                        },
                    )
                finally:
                    if getattr(writer, "backend", "").startswith("llama.cpp"):
                        writer.model.close()
                    del writer
                    gc.collect()
                    if device != "cpu":
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                write_json(source, script.model_dump())
                manifest(project / "script.cache.json", key, [source])
                metrics["cache_hits"]["script"] = False
                metrics["script_source"] = "ai"
        write_json(source, script.model_dump())
        metrics["stages"]["script_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        if voice_engine == "edge":
            from voice2reel.neural_voice import NeuralVoice, VERSION as NEURAL_VERSION, validate_settings

            selected_voice = validate_settings(script.language, neural_voice, speech_rate)
            voice_code = Path(__file__).with_name("neural_voice.py")
            voice_revision = NEURAL_VERSION
        else:
            selected_voice = None
            voice_code = (
                Path(__file__).with_name("fast_voice.py")
                if voice_engine == "piper"
                else Path(__file__).parent / "voice" / "synthesize.py"
            )
            voice_revision = VOICE_REVISION if voice_engine == "piper" else XTTS_REVISION
        voice_key = signature(
            {
                "texts": [speech_text(s) for s in script.scenes],
                "language": script.language,
                "engine": voice_engine,
                "voice_code": digest_file(voice_code),
                "arabic_code": digest_file(Path(__file__).with_name("arabic_text.py")),
                "lexicon": digest_file(Path(__file__).parent / "assets" / "arabic_lexicon.json"),
                "revision": voice_revision,
                "reference": digest_file(reference) if reference else None,
                "speaker": speaker,
                "neural_voice": selected_voice,
                "speech_rate": speech_rate,
            }
        )
        if cached(project / "voice.cache.json", voice_key, [audio, timing]):
            metadata = json.loads(timing.read_text())
            validate_audio(script, metadata, audio)
            metrics["cache_hits"]["voice"] = True
        else:
            status(
                "Generating online neural narration"
                if voice_engine == "edge"
                else "Generating local AI narration"
            )
            if voice_engine == "edge":
                metadata = NeuralVoice(script.language, selected_voice, speech_rate).synthesize(script, audio)
            elif voice_engine == "piper":
                from voice2reel.fast_voice import FastVoice

                metadata = FastVoice(script.language, voice_dir, threads).synthesize(script, audio)
            else:
                words = sum(len(speech_text(s).split()) for s in script.scenes)
                legacy = {
                    "schema_version": 1,
                    "title": script.title,
                    "language": script.language,
                    "target_duration_seconds": min(60, max(10, round(words / 2))),
                    "beats": [
                        {"narration": speech_text(s), "caption": s.caption, "pose": "IDLE", "camera": "medium"}
                        for s in script.scenes
                    ],
                }
                legacy_path = project / "xtts-script.json"
                write_json(legacy_path, legacy)
                args = [
                    sys.executable,
                    "-m",
                    "voice2reel.voice.synthesize",
                    "--script",
                    str(legacy_path),
                    "--device",
                    device,
                    "--base",
                    str(xtts_base),
                    "--output",
                    str(audio),
                ]
                if reference:
                    args += ["--reference", str(reference)]
                elif speaker:
                    args += ["--speaker", speaker]
                subprocess.run(args, check=True)
                metadata = json.loads(timing.read_text())
            validate_audio(script, metadata, audio)
            manifest(project / "voice.cache.json", voice_key, [audio, timing])
            metrics["cache_hits"]["voice"] = False
        metrics["stages"]["voice_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        from voice2reel.audio_mix import VERSION as MIX_VERSION, mix_audio

        soundtrack = project / "soundtrack.wav"
        mix_report = project / "soundtrack.json"
        mix_key = signature(
            {
                "audio": digest_file(audio),
                "metadata": digest_file(timing),
                "music": music,
                "sfx": sfx,
                "music_file": digest_file(music_path) if music_path else None,
                "version": MIX_VERSION,
                "code": digest_file(Path(__file__).with_name("audio_mix.py")),
            }
        )
        if cached(project / "mix.cache.json", mix_key, [soundtrack, mix_report]):
            metrics["cache_hits"]["mix"] = True
        else:
            status("Mixing speech, music and transition effects")
            metrics["mix"] = mix_audio(
                audio, metadata, soundtrack, music=music, sfx=sfx, music_path=music_path
            )
            manifest(project / "mix.cache.json", mix_key, [soundtrack, mix_report])
            metrics["cache_hits"]["mix"] = False
        metrics["stages"]["mix_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        video_key = signature(
            {
                "script": script.model_dump(),
                "audio": digest_file(audio),
                "soundtrack": digest_file(soundtrack),
                "timings": digest_file(timing),
                "profile": profile,
                "renderer": RENDER_VERSION,
                "renderer_code": digest_file(Path(__file__).with_name("presenter_render.py")),
                "font": digest_file(Path(__file__).parent / "assets" / "Cairo.ttf"),
            }
        )
        if cached(project / "video.cache.json", video_key, [output]):
            metrics["cache_hits"]["video"] = True
        else:
            status("Drawing and encoding cartoon scenes")
            metrics["render"] = render(script, audio, metadata, output, profile, soundtrack_path=soundtrack)
            manifest(project / "video.cache.json", video_key, [output])
            metrics["cache_hits"]["video"] = False
        metrics["stages"]["video_seconds"] = time.perf_counter() - stage
        metrics.update(
            total_seconds=time.perf_counter() - started,
            video_duration=metadata["duration"],
            target_duration_seconds=request["duration"] if request else None,
            profile=profile,
            voice_engine=voice_engine,
            synthetic_narration=True,
            subtitle_timing="provider word boundaries"
            if metadata.get("word_timing")
            else "estimated phrase pacing",
            music=music,
            transition_effects=sfx,
            mouth_motion="audio energy, not phoneme alignment",
        )
        write_json(project / "metrics.json", metrics)
        status("Complete")
        return output, metrics
    except BaseException as exc:
        write_json(
            project / "progress.json",
            {"stage": "failed", "error": str(exc), "elapsed_seconds": time.perf_counter() - started},
        )
        raise


def run(project, **options):
    """One writer per job directory; the OS releases the lock if a process dies."""
    import fcntl

    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    with (project / ".job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another job is running in this output directory") from exc
        started = time.perf_counter()
        try:
            return _run(project, **options)
        except BaseException as exc:
            write_json(
                project / "progress.json",
                {"stage": "failed", "error": str(exc), "elapsed_seconds": time.perf_counter() - started},
            )
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--idea")
    source.add_argument("--script")
    p.add_argument("--facts", default="")
    p.add_argument("--audience", default="General audience")
    p.add_argument("--language", choices=["ar", "en"], default="ar")
    p.add_argument("--duration", type=int, default=20)
    p.add_argument("--visual-focus", default="Clear cartoon presenter with explanatory labels")
    p.add_argument("--interactive", action="store_true", help="Ask the five brief questions")
    p.add_argument("--output", default="outputs/presenter")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gguf-model")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--profile", choices=["fast", "quality"], default="fast")
    p.add_argument("--voice-engine", choices=["edge", "piper", "xtts"], default="edge")
    p.add_argument("--voice-dir", default="models/piper")
    p.add_argument("--neural-voice", help="Matching locale voice, e.g. ar-JO-TaimNeural")
    p.add_argument("--speech-rate", type=int, default=0, help="Online speech speed, -20 to +25 percent")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--no-sfx", action="store_true")
    p.add_argument("--music", dest="music_path", help="Optional music file you are allowed to use")
    p.add_argument("--reference")
    p.add_argument("--speaker")
    p.add_argument("--xtts-base", default="models/xtts-v2")
    p.add_argument("--text-attempts", type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    if a.interactive:
        if not a.idea:
            p.error("--interactive requires --idea")
        a.audience = input("لمن الفيديو؟ ").strip()
        a.duration = int(input("المدة بالثواني، 10–60؟ "))
        a.language = input("لغة التعليق: ar أم en؟ ").strip()
        a.facts = input("ما الرسالة أو الحقائق التي يجب الالتزام بها؟ ").strip()
        a.visual_focus = input("ما العناصر أو الإيماءات التي تريد إبرازها؟ ").strip()
    if not 10 <= a.duration <= 60 or a.language not in ("ar", "en") or not 1 <= a.threads <= 32:
        p.error("Use duration 10–60, ar/en, threads 1–32")
    request = None
    if a.idea:
        if not a.idea.strip() or len(a.idea) > 2000 or len(a.facts) > 2000 or len(a.visual_focus) > 600:
            p.error("Keep idea/facts within 2000 and visual focus within 600 characters")
        request = {
            "idea": a.idea,
            "facts": a.facts,
            "language": a.language,
            "duration": a.duration,
            "audience": a.audience,
            "visual_focus": a.visual_focus,
        }
    result, metrics = run(
        a.output,
        script_path=a.script,
        request=request,
        device=a.device,
        gguf_model=a.gguf_model,
        threads=a.threads,
        profile=a.profile,
        voice_engine=a.voice_engine,
        voice_dir=a.voice_dir,
        reference=a.reference,
        speaker=a.speaker,
        xtts_base=a.xtts_base,
        text_attempts=a.text_attempts,
        neural_voice=a.neural_voice,
        speech_rate=a.speech_rate,
        music=not a.no_music,
        sfx=not a.no_sfx,
        music_path=a.music_path,
    )
    print(result)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
