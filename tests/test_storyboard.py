import copy
import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from voice2reel.realistic import assemble, command, probe, validate_settings
from voice2reel.director import Narration, clean_response, validate_narration_language
from voice2reel.scriptwriter.schema import Script
from voice2reel.storyboard import Brief, Plan, QUESTIONS, build_timeline


def assets():
    plan = Plan(title="الشرط", continuity="A programmer at a desk.", shots=[{
        "purpose": "Explain a conditional", "visual_prompt": "A programmer studies a laptop, slow push-in.",
        "opening_action": "The programmer looks at the screen.", "closing_action": "The programmer nods.",
    }] * 3)
    script = Script(schema_version=1, title=plan.title, language="ar", target_duration_seconds=15,
                    beats=[{"narration": "راجع شرط البرنامج قبل تشغيله مرة أخرى.", "caption": "راجع الشرط",
                            "pose": "IDLE", "camera": "medium"}] * 3)
    metadata = {"language": "ar", "duration": 1.5, "segments": [
        {"text": beat.narration, "start": i * .5, "end": (i + 1) * .5}
        for i, beat in enumerate(script.beats)
    ]}
    return plan, script, metadata


def test_brief_has_five_questions_and_optional_reference():
    assert len(QUESTIONS) == 5
    brief = Brief(idea="A programmer", audience="Beginners", setting="Office", message="Review conditions")
    assert brief.reference_video is None
    assert brief.shot_count == 4
    with pytest.raises(ValueError):
        Brief(**{**brief.model_dump(), "language": "fr"})


def test_reject_wrong_language_in_caption_and_video_prompt():
    narration = Narration(title="الشرط", beats=[
        {"narration": "راجع الشرط قبل التشغيل", "caption": "Wrong language"}
    ] * 3)
    with pytest.raises(ValueError, match="Arabic"):
        validate_narration_language(narration, "ar")
    plan, _, _ = assets()
    data = plan.model_dump()
    data["shots"][0]["visual_prompt"] = "شخص أمام شاشة"
    with pytest.raises(ValueError):
        Plan.model_validate(data)


def test_json_syntax_repair_keeps_text_and_still_requires_schema():
    text, repaired = clean_response('```json\n{"title": "عنوان", "continuity": "A person at a desk"\n```')
    assert repaired
    assert json.loads(text) == {"title": "عنوان", "continuity": "A person at a desk"}
    with pytest.raises(ValueError):
        Plan.model_validate_json(text)  # Missing shots cannot bypass the schema.
    valid = '{"text": "if (x); — راجع الشرط"}'
    assert clean_response(valid) == (valid, False)


def test_real_speech_timing_drives_frame_counts_and_split_limits():
    plan, script, metadata = assets()
    metadata["duration"] = 27.789
    for i, segment in enumerate(metadata["segments"]):
        segment["start"] = i * 9.263
        segment["end"] = (i + 1) * 9.263
    timeline = build_timeline(plan, script, metadata)
    assert len(timeline["clips"]) == 6
    assert timeline["total_frames"] == round(27.789 * 24)
    cursor = 0
    for clip in timeline["clips"]:
        assert clip["start_frame"] == cursor
        assert (clip["num_frames"] - 1) % 8 == 0
        assert clip["end_frame"] - cursor <= 120
        assert clip["num_frames"] >= clip["end_frame"] - cursor
        cursor = clip["end_frame"]


def test_reject_stale_audio_and_mismatched_plan():
    plan, script, metadata = assets()
    bad = copy.deepcopy(metadata)
    bad["segments"][0]["text"] = "stale script"
    with pytest.raises(ValueError, match="differs from script"):
        build_timeline(plan, script, bad)
    plan.shots.append(plan.shots[0])
    with pytest.raises(ValueError, match="one-to-one"):
        build_timeline(plan, script, metadata)


@pytest.mark.parametrize("settings", [(300, 512, 40), (288, 512, 4), (288, 2048, 40)])
def test_reject_incompatible_video_settings(settings):
    with pytest.raises(ValueError):
        validate_settings(*settings)


def test_real_clip_assembly_with_arabic_captions_and_audio(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    plan, script, metadata = assets()
    timeline = build_timeline(plan, script, metadata)
    clips = tmp_path / "clips"
    clips.mkdir()
    for i, color in enumerate(["red", "green", "blue"]):
        command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=288x512:r=24:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", clips / f"{i:03d}.mp4")
    wav = tmp_path / "voice.wav"
    sf.write(wav, np.sin(np.arange(36000) * .03) * .1, 24000)
    output = assemble(tmp_path, timeline, script, wav, tmp_path / "out.mp4", 288, 512)
    info = probe(output)
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    assert abs(float(info["format"]["duration"]) - 1.5) < .06
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 36
    # A source shorter than the speech must fail instead of freezing a frame.
    timeline["clips"][0]["end_frame"] = 100
    with pytest.raises(ValueError, match="too short"):
        assemble(tmp_path, timeline, script, wav, tmp_path / "bad.mp4", 288, 512)


def test_cpu_rejected_before_download(monkeypatch):
    import torch
    from voice2reel.realistic import generate

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no weights were downloaded"):
        generate("nonexistent-project")


@pytest.mark.parametrize("conditioned", [False, True])
def test_ltx_real_api_with_tiny_random_weights(conditioned):
    """Offline API test only: random tiny weights say nothing about visual quality."""
    import torch
    from PIL import Image

    diffusers = pytest.importorskip("diffusers")
    transformer = diffusers.LTXVideoTransformer3DModel(
        in_channels=8, out_channels=8, num_attention_heads=4, attention_head_dim=8,
        cross_attention_dim=32, num_layers=1, caption_channels=32,
    )
    vae = diffusers.AutoencoderKLLTXVideo(
        latent_channels=8, block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8),
        layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1, 1),
        spatio_temporal_scaling=(True, True, False, False),
        decoder_spatio_temporal_scaling=(True, True, False, False),
        decoder_inject_noise=(False,) * 5, upsample_residual=(False,) * 4,
        upsample_factor=(1,) * 4, timestep_conditioning=False, patch_size=1, patch_size_t=1,
    )
    pipe = diffusers.LTXConditionPipeline(
        transformer=transformer, vae=vae, scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
        tokenizer=None, text_encoder=None,
    )
    pipe.set_progress_bar_config(disable=True)
    frames = pipe(
        prompt_embeds=torch.randn(1, 8, 32), negative_prompt_embeds=torch.zeros(1, 8, 32),
        prompt_attention_mask=torch.ones(1, 8, dtype=torch.bool),
        negative_prompt_attention_mask=torch.ones(1, 8, dtype=torch.bool),
        image=Image.new("RGB", (32, 32), "gray") if conditioned else None,
        width=32, height=32, num_frames=9, frame_rate=24, num_inference_steps=2,
        guidance_scale=3.0, decode_timestep=.05, decode_noise_scale=.025,
        image_cond_noise_scale=.025, output_type="np",
    ).frames[0]
    assert frames.shape == (9, 32, 32, 3)
    assert np.isfinite(frames).all()
    json.dumps({"frames": len(frames)})


def test_ltx_text_encoder_only_stage_without_video_weights():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

    diffusers = pytest.importorskip("diffusers")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    )
    encoder = T5EncoderModel(T5Config(vocab_size=3, d_model=32, d_ff=64, num_layers=1, num_heads=4))
    pipe = diffusers.LTXConditionPipeline(
        transformer=None, vae=None, text_encoder=encoder, tokenizer=tokenizer,
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
    )
    prompt, mask, negative, negative_mask = pipe.encode_prompt(
        "A realistic office", negative_prompt="cartoon", device=torch.device("cpu"), max_sequence_length=8,
    )
    assert prompt.shape == negative.shape == (1, 8, 32)
    assert mask.shape == negative_mask.shape == (1, 8)
    assert mask.dtype == torch.bool
    assert torch.isfinite(prompt).all()
