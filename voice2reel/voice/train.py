"""Fine-tune XTTS GPT on one speaker. The vocoder is not retrained."""

import argparse
import csv
import json
from pathlib import Path

from voice2reel.common import digest_file, write_json
from voice2reel.voice.model import MODEL_ID, MODEL_REVISION, download_model


def build_config(base, data, output, epochs=5):
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsAudioConfig
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig

    dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="voice2reel_ar",
        path=str(data),
        meta_file_train="train.csv",
        meta_file_val="validation.csv",
        language="ar",
    )
    return GPTTrainerConfig(
        output_path=str(output),
        run_name="voice2reel_ar",
        project_name="voice2reel",
        datasets=[dataset],
        epochs=epochs,
        training_seed=42,
        model_args=GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=44100,
            max_wav_length=255995,
            max_text_length=200,
            debug_loading_failures=True,
            mel_norm_file=str(base / "mel_stats.pth"),
            dvae_checkpoint=str(base / "dvae.pth"),
            xtts_checkpoint=str(base / "model.pth"),
            tokenizer_file=str(base / "vocab.json"),
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        ),
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=1,
        eval_batch_size=1,
        batch_group_size=0,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        mixed_precision=True,
        precision="fp16",
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 0.01},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [10000], "gamma": 0.5},
        print_step=10,
        plot_step=100,
        save_step=100,
        save_n_checkpoints=2,
        save_checkpoints=True,
        test_sentences=[],
        dashboard_logger="tensorboard",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/voice")
    p.add_argument("--base", default="models/xtts-v2")
    p.add_argument("--output", default="outputs/voice")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--resume", help="Coqui run directory containing config and checkpoints")
    args = p.parse_args()
    if args.epochs < 1 or args.grad_accum < 1:
        p.error("epochs and grad-accum must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("XTTS training requires a GPU runtime")
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    report = json.loads((data / "report.json").read_text())
    if report["total_minutes"] < 20:
        print(
            "Small dataset: this run tests adaptation, not production voice quality. Aim for 20–60+ clean minutes."
        )
    base = download_model(args.base).resolve()
    tokenizer = VoiceBpeTokenizer(str(base / "vocab.json"))
    # Upstream otherwise skips oversized samples: fail before allocating the model.
    for split in ("train", "validation", "test"):
        with (data / f"{split}.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="|"):
                tokens = tokenizer.encode(row["text"], lang="ar")
                if len(tokens) > 200 or not tokens:
                    raise ValueError(f"Invalid transcript token length: {row['audio_file']}")
                if not (data / row["audio_file"]).is_file():
                    raise ValueError(f"Missing audio: {row['audio_file']}")
    config = build_config(base, data, output, args.epochs)
    train, validation = load_tts_samples(config, eval_split=True)
    metadata = {
        "base_model": MODEL_ID,
        "revision": MODEL_REVISION,
        "args": vars(args),
        "dataset_sha256": digest_file(data / "report.json"),
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.resume:
        previous = json.loads((Path(args.resume) / "voice2reel_run.json").read_text())
        if previous["dataset_sha256"] != metadata["dataset_sha256"]:
            raise ValueError("Dataset changed; start a new run")
    model = GPTTrainer(config)
    trainer = Trainer(
        TrainerArgs(continue_path=args.resume or "", grad_accum_steps=args.grad_accum, start_with_eval=True),
        config,
        output_path=str(output),
        model=model,
        train_samples=train,
        eval_samples=validation,
        parse_command_line_args=False,
    )
    write_json(Path(trainer.output_path) / "voice2reel_run.json", metadata)
    trainer.fit()
    print(f"Run saved to {trainer.output_path}. Evaluate best_model.pth against the base model.")


if __name__ == "__main__":
    main()
