"""Pin upstream assets and use the supported Coqui checkpoint loader."""

from pathlib import Path

MODEL_ID = "coqui/XTTS-v2"
MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"


def download_model(destination):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=destination,
            allow_patterns=[
                "model.pth",
                "config.json",
                "vocab.json",
                "dvae.pth",
                "mel_stats.pth",
                "LICENSE.txt",
            ],
        )
    )


def load_model(base_dir, checkpoint=None, device="cuda"):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(Path(base_dir) / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=base_dir,
        checkpoint_path=checkpoint,
        vocab_path=str(Path(base_dir) / "vocab.json"),
        use_deepspeed=False,
    )
    return model.to(device).eval()
