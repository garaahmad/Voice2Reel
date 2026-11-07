"""Pin upstream assets and use the supported Coqui checkpoint loader."""

from pathlib import Path

MODEL_ID = "coqui/XTTS-v2"
MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
DEFAULT_SPEAKER = "Claribel Dervla"


def download_model(destination, *, training=True):
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
                "speakers_xtts.pth",
                "LICENSE.txt",
            ]
            + (["dvae.pth", "mel_stats.pth"] if training else []),
        )
    )


def available_speakers(destination):
    """List the pinned preset voices without downloading the full model."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, "speakers_xtts.pth", revision=MODEL_REVISION, local_dir=destination)
    return sorted(torch.load(path, map_location="cpu", weights_only=True))


def load_model(base_dir, checkpoint=None, device="cuda"):
    import torch
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(Path(base_dir) / "config.json"))
    model = Xtts(config)
    # The pinned original checkpoint stores these config classes at legacy paths.
    # Scope the exact allowlist so upstream can load safely without rewriting the
    # cached model.pth (which would otherwise trigger a large re-download next run).
    with torch.serialization.safe_globals(
        [
            XttsConfig,
            BaseDatasetConfig,
            (XttsArgs, "TTS.tts.models.xtts.XttsArgs"),
            (XttsAudioConfig, "TTS.tts.models.xtts.XttsAudioConfig"),
        ]
    ):
        model.load_checkpoint(
            config,
            checkpoint_dir=base_dir,
            checkpoint_path=checkpoint,
            vocab_path=str(Path(base_dir) / "vocab.json"),
            use_deepspeed=False,
        )
    # XTTS overrides eval() without returning self, unlike torch.nn.Module.
    model.to(device)
    model.eval()
    return model
