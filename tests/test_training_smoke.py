"""A real CPU forward/backward/save/reload cycle, with a tiny random Qwen2, no downloads."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments

from voice2reel.scriptwriter.data import AnswerCollator


def test_lora_optimizer_updates_and_adapter_round_trip(tmp_path):
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    base = Qwen2ForCausalLM(config)
    original_state = {k: v.clone() for k, v in base.state_dict().items()}
    model = get_peft_model(
        base,
        LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_dropout=0
        ),
    )
    before = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
    samples = [{"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5, "labels": [-100, -100, 3, 4, 5]}] * 4
    trainer = Trainer(
        model=model,
        train_dataset=samples,
        data_collator=AnswerCollator(0),
        args=TrainingArguments(
            output_dir=str(tmp_path / "run"),
            use_cpu=True,
            max_steps=2,
            per_device_train_batch_size=1,
            learning_rate=0.01,
            report_to="none",
            save_strategy="no",
        ),
    )
    result = trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    model.save_pretrained(tmp_path / "adapter")
    reloaded_base = Qwen2ForCausalLM(config)
    reloaded_base.load_state_dict(original_state)
    reloaded = PeftModel.from_pretrained(reloaded_base, tmp_path / "adapter")
    model.eval()
    reloaded.eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        assert torch.allclose(model(ids).logits, reloaded(ids).logits, atol=1e-6)
