"""Recipe config + dataset rendering tests (no model loaded)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lora_recipes import load_config
from lora_recipes.config import DatasetConfig, TaskConfig
from lora_recipes.data import render_example

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("yaml_name", ["smollm_sst2_cpu.yaml", "qwen_sst2_gpu.yaml"])
def test_each_config_loads_with_required_fields(yaml_name: str) -> None:
    cfg = load_config(CONFIGS / yaml_name)
    assert cfg.base_model
    assert cfg.dataset.train_size > 0
    assert cfg.dataset.eval_size > 0
    assert cfg.task.label_tokens
    assert cfg.training.lora_r >= 1
    assert cfg.output.dir


def test_render_example_applies_templates() -> None:
    ds = DatasetConfig(
        name="x", split_train="train", split_eval="validation",
        train_size=10, eval_size=10, text_field="sentence", label_field="label",
        label_names=["negative", "positive"],
    )
    task = TaskConfig(
        type="binary_classification",
        prompt_template="Review: {text}\nSentiment:",
        target_template=" {label}",
        label_tokens=[" negative", " positive"],
    )
    out = render_example({"sentence": "this is great", "label": 1}, task, ds)
    assert out["prompt"] == "Review: this is great\nSentiment:"
    assert out["target"] == " positive"
    assert out["label_idx"] == 1
    assert out["label_name"] == "positive"


def test_render_example_negative_label() -> None:
    ds = DatasetConfig(
        name="x", split_train="train", split_eval="validation",
        train_size=10, eval_size=10, text_field="sentence", label_field="label",
        label_names=["negative", "positive"],
    )
    task = TaskConfig(
        type="binary_classification",
        prompt_template="Review: {text}\nSentiment:",
        target_template=" {label}",
        label_tokens=[" negative", " positive"],
    )
    out = render_example({"sentence": "this sucks", "label": 0}, task, ds)
    assert out["label_idx"] == 0
    assert out["label_name"] == "negative"
    assert out["target"] == " negative"


def test_smollm_config_targets_cpu_friendly_settings() -> None:
    cfg = load_config(CONFIGS / "smollm_sst2_cpu.yaml")
    assert cfg.training.bf16 is False
    assert cfg.training.fp16 is False
    assert cfg.training.lora_r <= 16
    assert cfg.dataset.train_size <= 1000  # CPU-feasible


def test_gpu_config_uses_bf16_and_richer_target_modules() -> None:
    cfg = load_config(CONFIGS / "qwen_sst2_gpu.yaml")
    assert cfg.training.bf16 is True
    assert "k_proj" in cfg.training.target_modules
    assert "o_proj" in cfg.training.target_modules
