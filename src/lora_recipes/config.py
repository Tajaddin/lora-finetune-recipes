"""Recipe configuration loader.

Loads YAML into a typed Pydantic-free dataclass tree so the recipes are
inspectable from Python without an extra runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str
    split_train: str
    split_eval: str
    train_size: int
    eval_size: int
    text_field: str
    label_field: str
    label_names: list[str]


@dataclass
class TaskConfig:
    type: str
    prompt_template: str
    target_template: str
    label_tokens: list[str]


@dataclass
class TrainingConfig:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-4
    warmup_ratio: float = 0.06
    max_steps: int = 0
    max_seq_length: int = 128
    bf16: bool = False
    fp16: bool = False
    seed: int = 7
    logging_steps: int = 10
    save_total_limit: int = 1


@dataclass
class OutputConfig:
    dir: str


@dataclass
class RecipeConfig:
    base_model: str
    dataset: DatasetConfig
    task: TaskConfig
    training: TrainingConfig
    output: OutputConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> RecipeConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RecipeConfig(
        base_model=raw["base_model"],
        dataset=DatasetConfig(**raw["dataset"]),
        task=TaskConfig(**raw["task"]),
        training=TrainingConfig(**raw["training"]),
        output=OutputConfig(**raw["output"]),
        raw=raw,
    )
