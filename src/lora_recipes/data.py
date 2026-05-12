"""Render the supervised-classification dataset into prompt/target token IDs."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from lora_recipes.config import DatasetConfig, TaskConfig


def load_split(ds_cfg: DatasetConfig, split: str, size: int) -> Dataset:
    """Load ``split`` from a HF dataset, take the first ``size`` examples."""
    ds = load_dataset(ds_cfg.name, split=split)
    if size and size < len(ds):
        ds = ds.select(range(size))
    return ds


def render_example(ex: dict[str, Any], task: TaskConfig, ds_cfg: DatasetConfig) -> dict[str, str]:
    """Apply ``prompt_template`` + ``target_template`` to one example."""
    text = ex[ds_cfg.text_field]
    label_idx = int(ex[ds_cfg.label_field])
    label_name = ds_cfg.label_names[label_idx]
    prompt = task.prompt_template.format(text=text).strip()
    target = task.target_template.format(label=label_name)
    return {"prompt": prompt, "target": target, "label_idx": label_idx, "label_name": label_name, "text": text}


def to_supervised_pairs(ds: Dataset, task: TaskConfig, ds_cfg: DatasetConfig) -> Dataset:
    """Map each row to ``{prompt, target, label_idx, label_name, text}``."""
    return ds.map(
        lambda ex: render_example(ex, task, ds_cfg),
        remove_columns=ds.column_names,
    )
