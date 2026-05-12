"""LoRA SFT trainer driven by a single YAML recipe.

The training loop is intentionally a thin wrapper over Hugging Face's
``Trainer``. The recipe is the contract — the code is just a glue layer that
hands the recipe to ``transformers`` and ``peft``.

Usage::

    lora-train --config configs/smollm_sst2_cpu.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from lora_recipes.config import RecipeConfig, load_config
from lora_recipes.data import load_split, to_supervised_pairs


def _tokenize_for_sft(tok, prompt: str, target: str, max_seq_length: int) -> dict:
    """Build (input_ids, labels) where loss is only over target tokens."""
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    full = prompt_ids + target_ids
    if len(full) > max_seq_length:
        full = full[-max_seq_length:]  # keep tail (target end)
        # If we truncated into the prompt, leave labels masked at front.
    labels = [-100] * len(prompt_ids) + target_ids
    if len(labels) > max_seq_length:
        labels = labels[-max_seq_length:]
    # Pad/truncate to consistent length later via the collator.
    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


def _prepare_dataset(cfg: RecipeConfig, tok) -> Dataset:
    raw = load_split(cfg.dataset, cfg.dataset.split_train, cfg.dataset.train_size)
    pairs = to_supervised_pairs(raw, cfg.task, cfg.dataset)
    max_len = cfg.training.max_seq_length

    def _mapper(ex):
        return _tokenize_for_sft(tok, ex["prompt"], ex["target"], max_len)

    return pairs.map(_mapper, remove_columns=pairs.column_names)


class _PaddingCollator:
    """Pads (input_ids, labels, attention_mask) to the max length in the batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_token_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}


def train(cfg: RecipeConfig) -> dict:
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lora-train] base_model={cfg.base_model}")
    print(f"[lora-train] output_dir={output_dir}")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (cfg.training.bf16 and torch.cuda.is_available()) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=cfg.training.target_modules,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable, total = _count_params(model)
    print(f"[lora-train] trainable_params={trainable:,} / {total:,} ({trainable / total:.4%})")

    train_ds = _prepare_dataset(cfg, tok)
    print(f"[lora-train] train_examples={len(train_ds)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        max_steps=cfg.training.max_steps if cfg.training.max_steps > 0 else -1,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        seed=cfg.training.seed,
        logging_steps=cfg.training.logging_steps,
        save_strategy="epoch",
        save_total_limit=cfg.training.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=_PaddingCollator(pad_token_id=tok.pad_token_id),
    )

    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"[lora-train] training done in {elapsed:.1f}s")

    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    (output_dir / "recipe.json").write_text(
        json.dumps({"base_model": cfg.base_model, "training": asdict(cfg.training)}, indent=2),
        encoding="utf-8",
    )

    return {
        "elapsed_secs": round(elapsed, 1),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": round(trainable / total, 6),
        "output_dir": str(output_dir),
    }


def _count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    result = train(cfg)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
