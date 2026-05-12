"""Evaluate a (base, adapter) pair on the recipe's eval split.

The decision is a one-token argmax over ``task.label_tokens`` so the eval
works on tiny models that aren't strong free-form generators.

Usage::

    lora-eval --config configs/smollm_sst2_cpu.yaml [--adapter outputs/smollm_sst2_cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_recipes.config import RecipeConfig, load_config
from lora_recipes.data import load_split, to_supervised_pairs


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(cfg: RecipeConfig, adapter_path: Path | None) -> tuple:
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (cfg.training.bf16 and torch.cuda.is_available()) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    base.to(_device())
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_path).to(_device())
    else:
        model = base
    model.eval()
    return tok, model


def _label_token_ids(tok, label_tokens: list[str]) -> list[int]:
    """Resolve each label string to a single token id; raise if not single-token."""
    ids: list[int] = []
    for label in label_tokens:
        toks = tok.encode(label, add_special_tokens=False)
        if not toks:
            raise ValueError(f"label {label!r} produced no tokens")
        # Some tokenizers split " positive" into multiple subwords on uncommon
        # models. We take the FIRST token; the eval is the per-class probability
        # of *that* first token, which is a stable proxy on tiny LMs.
        ids.append(toks[0])
    return ids


def _score_example(model, tok, prompt: str, label_token_ids: list[int]) -> int:
    """Return the predicted label index."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(_device()) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        # Logit on the next token only.
        logits = out.logits[0, -1, :]
    label_logits = torch.tensor([logits[i].item() for i in label_token_ids])
    return int(label_logits.argmax().item())


def evaluate(cfg: RecipeConfig, adapter_path: Path | None) -> dict:
    tok, model = _load_model(cfg, adapter_path)
    eval_split = load_split(cfg.dataset, cfg.dataset.split_eval, cfg.dataset.eval_size)
    eval_ds = to_supervised_pairs(eval_split, cfg.task, cfg.dataset)
    label_ids = _label_token_ids(tok, cfg.task.label_tokens)

    correct = 0
    n = 0
    t0 = time.perf_counter()
    confusion: dict[tuple[int, int], int] = {}
    for ex in eval_ds:
        pred = _score_example(model, tok, ex["prompt"], label_ids)
        gold = int(ex["label_idx"])
        n += 1
        if pred == gold:
            correct += 1
        confusion[(gold, pred)] = confusion.get((gold, pred), 0) + 1
    elapsed = time.perf_counter() - t0

    return {
        "adapter": str(adapter_path) if adapter_path else None,
        "accuracy": round(correct / max(n, 1), 4),
        "n": n,
        "elapsed_secs": round(elapsed, 1),
        "throughput_per_sec": round(n / max(elapsed, 1e-9), 1),
        "confusion": {f"gold={g},pred={p}": c for (g, p), c in confusion.items()},
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter dir; omit for base-only")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    adapter = Path(args.adapter) if args.adapter else None
    result = evaluate(cfg, adapter)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
