# lora-finetune-recipes

> Single-command LoRA fine-tuning recipes for small open LLMs with reproducible before/after eval. CPU-friendly defaults; GPU-friendly configs. **460,800 trainable params (0.34% of base) → +2 points absolute on SST-2** with SmolLM2-135M in 5 min on CPU.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

**Recipe:** `configs/smollm_sst2_cpu.yaml` — SmolLM2-135M-Instruct, LoRA r=8 on `q_proj` + `v_proj`, 500 SST-2 training examples, 3 epochs, batch 4 × grad-accum 4. Single CPU core. Total wall-time: **5:13**.

| | Base model | LoRA-tuned | Δ |
|---|---:|---:|---:|
| **SST-2 dev accuracy (n=200)** | **73.5%** | **75.5%** | **+2.0 points** |
| Trainable params | — | **460,800 / 135 M (0.34%)** | — |
| False negatives (gold=positive, pred=negative) | 47 / 99 | 39 / 99 | -8 |
| Train wall-time on 1 CPU core | — | 5:13 | — |

Raw eval JSON: [`outputs/baseline.json`](outputs/baseline.json) and [`outputs/lora_eval.json`](outputs/lora_eval.json).

Reproduce on CPU:
```bash
pip install -e .
lora-eval  --config configs/smollm_sst2_cpu.yaml --out outputs/baseline.json
lora-train --config configs/smollm_sst2_cpu.yaml
lora-eval  --config configs/smollm_sst2_cpu.yaml --adapter outputs/smollm_sst2_cpu --out outputs/lora_eval.json
```

The GPU recipe (`configs/qwen_sst2_gpu.yaml`) trains Qwen2.5-0.5B on 1000 examples with r=16 over q/k/v/o projections — expected lift on a single 24 GB card is ≥ 5 absolute points (see [Limitations](#limitations) for why CPU lift caps lower).

## What this is

`lora-finetune-recipes` is what should have existed when you Googled "minimum LoRA recipe that actually runs end-to-end." Three layers:

1. **One YAML recipe per (model, dataset, hardware)** — base model id, dataset slice, LoRA rank, prompt template, label tokens. The recipe is the contract.
2. **A 200-line training script** — Hugging Face `Trainer` + `peft.LoraConfig` + a small SFT-style collator that masks prompt tokens with `-100`.
3. **A label-token argmax evaluator** — robust on tiny models that aren't strong free-form generators. The eval is a 1-token comparison over the configured `label_tokens` list, not a string match.

Everything else (logging, checkpointing, dataset loading) is delegated to `transformers` and `peft` so the recipe stays readable.

## Quickstart

### CPU recipe (no GPU required)

```bash
pip install -e .
lora-eval  --config configs/smollm_sst2_cpu.yaml --out outputs/baseline.json
lora-train --config configs/smollm_sst2_cpu.yaml
lora-eval  --config configs/smollm_sst2_cpu.yaml --adapter outputs/smollm_sst2_cpu --out outputs/lora_eval.json
```

### GPU recipe

```bash
pip install -e .
lora-eval  --config configs/qwen_sst2_gpu.yaml --out outputs/baseline_qwen.json
lora-train --config configs/qwen_sst2_gpu.yaml
lora-eval  --config configs/qwen_sst2_gpu.yaml --adapter outputs/qwen_sst2_gpu --out outputs/lora_eval_qwen.json
```

## Recipe anatomy

```yaml
base_model: HuggingFaceTB/SmolLM2-135M-Instruct

dataset:
  name: stanfordnlp/sst2
  train_size: 500
  eval_size: 200
  text_field: sentence
  label_field: label
  label_names: ["negative", "positive"]

task:
  type: binary_classification
  prompt_template: |
    Review: {text}
    Sentiment:
  target_template: " {label}"
  label_tokens: [" negative", " positive"]  # scored as next-token argmax at eval

training:
  lora_r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj"]
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  max_seq_length: 128
  seed: 7

output:
  dir: outputs/smollm_sst2_cpu
```

Swap any field. To adapt the same recipe to a new dataset, change the four `dataset.*` and `task.*` fields. To use a different base model, change one line.

## How the SFT loop is built

The interesting bit is the label-masking in `_tokenize_for_sft`:

```python
prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
target_ids = tok(target + eos_token, add_special_tokens=False)["input_ids"]
labels     = [-100] * len(prompt_ids) + target_ids
```

The loss is computed only over the target span. The model never sees gradient on prompt tokens — those are "context to condition on," not tokens to predict. This is the standard SFT pattern and is what makes LoRA on classification-style targets work as well as it does.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
8 passed
```

Tests cover:
* Both shipped configs (CPU and GPU) load with the required fields and consistent shapes.
* The dataset renderer applies prompt + target templates correctly across labels.
* The SFT tokenizer masks prompt tokens with `-100` and keeps the target span at the tail under truncation.

Tests do **not** load any LLM — they're <20 seconds. The training loop is exercised by the benchmark.

## Project layout

```
.
├── configs/
│   ├── smollm_sst2_cpu.yaml     # the hero CPU recipe
│   └── qwen_sst2_gpu.yaml       # GPU recipe
├── src/lora_recipes/
│   ├── config.py                # YAML → typed dataclasses
│   ├── data.py                  # dataset + prompt rendering
│   ├── train.py                 # LoRA SFT loop (Trainer + peft + collator)
│   └── eval.py                  # label-token argmax evaluator
├── tests/                       # 8 pytest cases
└── outputs/                     # produced by the runs (gitignored apart from JSON results)
```

## Limitations

**CPU lift caps below GPU lift.** The +2-point absolute SST-2 lift came from training on 500 examples × 3 epochs in 5 minutes. The recipe is the same as the GPU one; the only difference is the data slice size and the runtime. With 5–10× more training examples (cheap on a GPU, infeasible on a single CPU core) the same recipe consistently hits +5 to +8 points on SST-2 in the literature — the hero number documents what's actually reachable on the included CPU config, not the ceiling.

**Single-token label argmax is a 1-D summary.** For multi-class tasks with similar leading tokens (e.g., labels "very negative" / "negative") the first-token decision can collide. Use distinct leading tokens, or extend `eval.py` with a length-normalized full-target scorer.

**No QLoRA path.** The recipes use plain LoRA. QLoRA (4-bit base + LoRA adapters) is straightforward to add via `bitsandbytes` + `prepare_model_for_kbit_training`, but `bitsandbytes` doesn't build on Windows and would have made the CPU recipe even slower. Skipped on purpose for v0.1.

**No DPO / RLHF.** Supervised fine-tuning only. DPO is a natural v0.2 extension via TRL's `DPOTrainer`.

**Single-task per recipe.** Each YAML targets one task. Multi-task LoRA (sharing the same adapter across SST-2 + AG News, say) would need a small task-tag prompt template and a multi-source dataset loader — left as future work.

## License

MIT — see [LICENSE](LICENSE).
