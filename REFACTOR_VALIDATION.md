# Validation guide for the FullCramming refactor

This document is a self-contained walkthrough you can use to convince yourself
the refactor is correct before you commit and run the SmolLM2 experiment.

---

## 0. One-shot sanity checks (60 seconds total)

```bash
# Activate the project venv first if needed.
PYTHONPATH=./src .venv/bin/python -c "
from compression_horizon.train.base import BaseTrainer
from compression_horizon.train.full_cramming_trainer import FullCrammingTrainer
from compression_horizon.train.progressive_cramming_trainer import ProgressiveCrammingTrainer
from compression_horizon.train.low_dim_trainer import LowDimTrainer
from compression_horizon.train.prefix_tuning_trainer import PrefixTuningTrainer
from compression_horizon.train.compression_head_trainer import CompressionHeadTrainer
from compression_horizon.train.embedding_init import prepare_embedding_init, create_compression_embedding
from compression_horizon.train.optimization import build_optimizer_and_scheduler, build_low_dim_projection
from compression_horizon.train.parametrization import DirectParametrization, PretrainedPCAParametrization, build_parametrization
from compression_horizon.train.inputs import build_compression_attention_mask, build_united_input
from compression_horizon.analysis import ConvergedSamplesGuard, ConvergenceTracker, compute_information_gain
from compression_horizon.analysis.attention_intervention import get_decoder_layers
from compression_horizon.analysis.perplexity import estimate_token_perplexity
from compression_horizon.config import (
    AlignmentArgs, CompressionArgs, DataArgs, EvalArgs,
    LowDimArgs, ModelArgs, ProgressiveArgs,
)
print('imports OK')
"

PYTHONPATH=./src .venv/bin/python -m pytest tests/ --ignore=tests/books_html -q
```

Expected: `imports OK`, then **`36 passed, 46 skipped`**.

If any non-CUDA-skipped test fails — the refactor broke something. Stop and
revert before continuing.

---

## 1. The shape of the codebase

`src/compression_horizon/` after the refactor:

```
analysis/                  # everything that PROBES / MEASURES the model
  __init__.py              # re-exports the most-used analysis classes
  attention_intervention.py  # paper Section 5.5 — attention mass + knockouts (formerly intervention.py)
  convergence.py             # ConvergenceTracker + ConvergedSamplesGuard (per-step state)
  information_gain.py        # paper Eq. 9 — per-sample IG in bits
  perplexity.py              # generic perplexity helpers (formerly metric.py)

config/                    # CLI argument groups composable via HfArgumentParser
  alignment.py compression.py data.py eval.py low_dim.py model.py progressive.py

train/                     # everything that TRAINS the compression embedding
  arguments.py             # MyTrainingArguments (legacy combined dataclass)
  base.py                  # BaseTrainer: forward+loss+logging+saving (was 789 LOC, now 312)
  embedding_init.py        # NEW: prepare_embedding_init + create_compression_embedding
  optimization.py          # NEW: build_optimizer_and_scheduler + build_low_dim_projection
  parametrization.py       # Direct + PretrainedPCAParametrization + build_parametrization
  inputs.py                # build_compression_attention_mask + build_united_input
  loss.py                  # CE + activation alignment loss
  full_cramming_trainer.py
  progressive_cramming_trainer.py
  low_dim_trainer.py
  prefix_tuning_trainer.py
  compression_head_trainer.py
  trainer.py               # legacy re-export shim (do not delete: tests rely on it)

inference/                 # HF model.generate-style helpers
models/                    # Custom HF model variants (LlamaForCausalLMCompressionHead)
utils/                     # Misc (launch, exceptions, token counting)
```

## 2. What changed in this iteration

### 2.1 Top-level analysis modules moved into `analysis/`

* `compression_horizon/intervention.py` → `compression_horizon/analysis/attention_intervention.py`
* `compression_horizon/metric.py` → `compression_horizon/analysis/perplexity.py`
* `compression_horizon/utils/train.py` (`ConvergedSamplesGuard`)
  → merged into `compression_horizon/analysis/convergence.py` (next to `ConvergenceTracker`)

Import path updates:

* `tests/test_intervention.py` — points to `compression_horizon.analysis.attention_intervention`.
* `src/compression_horizon/train/full_cramming_trainer.py` — imports `ConvergedSamplesGuard` from `compression_horizon.analysis`.

> **Heads-up for scripts**: `scripts/{arc,hellaswag}_compress_evaluate.py` still
> import from `compression_horizon.intervention` and `compression_horizon.metric`.
> Those scripts are evaluation entry points (not part of this refactor scope) —
> they will need an import update **before** you run them, but that's
> independent of the FullCramming refactor itself. Run the SmolLM2 cramming
> experiment first; eval scripts later.

### 2.2 BaseTrainer slimmed from 789 → 312 LOC

Extracted into focused modules (each method now lives where it conceptually belongs):

| Was in `BaseTrainer` | Now lives in |
|---|---|
| `_init_compression_tokens` (132 LOC switch on 20+ init methods) | `train/embedding_init.py:create_compression_embedding` |
| `_prepare_embedding_init` (200 LOC, mvnormal/PCA fitting/load_from_disk) | `train/embedding_init.py:prepare_embedding_init` + private helpers |
| `_build_optimizer_and_scheduler` | `train/optimization.py:build_optimizer_and_scheduler` |
| `_prepare_low_dim_proj` | `train/optimization.py:build_low_dim_projection` |
| `_sample_prefix_lengths`, `_build_compressed_inputs` (only used by CompressionHead) | private functions in `train/compression_head_trainer.py` |
| `_find_prefix_embedding_parameter`, `_log_step_prefix_tuning`, `_save_prefix_tuning_artifacts` (only used by PrefixTuning) | local in `train/prefix_tuning_trainer.py` (or unified into `_log_step`) |

`BaseTrainer` keeps thin compatibility wrappers (`self._init_compression_tokens`,
`self._prepare_embedding_init`, `self._build_optimizer_and_scheduler`,
`self._prepare_low_dim_proj`) that delegate to the new modules — so subclasses
can keep calling them without modification. This preserves backward compatibility
for ProgressiveCramming, LowDim, etc., that haven't been re-architected yet.

### 2.3 Save methods unified

`_save_artifacts` and `_save_prefix_tuning_artifacts` were 99% identical. Now a
single method:

```python
self._save_artifacts(
    rows,
    tensor=embedding_tensor_or_None,
    tensor_filename="compression_embeddings.pt",  # or "prefix_tuning_embeddings.pt"
    subdir_name="compressed_prefixes",            # or "progressive_prefixes" / "prefix_tuning_prefixes"
)
```

### 2.4 Log methods unified

`_log_step_prefix_tuning` was 95% the same as `_log_step` with `prefix_tuning/*`
TensorBoard namespace. Now `_log_step` accepts two namespace overrides:

```python
self._log_step(
    loss, alignment_loss, convergence_per_sample,
    prefix_param,                               # any param to log mean/std/grad
    lr_scheduler,
    embedding_namespace="prefix_tuning",        # default: "compression_token_embeddings"
    grad_norm_namespace="prefix_tuning",        # default: "train"
)
```

> Side effect: TensorBoard tag for prefix tuning embedding mean is now
> `prefix_tuning/mean` (was `prefix_tuning/emb_mean`); same for `/std`. Old
> tensorboard runs are still readable; new runs use the renamed tags. Acceptable.

---

## 3. Read-through order before commit (recommended)

The structure is now arranged so that bottom-level helpers are simple and you
can read in order of increasing complexity.

### Layer 1 — leaf helpers (≤ 60 LOC each, read in any order)

1. `train/inputs.py` — `build_compression_attention_mask`, `build_united_input`. Pure functions.
2. `analysis/convergence.py` — `ConvergenceTracker`, `ConvergedSamplesGuard`. Two small classes.
3. `analysis/information_gain.py` — paper Eq. 9. Pure stateless function.
4. `train/optimization.py` — `build_optimizer_and_scheduler` + `build_low_dim_projection`.

### Layer 2 — embedding initialization (read together)

5. `train/embedding_init.py` — start at the bottom (`create_compression_embedding`,
   the per-batch sampler), then read `prepare_embedding_init` (the once-per-run
   fitting). Note: 20+ legacy init methods are now organized in
   `_DIRECT_INIT_STRATEGIES` dict + a few special-cased methods.
6. `train/parametrization.py` — `DirectParametrization`, `PretrainedPCAParametrization`,
   `build_parametrization`. Tiny classes, used by FullCramming.

### Layer 3 — base class

7. `train/base.py` — read top-to-bottom. Sections are explicit:
   - `__init__` (Accelerator, writer)
   - subclass entry: `train()` raises NotImplementedError
   - `compute_loss` + `compute_target_hidden` (forward pass)
   - compatibility wrappers (`_init_compression_tokens`, etc.)
   - `_create_dataloader`, `_log_step`, `_save_artifacts`

### Layer 4 — the trainer you care about

8. `train/full_cramming_trainer.py` — start at the bottom and read up:
   - 3 `_RunContext` / `_BatchInputs` / `_BatchOptimizationResult` dataclasses
     describe the data flow.
   - `train()` is one line: calls `_train_full_cramming`.
   - `_train_full_cramming` is a 22-line orchestrator: `for batch: prepare → optimize → collect`.
   - `_build_run_context` = once-per-run setup.
   - `_prepare_batch_inputs`, `_optimize_compression`, `_run_optimization_loop`,
     `_log_progress`, `_collect_batch_rows`, `_build_sample_row` — all ≤ 60 LOC.

### Layer 5 — the safety net

9. `tests/test_full_cramming_golden.py` — the **byte-identical regression test**
   on a tiny CPU GPT2 with 4 samples × 16 tokens × 20 steps. This is the test
   that has been kept green through every refactor step. Read it to understand
   exactly what numbers are pinned.

10. `tests/test_information_gain.py` — verifies the IG implementation against a
    pre-refactor reference. Five parametrized test cases.

---

## 4. Conceptual checks (catch errors at the level of ideas)

For each of these, mentally trace through the code and answer:

### 4.1 "Where does the compression embedding live?"

It's owned by an instance of `DirectParametrization` (or
`PretrainedPCAParametrization`) created in
`FullCrammingTrainer._optimize_compression`. The optimizer sees
`parametrization.parameters` (a list with one Parameter). On each step
`parametrization.materialize()` returns the current `[B, K, H]` tensor,
optionally derived from low-rank coefficients.

### 4.2 "How is information gain computed?"

`compute_information_gain` (paper Eq. 9) makes **two forward passes** through
the frozen LM per sample: one with the original tokens, one with
`[compression || tokens]`. Both pass cross-entropies are converted to bits
(`/ log(2)`). IG = H_LM − H_compLM. This is verified by
`tests/test_information_gain.py` against a copy of the pre-refactor code.

### 4.3 "What freezes a converged sample?"

`ConvergedSamplesGuard` (in `analysis/convergence.py`):
* `before_step(mask)` → zeros the gradients for samples where `mask[j] == True`,
  snapshots the current parameter values for those indices.
* `optimizer.step()` runs as normal — but for converged samples the gradient
  was zero, so AdamW only touches them via momentum and weight decay.
* `after_step(mask)` → restores the snapshot for converged samples, undoing any
  remaining drift.

### 4.4 "How are different convergence thresholds tracked?"

`ConvergenceTracker` keeps three CPU buffers `[max_steps, B]`, one per
threshold (defaults: 0.95, 0.99, 1.0). `update(step_i, conv_per_sample)` writes
`(conv_per_sample < threshold)` into each buffer. `steps_below(threshold)`
returns the per-sample sum (i.e., how many steps were spent below the
threshold). `fully_converged` is the boolean mask used by the guard.

### 4.5 "What's the row schema saved on disk?"

`FullCrammingTrainer._build_sample_row` is the **single source of truth** for
the 24-field row dict. Downstream eval scripts (HellaSwag, ARC, MMLU) read this
schema. Don't change field names or types without updating the eval scripts.

### 4.6 "Why do we have a `single_compressed_init`?"

`single_*` init methods (`single_random`, `single_random0.02`) seed the
embedding once and broadcast it to every sample in the batch. The seed is
computed once per **run** (in `_maybe_init_single_compressed`) and reused
across all batches.

---

## 5. Final commit checklist

Before `git commit`, run these in order:

```bash
# (a) imports + tests
PYTHONPATH=./src .venv/bin/python -c "import compression_horizon; print('ok')"
PYTHONPATH=./src .venv/bin/python -m pytest tests/ --ignore=tests/books_html -q

# (b) git status overview
git status --short
```

You should see:

* `RM` `src/compression_horizon/intervention.py` (renamed)
* `R` `src/compression_horizon/metric.py` (renamed)
* `D` `src/compression_horizon/utils/train.py` (deleted; merged into analysis/convergence.py)
* `M` `src/compression_horizon/train/base.py` (slimmed)
* `M` `src/compression_horizon/train/{full,low_dim,progressive,prefix_tuning,compression_head}_cramming_trainer.py`
* `??` `src/compression_horizon/train/embedding_init.py` (new)
* `??` `src/compression_horizon/train/optimization.py` (new)
* `??` `src/compression_horizon/train/parametrization.py` (new)
* `??` `src/compression_horizon/train/inputs.py` (new)
* `??` `src/compression_horizon/analysis/` (new package: information_gain, convergence, attention_intervention, perplexity)
* `??` `src/compression_horizon/config/` (new package: 7 dataclass groups)
* `??` `tests/test_full_cramming_golden.py`, `tests/test_information_gain.py`, `tests/test_config.py` (new)
* `M` `tests/test_intervention.py` (import path updated)

If anything else has changed unexpectedly, investigate before committing.

---

## 6. Running the SmolLM2-135M experiment after commit

Once the refactor is committed and pushed, on the GPU machine:

```bash
PYTHONPATH=./src python scripts/activation_distillation.py \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
  --dataset_name mrsndmn/pg19 \
  --max_sequence_length 64 \
  --limit_dataset_items 4 \
  --max_optimization_steps_per_sample 5000 \
  --learning_rate 0.01 \
  --embedding_init_method random0.02 \
  --loss_type cross_entropy \
  --per_device_train_batch_size 4 \
  --dtype bf16 \
  --output_dir artifacts/smoke_smollm135m_full
```

Expected:

* `convergence_after_steps[j]` < 5000 for at least some samples (i.e., they
  converged before exhausting the budget).
* `final_convergence` ≥ 0.95 for converged samples.
* `information_gain_bits` > 0 for converged samples (paper Table 13:
  SmolLM2-135M progressive baseline ≈ 168 ± 66 bits — Full Cramming on a
  comparable token budget should be in the same order of magnitude).
* No PCA / load_from_disk / mvnormal code paths are exercised — only the
  `random0.02` direct path. The PCA / mvnormal paths are kept for backward
  compatibility but are not part of this smoke test.

After the run, inspect the saved Dataset:

```bash
PYTHONPATH=./src python -c "
from datasets import Dataset
ds = Dataset.load_from_disk('artifacts/smoke_smollm135m_full/compressed_prefixes')
for row in ds:
    print({
        'sample_id': row['sample_id'],
        'final_convergence': row['final_convergence'],
        'convergence_after_steps': row['convergence_after_steps'],
        'information_gain_bits': row['information_gain_bits'],
        'num_input_tokens': row['num_input_tokens'],
    })
"
```
