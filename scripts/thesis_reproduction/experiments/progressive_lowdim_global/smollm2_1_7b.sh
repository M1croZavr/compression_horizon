#!/usr/bin/env bash
# **NOT paper Table 6** — this is an *additional* experiment exploring the
# paper's central claim that the compression embedding lives on a shared
# low-dimensional manifold (Section 4.3, Section 5.2).
#
# Difference from progressive_lowdim/smollm2_1_7b.sh:
#   - This run uses --low_dim_projection_global, which means ONE nn.Linear
#     ∈ R^{hidden×256} lives through the whole 50-sample dataset.  Its
#     AdamW state and cosine LR-scheduler accumulate continuously — every
#     sample contributes gradients to the same W.  At the end we save the
#     trained W as `low_dim_projection.pt`, which encodes the *common
#     compression basis* learned across all 50 PG19 texts.
#
# What this tests:
#   1. **Existence of a shared basis.**  If cram_tokens / IG with global=True
#      come out close to paper Table 6's per-sample number (957.4 / 3271),
#      that's direct evidence the same 256-dim basis suffices for all texts —
#      the paper's manifold hypothesis is borne out.  If they collapse, the
#      basis has to be heavily per-sample-specialized.
#   2. **Transferability of the basis.**  The saved `low_dim_projection.pt`
#      can be loaded into a downstream run with --low_dim_projection_checkpoint
#      to test how many cram tokens a *frozen* basis (--no_low_dim_projection_train)
#      buys on a held-out text set.  Real "compression-basis transfer learning".
#
# Why batch_size still = 1 (matching a0d39f6:run_jobs_progressive.py:230):
#   With bs=1 each "batch" is one sample, but unlike global=False the same W
#   is reused — z stays per-sample (each text gets its own optimal coefficients),
#   while W is global.  This is the cleanest way to study the basis itself,
#   isolating the per-sample variation into z alone.
#
# All other paper-fidelity parameters (Appendix A) identical to the non-global
# script: lr=0.1, max_seq_len=4096, 10k steps/sample, 1k steps/token, random0.02
# init, single compression token, cross-entropy loss, cosine_with_min_lr.
#
# Time on a single A100 80GB: ~6-10 h, similar to the non-global variant.
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive_lowdim_global/smollm2_1_7b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"
mkdir -p "$OUTPUT_DIR"

# tee duplicates stdout+stderr to ``train.log`` in the output dir so the
# terminal-output is preserved alongside the shared-basis artifact
# (``low_dim_projection.pt``), the raw per-stage rows
# (``progressive_prefixes/``), and the JSON summary
# (``analysis_summary.json``).  Convenient for the thesis writeup when you
# come back to numbers months later.
{
  uv run python scripts/thesis_reproduction/train.py \
    --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
    --dataset_name LarryLovestein/pg19_1k \
    --max_sequence_length 4096 \
    --limit_dataset_items 50 \
    --per_device_train_batch_size 1 \
    --max_optimization_steps_per_sample 10000 \
    --max_optimization_steps_per_token 1000 \
    --learning_rate 0.1 \
    --warmup_steps 100 \
    --embedding_init_method random0.02 \
    --loss_type cross_entropy \
    --low_dim_projection \
    --low_dim_projection_global \
    --low_dim_size 256 \
    --progressive_train \
    --progressive_min_seq_len 1 \
    --progressive_step 1 \
    --progressive_convergence_threshold 1.0 \
    --dtype bf16 \
    --output_dir "$OUTPUT_DIR"

  echo
  echo "==============================================================================="
  echo "Training finished. Run-shared basis saved to:"
  echo "  ${OUTPUT_DIR}/low_dim_projection.pt"
  echo
  echo "Compare with the per-sample-basis variant (progressive_lowdim/smollm2_1_7b.sh)"
  echo "to see whether a SHARED 256-dim basis suffices for all 50 texts (paper's"
  echo "manifold-hypothesis test, Section 4.3 + Section 5.2)."
  echo "==============================================================================="
  uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
} 2>&1 | tee "${OUTPUT_DIR}/train.log"
