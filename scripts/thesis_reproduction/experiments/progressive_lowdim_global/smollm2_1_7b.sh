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
# Why --per_device_train_batch_size 50 here (vs bs=1 in non-global script):
#   With global=True the optimization target is "find a 256-dim subspace
#   simultaneously good for ALL 50 texts". The right way to learn that is
#   mini-batch gradient averaging: ∂L/∂W = (1/N) Σ_j ∂L_j/∂W, which moves
#   W toward the compromise basis at every step. With bs=1 we'd instead
#   get sequential per-sample fitting — catastrophic forgetting in pure
#   form: sample 0 trains W for 10k steps on text #0, then sample 1
#   pushes W away from that optimum to fit text #1, and so on. Sample 49
#   would see a much more "trained" W than sample 0, and the metric
#   becomes a mix of "W after 0 of 49 prior updates" through "W after 49
#   of 49 prior updates" — uninterpretable as a single number.
#
#   With bs=50 (whole dataset in one batch): every sample inside the
#   batch sees the SAME W in forward, all per-sample numbers are on
#   equal footing, the lr_scheduler runs one continuous cosine curve
#   through all max_optimization_steps_per_sample steps with no
#   batch-boundary resets, and we get a clean test of the manifold
#   hypothesis. Memory-wise this should fit on A100 80GB — baseline
#   progressive/smollm2_1_7b already uses bs=25 on the same model +
#   max_seq_len=4096; the 256x2048 W gradient adds only ~524k params
#   (negligible). If OOM, drop to bs=25 — two-batch averaging is still
#   dramatically better than bs=1.
#
# All other paper-fidelity parameters (Appendix A) identical to the
# non-global script: lr=0.1, max_seq_len=4096, 10k steps/sample, 1k steps/
# token, random0.02 init, single compression token, cross-entropy loss,
# cosine_with_min_lr.
#
# Time on a single A100 80GB: ~1.5-3 h with bs=50 (vs 6-10 h had we kept
# bs=1, which we don't — see the bs reasoning above).
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
    --per_device_train_batch_size 50 \
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
