#!/usr/bin/env bash
# Reproduce paper Table 11, row "Pythia160m / Full":
#   compressed_tokens = 32, information_gain_bits = 105 ± 20, final_convergence = 0.684 ± 0.175.
#
# Time budget on a single A100: ~5-10 min.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="full_cramming/pythia_160m"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

PYTHONPATH=./src python scripts/thesis_reproduction/train.py \
  --model_checkpoint EleutherAI/pythia-160m \
  --dataset_name mrsndmn/pg19 \
  --max_sequence_length 32 \
  --limit_dataset_items 10 \
  --per_device_train_batch_size 10 \
  --max_optimization_steps_per_sample 10000 \
  --learning_rate 0.01 \
  --warmup_steps 100 \
  --embedding_init_method random0.02 \
  --loss_type cross_entropy \
  --dtype bf16 \
  --output_dir "$OUTPUT_DIR"

echo
echo "==============================================================================="
echo "Training finished. Comparing with paper expected values..."
echo "==============================================================================="
PYTHONPATH=./src python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
