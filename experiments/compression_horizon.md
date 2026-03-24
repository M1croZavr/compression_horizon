# Compression Horizon: Progressive Activation Distillation

## Hypothesis

Progressive activation distillation with varying compression techniques (simple, low-dimensional projection, hybrid cosine loss, and no-BOS variants) will produce different compression horizons across multiple model architectures. Testing these variants systematically will reveal which approaches best preserve model performance during activation compression.

## Setup

- **Training function**: `scripts.activation_distillation`
- **Instance type**: `a100.1gpu`
- **Models**: Llama-3.1-8B, Pythia-1.4B, SmolLM2-1.7B, Gemma-3-4B
- **Variants**: simple, lowdim, hybrid, hybrid_lowdim, nobos
- **Artifact path**: `artifacts/experiments_progressive/<experiment_name>/`

All experiment configurations are defined in `scripts/jobs/run_training.py`.

## Results

_To be filled after running the experiment._

## Conclusions

_To be filled after analysis._

## Changelog

- 2026-03-23: Initial experiment plan created
