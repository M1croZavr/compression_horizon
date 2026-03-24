#!/usr/bin/env python3
"""
Results aggregation script for compression_horizon experiments.

Reads experiment configs from run_training.py, filters out nobos variants,
and runs scripts/paper/low_dimesional.py to produce the results table.
"""
from __future__ import annotations

import subprocess
import sys

# Import experiment configs from the training script
sys.path.insert(0, ".")
from scripts.jobs.run_training import experiment_configs

VARIANTS_PER_MODEL = 4  # simple, lowdim, hybrid, hybrid_lowdim (nobos excluded)


def get_results_checkpoints() -> list[str]:
    """Return ordered checkpoint paths for non-nobos experiments."""
    configs = [cfg for cfg in experiment_configs if cfg.get("variant") != "nobos"]
    return [f"{cfg['output_dir']}/progressive_prefixes" for cfg in configs]


def get_midrule_indices(n_variants: int = VARIANTS_PER_MODEL) -> list[int]:
    """Compute midrule indices (last index of each model group)."""
    checkpoints = get_results_checkpoints()
    n_models = len(checkpoints) // n_variants
    return [i * n_variants - 1 for i in range(1, n_models)]


def main():
    checkpoints = get_results_checkpoints()
    midrule_indices = get_midrule_indices()

    cmd = [
        sys.executable,
        "scripts/paper/low_dimesional.py",
        "--checkpoints",
        *checkpoints,
        "--n_components",
        "4",
        "--sample_id",
        "0",
        "--midrule_indicies",
        *[str(i) for i in midrule_indices],
        "--show_labels",
        "--only_stat_table",
        "--tablefmt",
        "github",
    ]

    print(f"Running: {' '.join(cmd[:5])} ... ({len(checkpoints)} checkpoints)")
    result = subprocess.run(cmd, env={"PYTHONPATH": "./src:."})
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
