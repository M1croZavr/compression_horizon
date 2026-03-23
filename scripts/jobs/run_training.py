#!/usr/bin/env python3
"""
Training job launcher for compression_horizon progressive experiments.

All experiment configurations are defined in Python code below.
CLI args are only for filtering and execution control (--dry, --force, --model).

Usage:
  python scripts/jobs/run_training.py --dry
  python scripts/jobs/run_training.py --dry --model llama pythia
  python scripts/jobs/run_training.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys


# ── Constants ───────────────────────────────────────────────────────────────────

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
PROFILE = "default"
AUTHOR_NAME = "d.tarasov"
INSTANCE_TYPE = "a100.1gpu"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"

DATASET_NAME = "LarryLovestein/pg19_1k"
LIMIT_DATASET_ITEMS = 100
MAX_SEQ_LEN = 4096
MAX_OPTIMIZATION_STEPS_PER_SAMPLE = 10_000
MAX_OPTIMIZATION_STEPS_PER_TOKEN = 1_000
EMBEDDING_INIT_METHOD = "random0.02"

# Training configuration
PER_DEVICE_TRAIN_BATCH_SIZE = 1
NUM_GPUS = 1  # from a100.1gpu instance
GRADIENT_ACCUMULATION_STEPS = 1
TOTAL_BATCH_SIZE = PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
MAX_STEPS = LIMIT_DATASET_ITEMS // TOTAL_BATCH_SIZE  # 100 samples / 1 batch_size = 100 steps

# ── Per-model settings ──────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "learning_rate": 0.1,
        "low_dim_size": 256,
        "hybrid_num_alignment_layers": 4,
        "hybrid_lowdim_num_alignment_layers": 8,
    },
    {
        "checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "low_dim_size": 256,
        "hybrid_num_alignment_layers": 8,
        "hybrid_lowdim_num_alignment_layers": 8,
    },
    {
        "checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
        "learning_rate": 0.1,
        "low_dim_size": 256,
        "hybrid_num_alignment_layers": 8,
        "hybrid_lowdim_num_alignment_layers": 8,
    },
    {
        "checkpoint": "unsloth/gemma-3-4b-pt",
        "learning_rate": 0.1,
        "low_dim_size": 32,
        "hybrid_num_alignment_layers": 8,
        "hybrid_lowdim_num_alignment_layers": 8,
    },
]


def _make_variants(
    low_dim_size: int,
    hybrid_num_alignment_layers: int,
    hybrid_lowdim_num_alignment_layers: int,
) -> list[dict]:
    """5 experiment variants per model."""
    return [
        {
            "name": "simple",
            "loss_type": "cross_entropy",
            "num_alignment_layers": 1,
            "hybrid_alpha": None,
            "low_dim_projection": False,
            "low_dim_size": None,
            "no_bos_token": False,
        },
        {
            "name": "lowdim",
            "loss_type": "cross_entropy",
            "num_alignment_layers": 1,
            "hybrid_alpha": None,
            "low_dim_projection": True,
            "low_dim_size": low_dim_size,
            "no_bos_token": False,
        },
        {
            "name": "hybrid",
            "loss_type": "cosine",
            "num_alignment_layers": hybrid_num_alignment_layers,
            "hybrid_alpha": 1.0,
            "low_dim_projection": False,
            "low_dim_size": None,
            "no_bos_token": False,
        },
        {
            "name": "hybrid_lowdim",
            "loss_type": "cosine",
            "num_alignment_layers": hybrid_lowdim_num_alignment_layers,
            "hybrid_alpha": 1.0,
            "low_dim_projection": True,
            "low_dim_size": low_dim_size,
            "no_bos_token": False,
        },
        {
            "name": "nobos",
            "loss_type": "cross_entropy",
            "num_alignment_layers": 1,
            "hybrid_alpha": None,
            "low_dim_projection": False,
            "low_dim_size": None,
            "no_bos_token": True,
        },
    ]


def build_experiment_configs() -> list[dict]:
    """Build all progressive experiment configs from MODEL_CONFIGS x variants."""
    workdir = os.getcwd()
    dataset_suffix = DATASET_NAME.split("/")[-1]
    configs: list[dict] = []

    for mcfg in MODEL_CONFIGS:
        checkpoint = mcfg["checkpoint"]
        lr = mcfg["learning_rate"]
        model_short = checkpoint.split("/")[-1]
        variants = _make_variants(
            mcfg["low_dim_size"],
            mcfg["hybrid_num_alignment_layers"],
            mcfg["hybrid_lowdim_num_alignment_layers"],
        )

        for variant in variants:
            # Build command arguments (mirrors run_jobs_progressive.py logic)
            cmd_args = [
                "--remove_unused_columns False",
                f"--num_alignment_layers {variant['num_alignment_layers']}",
                f"--loss_type {variant['loss_type']}",
                f"--max_sequence_length {MAX_SEQ_LEN}",
                "--warmup_steps 100",
                f"--model_checkpoint {checkpoint}",
                f"--per_device_train_batch_size {PER_DEVICE_TRAIN_BATCH_SIZE}",
                f"--gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS}",
                f"--max_steps {MAX_STEPS}",
                f"--max_optimization_steps_per_sample {MAX_OPTIMIZATION_STEPS_PER_SAMPLE}",
                f"--max_optimization_steps_per_token {MAX_OPTIMIZATION_STEPS_PER_TOKEN}",
                f"--learning_rate {lr}",
                "--progressive_train 1",
                f"--embedding_init_method {EMBEDDING_INIT_METHOD}",
                f"--limit_dataset_items {LIMIT_DATASET_ITEMS}",
                f"--dataset_name {DATASET_NAME}",
            ]

            if variant["no_bos_token"]:
                cmd_args.append("--no_bos_token")
            if variant["hybrid_alpha"] is not None:
                cmd_args.append(f"--hybrid_alpha {variant['hybrid_alpha']}")
            if variant["low_dim_size"] is not None:
                cmd_args.append(f"--low_dim_size {variant['low_dim_size']}")
            if variant["low_dim_projection"]:
                cmd_args.append("--low_dim_projection")

            # Build output dir suffix (mirrors run_jobs_progressive.py naming)
            exp_suffix = f"sl_{MAX_SEQ_LEN}_{model_short}_ds_{dataset_suffix}_limit_{LIMIT_DATASET_ITEMS}"
            if variant["low_dim_size"] is not None:
                exp_suffix = f"{exp_suffix}_lowdim_{variant['low_dim_size']}"
            if variant["low_dim_projection"]:
                exp_suffix = f"{exp_suffix}_lowproj"
            if variant["no_bos_token"]:
                exp_suffix = f"{exp_suffix}_nobos"
            if lr != 0.01:
                exp_suffix = f"{exp_suffix}_lr_{lr}"
            if variant["loss_type"] != "cross_entropy":
                exp_suffix = f"{exp_suffix}_loss_{variant['loss_type']}"
            if variant["hybrid_alpha"] is not None:
                exp_suffix = f"{exp_suffix}_hybrid_{variant['hybrid_alpha']}"
            if variant["num_alignment_layers"] != 1:
                exp_suffix = f"{exp_suffix}_align_{variant['num_alignment_layers']}"

            output_dir = f"artifacts/experiments_progressive/{exp_suffix}"
            cmd_args.append(f"--output_dir {output_dir}")

            base_cmd = (
                f"cd {workdir} && {PYTHON_PATH} scripts/activation_distillation.py"
                f" {' '.join(cmd_args)}"
            )

            config = {
                "experiment_name": exp_suffix,
                "variant": variant["name"],
                "model_name": checkpoint,
                "output_dir": output_dir,
                "loss_type": variant["loss_type"],
                "hybrid_alpha": variant["hybrid_alpha"],
                "embedding_init_method": EMBEDDING_INIT_METHOD,
                "random_seed": 42,
                "max_sequence_length": MAX_SEQ_LEN,
                "num_alignment_layers": variant["num_alignment_layers"],
                "learning_rate": lr,
                "limit_dataset_items": LIMIT_DATASET_ITEMS,
                "dataset_name": DATASET_NAME,
                "low_dim_projection": variant["low_dim_projection"],
                "low_dim_size": variant["low_dim_size"],
                "no_bos_token": variant["no_bos_token"],
                "command": base_cmd,
                "training_function": "scripts.activation_distillation",
                "instance_type": INSTANCE_TYPE,
                "base_image": BASE_IMAGE,
            }
            configs.append(config)

    return configs


# ── Module-level variable ───────────────────────────────────────────────────────
experiment_configs: list[dict] = build_experiment_configs()


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch compression_horizon progressive training jobs."
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated configs, do not launch jobs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run jobs even if a matching experiment output directory already exists.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter experiments by model name (substring match).",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=None,
        help="Filter experiments by variant name (e.g. simple, lowdim, hybrid, hybrid_lowdim, nobos).",
    )
    return parser.parse_args()


def filter_configs(
    configs: list[dict],
    model_filters: list[str] | None,
    variant_filters: list[str] | None,
) -> list[dict]:
    filtered = configs
    if model_filters:
        model_filters_lower = [m.lower() for m in model_filters]
        filtered = [
            c for c in filtered
            if any(f in c["model_name"].lower() for f in model_filters_lower)
        ]
    if variant_filters:
        variant_filters_lower = [v.lower() for v in variant_filters]
        filtered = [
            c for c in filtered
            if c.get("variant", "").lower() in variant_filters_lower
        ]
    return filtered


if __name__ == "__main__":
    args = build_args()
    configs = filter_configs(experiment_configs, args.model, args.variant)

    if not configs:
        print("\033[33mNo experiments matched the filters.\033[0m")
        sys.exit(0)

    if args.dry:
        skipped = 0
        printed = 0
        for cfg in configs:
            if os.path.isdir(cfg["output_dir"]) and not args.force:
                print(
                    f"\033[33mSkipping: experiment already exists at:\033[0m {cfg['output_dir']}"
                )
                skipped += 1
                continue
            print(f"\033[32m[DRY] {cfg['experiment_name']}\033[0m")
            print(f"       Variant:    {cfg['variant']}")
            print(f"       Model:      {cfg['model_name']}")
            print(f"       Loss:       {cfg['loss_type']} (alpha={cfg['hybrid_alpha']})")
            print(f"       Seq len:    {cfg['max_sequence_length']}")
            print(f"       LR:         {cfg['learning_rate']}")
            print(f"       Output:     {cfg['output_dir']}")
            print(f"       Command:    {cfg['command']}")
            print()
            printed += 1
        print(f"[DRY] Total configs: {len(configs)}")
        print(f"[DRY] Would launch: {printed}")
        print(f"[DRY] Skipped (already exist): {skipped}")
        sys.exit(0)

    # Non-dry: import MLS SDK and launch jobs
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, extra_options = training_job_api_from_profile(PROFILE)
    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    jobs_launched = 0
    launched_jobs: list[dict] = []
    for cfg in configs:
        if os.path.isdir(cfg["output_dir"]) and not args.force:
            print(
                f"\033[33mSkipping: experiment already exists at:\033[0m {cfg['output_dir']}"
            )
            continue

        job_desc = (
            f"CH: progressive {cfg['experiment_name']}"
            f" #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"
        )

        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue:\033[0m {job_desc}")
            continue

        payload = {
            "script": cfg["command"],
            "job_desc": job_desc,
            "env_variables": {
                "PYTHONPATH": "./src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
            },
            "instance_type": cfg["instance_type"],
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": cfg["base_image"],
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        print(f"\033[32mLaunching:\033[0m {job_desc}")
        result = client.run_job(payload=payload)
        jobs_launched += 1
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if job_name:
            launched_jobs.append({
                "job_name": job_name,
                "job_desc": job_desc,
                "output_dir": cfg["output_dir"],
            })
        print(f"  Result: {result}")

    print(f"Total configs: {len(configs)}")
    print(f"Jobs launched: {jobs_launched}")

    out = {"jobs": launched_jobs, "launched": len(launched_jobs)}
    print("__TRAINING_JOBS_JSON__")
    print(json.dumps(out))
