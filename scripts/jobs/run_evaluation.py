#!/usr/bin/env python3
"""
Evaluation job launcher for compression_horizon experiments.

This script provides a unified entry-point for launching evaluation jobs
across all benchmarks (ARC, HellaSwag, etc.).

Interface:
  - Supports `--dry` flag for dry-run mode (no MLS SDK required)
  - Executable via: PYTHONPATH=./src python scripts/jobs/run_evaluation.py --dry
  - Accepts checkpoint paths from artifacts directory

For benchmark-specific configuration, use the individual scripts:
  - run_jobs_arc_evaluate.py
  - run_jobs_hellaswag_evaluate.py
  - run_jobs_hellaswag_compression_head_evaluate.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List


BENCHMARKS = ["arc", "hellaswag"]

DEFAULT_CHECKPOINTS = [
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "unsloth/Llama-3.2-3B",
    "Qwen/Qwen3-4B",
    "unsloth/Meta-Llama-3.1-8B",
    "Qwen/Qwen3-8B",
    "allenai/OLMo-1B-hf",
    "allenai/Olmo-3-1025-7B",
    "unsloth/gemma-3-4b-pt",
    "unsloth/gemma-3-1b-pt",
    "EleutherAI/pythia-1.4b",
]


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch compression_horizon evaluation jobs."
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated configs, do not launch jobs.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=BENCHMARKS,
        choices=BENCHMARKS,
        help=f"Benchmarks to evaluate. Default: {BENCHMARKS}",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter models by name (substring match).",
    )
    parser.add_argument(
        "--arc_split",
        type=str,
        default="ARC-Easy",
        choices=["ARC-Easy", "ARC-Challenge"],
        help="ARC dataset split to use (default: ARC-Easy)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype to use. Default: bf16.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples.",
    )
    parser.add_argument(
        "--num_compression_tokens",
        type=int,
        default=None,
        help="Number of compression tokens.",
    )
    parser.add_argument(
        "--max_optimization_steps",
        type=int,
        default=None,
        help="Maximum optimization steps for compression.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for compression optimization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for compression and evaluation.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help="Loss type for optimization: cross_entropy | l2 | l1 | cosine.",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="Hybrid alpha value for combined loss.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=None,
        help="Number of layers to align.",
    )
    parser.add_argument(
        "--inverted_alignment",
        action="store_true",
        help="Align the last num_alignment_layers instead of the first.",
    )
    parser.add_argument(
        "--no_bos_token",
        action="store_true",
        default=False,
        help="Disable BOS token insertion during tokenization.",
    )
    return parser.parse_args()


def filter_checkpoints(checkpoints: List[str], model_filters: List[str] | None) -> List[str]:
    """Filter checkpoints by substring match on model name."""
    if not model_filters:
        return checkpoints
    filters = [m.lower() for m in model_filters]
    filtered = []
    for ckpt in checkpoints:
        ckpt_lower = ckpt.lower()
        model_name = ckpt.split("/")[-1].lower() if "/" in ckpt else ckpt_lower
        if any(f in ckpt_lower or f in model_name for f in filters):
            filtered.append(ckpt)
    return filtered


def build_eval_configs(args: argparse.Namespace) -> list[dict]:
    """Build evaluation configs for all requested benchmarks and models."""
    checkpoints = filter_checkpoints(DEFAULT_CHECKPOINTS, args.model)
    if not checkpoints:
        print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
        return []

    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    # Resolve defaults
    limit_samples = args.limit_samples if args.limit_samples is not None else 100
    num_compression_tokens = args.num_compression_tokens if args.num_compression_tokens is not None else 1
    max_optimization_steps = args.max_optimization_steps if args.max_optimization_steps is not None else 1000
    learning_rate = args.learning_rate if args.learning_rate is not None else 0.01
    batch_size = args.batch_size if args.batch_size is not None else 4
    dtype = args.dtype if args.dtype is not None else "bf16"
    loss_type = args.loss_type if args.loss_type is not None else "cross_entropy"
    num_alignment_layers = args.num_alignment_layers if args.num_alignment_layers is not None else 0

    configs: list[dict] = []

    for benchmark in args.benchmarks:
        for model_checkpoint in checkpoints:
            model_short = model_checkpoint.split("/")[-1] if "/" in model_checkpoint else model_checkpoint

            # Build experiment suffix
            if benchmark == "arc":
                split_suffix = args.arc_split.replace("-", "_")
                exp_suffix = f"arc_{split_suffix}_{model_short}"
                script_module = "scripts.arc_compress_evaluate"
            elif benchmark == "hellaswag":
                exp_suffix = f"hellaswag_{model_short}"
                script_module = "scripts.hellaswag_compress_evaluate"
            else:
                continue

            # Build command arguments
            cmd_args = [
                f"--model_checkpoint {model_checkpoint}",
                f"--limit_samples {limit_samples}",
                f"--num_compression_tokens {num_compression_tokens}",
                f"--max_optimization_steps {max_optimization_steps}",
                f"--learning_rate {learning_rate}",
                f"--batch_size {batch_size}",
                f"--dtype {dtype}",
                f"--loss_type {loss_type}",
                f"--num_alignment_layers {num_alignment_layers}",
            ]
            if benchmark == "arc":
                cmd_args.append(f"--arc_split {args.arc_split}")
            if args.hybrid_alpha is not None:
                cmd_args.append(f"--hybrid_alpha {args.hybrid_alpha}")
            if args.inverted_alignment:
                cmd_args.append("--inverted_alignment")
            if args.no_bos_token:
                cmd_args.append("--no_bos_token")

            # Add non-default params to suffix
            if args.random_seed is not None and args.random_seed != 42:
                cmd_args.append(f"--random_seed {args.random_seed}")
                exp_suffix = f"{exp_suffix}_seed_{args.random_seed}"
            if args.limit_samples is not None and args.limit_samples != 100:
                exp_suffix = f"{exp_suffix}_samples_{args.limit_samples}"
            if args.num_compression_tokens is not None and args.num_compression_tokens != 1:
                exp_suffix = f"{exp_suffix}_tokens_{args.num_compression_tokens}"
            if args.max_optimization_steps is not None and args.max_optimization_steps != 1000:
                exp_suffix = f"{exp_suffix}_steps_{args.max_optimization_steps}"
            if args.learning_rate is not None and args.learning_rate != 0.01:
                exp_suffix = f"{exp_suffix}_lr_{args.learning_rate}"
            if args.batch_size is not None and args.batch_size != 4:
                exp_suffix = f"{exp_suffix}_batch_{args.batch_size}"
            if args.dtype is not None and args.dtype != "bf16":
                exp_suffix = f"{exp_suffix}_dtype_{args.dtype}"
            if args.loss_type is not None and args.loss_type != "cross_entropy":
                exp_suffix = f"{exp_suffix}_loss_{args.loss_type}"
            if args.hybrid_alpha is not None:
                exp_suffix = f"{exp_suffix}_hybrid_{args.hybrid_alpha}"
            if args.num_alignment_layers is not None and args.num_alignment_layers != 0:
                exp_suffix = f"{exp_suffix}_align_{args.num_alignment_layers}"
            if args.inverted_alignment:
                exp_suffix = f"{exp_suffix}_inv_align"
            if args.no_bos_token:
                exp_suffix = f"{exp_suffix}_nobos"

            out_dir_name = f"artifacts/{benchmark}_evaluation/{exp_suffix}"
            cmd_args.append(f"--output_dir {out_dir_name}")

            command = (
                f"cd {workdir} && {python_path} -m {script_module} {' '.join(cmd_args)}"
            )

            config = {
                "experiment_name": exp_suffix,
                "benchmark": benchmark,
                "model_name": model_checkpoint,
                "model_short": model_short,
                "output_dir": out_dir_name,
                "command": command,
                "script_module": script_module,
                "limit_samples": limit_samples,
                "num_compression_tokens": num_compression_tokens,
                "max_optimization_steps": max_optimization_steps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "dtype": dtype,
                "loss_type": loss_type,
            }
            configs.append(config)

    return configs


if __name__ == "__main__":
    args = build_args()
    eval_configs = build_eval_configs(args)

    if args.dry:
        skipped = 0
        printed = 0
        for cfg in eval_configs:
            if os.path.exists(cfg["output_dir"]):
                print(
                    f"\033[33mSkipping: experiment already exists at:\033[0m {cfg['output_dir']}"
                )
                skipped += 1
                continue
            print(f"\033[32m[DRY] {cfg['benchmark'].upper()}: {cfg['experiment_name']}\033[0m")
            print(f"       Model:   {cfg['model_name']}")
            print(f"       Output:  {cfg['output_dir']}")
            print(f"       Command: {cfg['command']}")
            print()
            printed += 1
        print(f"[DRY] Total configs: {len(eval_configs)}")
        print(f"[DRY] Would launch: {printed}")
        print(f"[DRY] Skipped (already exist): {skipped}")
        sys.exit(0)

    # Non-dry: import MLS SDK and launch jobs
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, extra_options = training_job_api_from_profile("default")
    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    author_name = "d.tarasov"
    jobs_launched = 0

    for cfg in eval_configs:
        if os.path.exists(cfg["output_dir"]):
            print(
                f"\033[33mSkipping: experiment already exists at:\033[0m {cfg['output_dir']}"
            )
            continue

        job_desc = (
            f"CH: {cfg['benchmark']} eval {cfg['experiment_name']} "
            f"#{author_name} #multimodal #notify_completed @mrsndmn"
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
            "instance_type": "a100.1gpu",
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        print(f"\033[32mLaunching:\033[0m {job_desc}")
        result = client.run_job(payload=payload)
        jobs_launched += 1
        print(f"  Result: {result}")

    print(f"Total configs: {len(eval_configs)}")
    print(f"Jobs launched: {jobs_launched}")
