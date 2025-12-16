import argparse
import hashlib
import os
from itertools import product
from typing import List, Optional

from mls.manager.job.utils import training_job_api_from_profile


def parse_hybrid_alpha_list(values: List[str]) -> List[Optional[float]]:
    parsed: List[Optional[float]] = []
    for v in values:
        if v is None:
            parsed.append(None)
            continue
        s = str(v).strip().lower()
        if s in {"none", "null", "nil"}:
            parsed.append(None)
        else:
            parsed.append(float(s))
    return parsed


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch compression_horizon training jobs over a configurable grid.")

    # Grid parameters
    parser.add_argument(
        "--embedding_init_methods",
        nargs="+",
        default=["mvnormal"],
        help="List of embedding initialization methods.",
    )
    parser.add_argument(
        "--random_seeds",
        nargs="+",
        type=int,
        # default=[42, 533, 134, 733],
        default=[42],
        help="List of random seeds.",
    )
    parser.add_argument(
        "--fix_position_ids",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Fix position ids or not?",
    )
    parser.add_argument(
        "--max_sequence_lengths",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="List of max sequence lengths.",
    )
    parser.add_argument(
        "--hybrid_alphas",
        nargs="+",
        default=[None, "1.0"],
        help='List of hybrid alpha values. Use "none" to disable hybrid and use cross-entropy loss.',
    )

    # General execution/runtime configuration
    parser.add_argument(
        "--python_path",
        default="/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python",
        help="Absolute path to the Python interpreter inside the target environment.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile name for training_job_api_from_profile.",
    )
    parser.add_argument(
        "--author_name",
        default="d.tarasov",
        help="Author name tag for job description.",
    )
    parser.add_argument(
        "--model_checkpoint",
        default="HuggingFaceTB/SmolLM2-1.7B",
        help="Base model checkpoint to use.",
    )
    parser.add_argument(
        "--model_checkpoints",
        nargs="+",
        default=None,
        help="List of model checkpoints to grid over (overrides --model-checkpoint).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Torch dtype to use: auto | float32/fp32 | bfloat16/bf16 | float16/fp16.",
    )

    # Training defaults that were previously hardcoded
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--max_optimization_steps_per_sample", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument(
        "--limit_dataset_items",
        type=int,
        default=100,
        help="Limit the number of dataset items to use.",
    )
    parser.add_argument(
        "--remove_unused_columns",
        action="store_true",
        default=False,
        help="Pass to set --remove_unused_columns True (default False).",
    )
    parser.add_argument(
        "--num_alignment_layers_nonhybrid",
        type=int,
        default=0,
        help="num_alignment_layers to use when hybrid is disabled.",
    )
    parser.add_argument(
        "--num_alignment_layers_hybrid",
        type=int,
        default=5,
        help="num_alignment_layers to use when hybrid is enabled.",
    )

    # Infra
    parser.add_argument("--instance_type", default="a100.1gpu")
    parser.add_argument(
        "--base_image",
        default="cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36",
    )
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--processes_per_worker", type=int, default=1)

    # Behavior
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run jobs even if a matching experiment output directory already exists.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    workdir = os.getcwd()
    python_path = args.python_path

    client, extra_options = training_job_api_from_profile(args.profile)

    author_name = args.author_name

    hybrid_alpha_values = parse_hybrid_alpha_list(args.hybrid_alphas)
    model_checkpoints = args.model_checkpoints if args.model_checkpoints is not None else [args.model_checkpoint]

    jobs_planned = 0
    jobs_launched = 0
    jobs_dry = 0

    for hybrid_alpha, embedding_init_method, random_seed, max_sequence_length, fix_position_ids, model_checkpoint in product(
        hybrid_alpha_values,
        args.embedding_init_methods,
        args.random_seeds,
        args.max_sequence_lengths,
        args.fix_position_ids,
        model_checkpoints,
    ):
        jobs_planned += 1
        is_hybrid = hybrid_alpha is not None
        loss_type = "cosine" if is_hybrid else "cross_entropy"
        num_alignment_layers = args.num_alignment_layers_hybrid if is_hybrid else args.num_alignment_layers_nonhybrid

        remove_unused_columns_str = "True" if args.remove_unused_columns else "False"

        # Build arguments-only string (exactly what the script receives) for hashing
        args_parts = [
            f"--remove_unused_columns {remove_unused_columns_str}",
            f"--num_alignment_layers {num_alignment_layers}",
            f"--loss_type {loss_type}",
            f"--max_sequence_length {max_sequence_length}",
            f"--dtype {args.dtype}",
            f"--warmup_steps {args.warmup_steps}",
            f"--model_checkpoint {model_checkpoint}",
            f"--per_device_train_batch_size {args.per_device_train_batch_size}",
            f"--max_optimization_steps_per_sample {args.max_optimization_steps_per_sample}",
            f"--learning_rate {args.learning_rate}",
            f"--limit_dataset_items {args.limit_dataset_items}",
            f"--random_seed {random_seed}",
            f"--embedding_init_method {embedding_init_method}",
            f"--fix_position_ids {fix_position_ids}",
        ]
        if is_hybrid:
            args_parts.append(f"--hybrid_alpha {hybrid_alpha}")
        args_for_hash = " ".join(args_parts).strip()

        # Build deterministic output directory: essential prefix + hash of arguments
        prefix = f"ch_{loss_type}_hybrid_alpha_{hybrid_alpha}_init_{embedding_init_method}_seq_len_{max_sequence_length}"
        cmd_hash8 = hashlib.sha1(args_for_hash.encode("utf-8")).hexdigest()[:8]
        output_dir = os.path.join("artifacts/experiments", f"{prefix}_{cmd_hash8}")

        # If the directory already exists, treat as duplicate and skip
        if os.path.isdir(output_dir) and not args.force:
            print(f"\033[33mSkipping: matching experiment already exists at:\033[0m {output_dir}")
            continue
        elif os.path.isdir(output_dir) and args.force:
            print(f"\033[35mForce enabled: re-running even though directory exists:\033[0m {output_dir}")

        # Compose full command with explicit output_dir
        base_cmd = (
            f"cd {workdir} && {python_path} scripts/activation_distillation.py {args_for_hash} --output_dir {output_dir} "
        )

        job_desc = (
            f"CH: {cmd_hash8} init={embedding_init_method} "
            f"hybrid_alpha={hybrid_alpha} "
            f"seq_len={max_sequence_length} seed={random_seed} "
            f"ckpt={model_checkpoint} "
            f"dtype={args.dtype} "
            f"fix_position_ids={fix_position_ids} "
            f"#{author_name} #rnd #multimodal @mrsndmn"
        )

        payload = {
            "script": base_cmd,
            "job_desc": job_desc,
            "env_variables": {
                "PYTHONPATH": "./src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
            },
            "instance_type": args.instance_type,
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": args.base_image,
            "n_workers": args.n_workers,
            "processes_per_worker": args.processes_per_worker,
        }

        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {base_cmd}\033[0m")
        jobs_dry += 1
        if args.dry:
            continue

        result = client.run_job(payload=payload)
        jobs_launched += 1
        print(embedding_init_method, random_seed, max_sequence_length, result)

    if args.dry:
        print(f"[DRY] Total jobs planned: {jobs_planned}")
        print(f"[DRY] Jobs printed (dry): {jobs_dry}")
    else:
        print(f"Total jobs planned: {jobs_planned}")
        print(f"Jobs launched: {jobs_launched}")
