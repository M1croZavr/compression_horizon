import argparse
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
        "--embedding-init-methods",
        nargs="+",
        default=["mvnormal"],
        help="List of embedding initialization methods.",
    )
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=[42, 533, 100, 200],
        help="List of random seeds.",
    )
    parser.add_argument(
        "--max-sequence-lengths",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64],
        help="List of max sequence lengths.",
    )
    parser.add_argument(
        "--hybrid-alphas",
        nargs="+",
        default=[None, "1.0"],
        help='List of hybrid alpha values. Use "none" to disable hybrid and use cross-entropy loss.',
    )

    # General execution/runtime configuration
    parser.add_argument(
        "--python-path",
        default="/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python",
        help="Absolute path to the Python interpreter inside the target environment.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile name for training_job_api_from_profile.",
    )
    parser.add_argument(
        "--author-name",
        default="d.tarasov",
        help="Author name tag for job description.",
    )
    parser.add_argument(
        "--model-checkpoint",
        default="HuggingFaceTB/SmolLM2-1.7B",
        help="Base model checkpoint to use.",
    )
    parser.add_argument(
        "--model-checkpoints",
        nargs="+",
        default=None,
        help="List of model checkpoints to grid over (overrides --model-checkpoint).",
    )

    # Training defaults that were previously hardcoded
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--max-optimization-steps-per-sample", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--remove-unused-columns",
        action="store_true",
        default=False,
        help="Pass to set --remove_unused_columns True (default False).",
    )
    parser.add_argument(
        "--num-alignment-layers-nonhybrid",
        type=int,
        default=1,
        help="num_alignment_layers to use when hybrid is disabled.",
    )
    parser.add_argument(
        "--num-alignment-layers-hybrid",
        type=int,
        default=5,
        help="num_alignment_layers to use when hybrid is enabled.",
    )

    # Infra
    parser.add_argument("--instance-type", default="a100.1gpu")
    parser.add_argument(
        "--base-image",
        default="cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36",
    )
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--processes-per-worker", type=int, default=1)

    # Behavior
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
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

    for hybrid_alpha, embedding_init_method, random_seed, max_sequence_length, model_checkpoint in product(
        hybrid_alpha_values,
        args.embedding_init_methods,
        args.random_seeds,
        args.max_sequence_lengths,
        model_checkpoints,
    ):
        is_hybrid = hybrid_alpha is not None
        loss_type = "cosine" if is_hybrid else "cross_entropy"
        num_alignment_layers = args.num_alignment_layers_hybrid if is_hybrid else args.num_alignment_layers_nonhybrid

        remove_unused_columns_str = "True" if args.remove_unused_columns else "False"

        base_cmd = (
            f"cd {workdir} && {python_path} scripts/activation_distillation.py "
            f"--remove_unused_columns {remove_unused_columns_str} "
            f"--num_alignment_layers {num_alignment_layers} "
            f"--loss_type {loss_type} "
            f"--max_sequence_length {max_sequence_length} "
            f"--warmup_steps {args.warmup_steps} "
            f"--model_checkpoint {model_checkpoint} "
            f"--per_device_train_batch_size {args.per_device_train_batch_size} "
            f"--max_optimization_steps_per_sample {args.max_optimization_steps_per_sample} "
            f"--learning_rate {args.learning_rate} "
            f"--random_seed {random_seed} "
            f"--embedding_init_method {embedding_init_method} "
        )

        if is_hybrid:
            base_cmd += f"--hybrid_alpha {hybrid_alpha} "

        job_desc = (
            f"CH: compress init={embedding_init_method} "
            f"seq_len={max_sequence_length} seed={random_seed} "
            f"ckpt={model_checkpoint} "
            f"#{author_name} #rnd #multimodal @mrsndmn"
        )

        payload = {
            "script": base_cmd,
            "job_desc": job_desc,
            "env_variables": {
                "PYTHONPATH": "./src",
            },
            "instance_type": args.instance_type,
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": args.base_image,
            "n_workers": args.n_workers,
            "processes_per_worker": args.processes_per_worker,
        }

        if args.dry:
            print("[DRY] Would launch with payload:")
            print(payload)
            continue

        result = client.run_job(payload=payload)
        print(embedding_init_method, random_seed, max_sequence_length, result)
