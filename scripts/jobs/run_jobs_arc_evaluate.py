import argparse
import os
import sys
import time

from mls.manager.job.utils import training_job_api_from_profile

"""
python scripts/jobs/run_jobs_arc_evaluate.py \
  --limit_samples 512 \
  --num_compression_tokens 1 \
  --max_optimization_steps 1000 \
  --learning_rate 0.1 \
  --batch_size 32 \
  --model Llama-3.1 SmolLM2-1.7B gemma-3-4b-pt EleutherAI/pythia-1.4b
"""


def get_in_progress_jobs(client, region, statuses=None):
    """
    Example:
        from sentence_attention.integration.job import get_in_progress_jobs
        from mls.manager.job.utils import training_job_api_from_profile

        client, extra_options = training_job_api_from_profile("default")
        in_progress_jobs = get_in_progress_jobs(client, extra_options["region"])

    """

    all_in_progress_jobs = []

    if statuses is None:
        statuses = ["Pending", "Running"]

    for non_final_status in statuses:
        while True:
            non_final_jobs = client.get_list_jobs(
                region=region,
                allocation_name="alloc-officecds-multimodal-2-sr004",
                status=non_final_status,
                limit=1000,
                offset=0,
            )
            if "jobs" in non_final_jobs:
                break
            elif "error_code" in non_final_jobs and non_final_jobs["error_code"] == [
                32,
                20,
            ]:  # no active session, access_token expired
                print("Error:", non_final_jobs, "try again")
                time.sleep(5)
                client, _ = training_job_api_from_profile("default")
            else:
                raise ValueError("Unknown error in get_in_progress_jobs:", non_final_jobs)

        all_in_progress_jobs.extend(non_final_jobs["jobs"])

    return all_in_progress_jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch ARC compression evaluation jobs.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--arc_split",
        type=str,
        default="ARC-Easy",
        choices=["ARC-Easy", "ARC-Challenge"],
        help="ARC dataset split to use (default: ARC-Easy)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter models by name (substring match). Can specify multiple models. Matches against model name or full checkpoint path.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype to use: auto | float32/fp32 | bfloat16/bf16 | float16/fp16. If not specified, defaults to 'bf16' and is not included in output dir.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate. If not specified, defaults to 100 and is not included in output dir.",
    )
    parser.add_argument(
        "--num_compression_tokens",
        type=int,
        default=None,
        help="Number of compression tokens. If not specified, defaults to 1 and is not included in output dir.",
    )
    parser.add_argument(
        "--max_optimization_steps",
        type=int,
        default=None,
        help="Maximum optimization steps for compression. If not specified, defaults to 1000 and is not included in output dir.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for compression optimization. If not specified, defaults to 0.01 and is not included in output dir.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for compression and evaluation. If not specified, defaults to 4 and is not included in output dir.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not specified, defaults to 42 and is not included in output dir.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help="Loss type for optimization: cross_entropy | l2 | l1 | cosine. If not specified, defaults to cross_entropy.",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="If set and loss_type != cross_entropy, adds hybrid_alpha * alignment_loss to CE loss.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=None,
        help="Number of layers to align (0 = all layers). If not specified, defaults to 0.",
    )
    parser.add_argument(
        "--inverted_alignment",
        action="store_true",
        help="If set, aligns the last num_alignment_layers instead of the first.",
    )
    args = parser.parse_args()
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    # Get in-progress jobs once at the start
    region = extra_options["region"]
    in_progress_jobs = get_in_progress_jobs(client, region)
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    checkpoints = [
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

    # Filter checkpoints by --model flag if provided
    if args.model:
        model_filters = [m.lower() for m in args.model]
        filtered_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint_lower = checkpoint.lower()
            model_name = checkpoint.split("/")[-1].lower() if "/" in checkpoint else checkpoint_lower
            # Match if any filter is found in full checkpoint path or model name
            if any(filt in checkpoint_lower or filt in model_name for filt in model_filters):
                filtered_checkpoints.append(checkpoint)
        checkpoints = filtered_checkpoints
        if not checkpoints:
            print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
            sys.exit(0)

    for model_checkpoint in checkpoints:
        # Build experiment suffix with split name
        split_suffix = args.arc_split.replace("-", "_")  # ARC-Easy -> ARC_Easy, ARC-Challenge -> ARC_Challenge
        exp_suffix = f"arc_{split_suffix}_{model_checkpoint.split('/')[1]}"

        # Build command arguments with defaults
        limit_samples = args.limit_samples if args.limit_samples is not None else 100
        num_compression_tokens = args.num_compression_tokens if args.num_compression_tokens is not None else 1
        max_optimization_steps = args.max_optimization_steps if args.max_optimization_steps is not None else 1000
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.01
        batch_size = args.batch_size if args.batch_size is not None else 4
        dtype = args.dtype if args.dtype is not None else "bf16"
        loss_type = args.loss_type if args.loss_type is not None else "cross_entropy"
        num_alignment_layers = args.num_alignment_layers if args.num_alignment_layers is not None else 0

        cmd_args = [
            f"--model_checkpoint {model_checkpoint}",
            f"--arc_split {args.arc_split}",
            f"--limit_samples {limit_samples}",
            f"--num_compression_tokens {num_compression_tokens}",
            f"--max_optimization_steps {max_optimization_steps}",
            f"--learning_rate {learning_rate}",
            f"--batch_size {batch_size}",
            f"--dtype {dtype}",
            f"--loss_type {loss_type}",
            f"--num_alignment_layers {num_alignment_layers}",
        ]
        if args.hybrid_alpha is not None:
            cmd_args.append(f"--hybrid_alpha {args.hybrid_alpha}")
        if args.inverted_alignment:
            cmd_args.append("--inverted_alignment")

        # Add random_seed if specified (non-default)
        if args.random_seed is not None and args.random_seed != 42:
            cmd_args.append(f"--random_seed {args.random_seed}")
            exp_suffix = f"{exp_suffix}_seed_{args.random_seed}"

        # Add limit_samples to output dir if specified (non-default)
        if args.limit_samples is not None and args.limit_samples != 100:
            exp_suffix = f"{exp_suffix}_samples_{args.limit_samples}"

        # Add num_compression_tokens to output dir if specified (non-default)
        if args.num_compression_tokens is not None and args.num_compression_tokens != 1:
            exp_suffix = f"{exp_suffix}_tokens_{args.num_compression_tokens}"

        # Add max_optimization_steps to output dir if specified (non-default)
        if args.max_optimization_steps is not None and args.max_optimization_steps != 1000:
            exp_suffix = f"{exp_suffix}_steps_{args.max_optimization_steps}"

        # Add learning_rate to output dir if specified (non-default)
        if args.learning_rate is not None and args.learning_rate != 0.01:
            exp_suffix = f"{exp_suffix}_lr_{args.learning_rate}"

        # Add batch_size to output dir if specified (non-default)
        if args.batch_size is not None and args.batch_size != 4:
            exp_suffix = f"{exp_suffix}_batch_{args.batch_size}"

        # Add dtype to output dir if specified (non-default)
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

        out_dir_name = f"artifacts/arc_evaluation/{exp_suffix}"
        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        # Add output_dir to command
        cmd_args.append(f"--output_dir {out_dir_name}")
        script = f" cd {workdir} && {python_path} -m scripts.arc_compress_evaluate  {' '.join(cmd_args)}"
        job_desc = f"CH: arc {args.arc_split} eval {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

        # Check if job with same description already exists in queue
        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue with description:\033[0m {job_desc}")
            continue

        payload = {
            "script": script,
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
            "n_workers": 1,  # Количество воркеров.
            "processes_per_worker": 1,  # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
        }

        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name, result)
