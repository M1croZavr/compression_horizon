import argparse
import os
import sys
import time

from mls.manager.job.utils import training_job_api_from_profile


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
    parser = argparse.ArgumentParser(description="Launch progressive compression_horizon training jobs.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer to use (e.g., 'adamw_torch', 'sgd'). Default: 'adamw_torch'.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=None,
        help="Adam beta1 parameter. Default: 0.9.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=None,
        help="Adam beta2 parameter. Default: 0.999.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter models by name (substring match). Can specify multiple models. Matches against model name or full checkpoint path.",
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
        "unsloth/Llama-3.2-3B",
        "Qwen/Qwen3-4B",
        "unsloth/Meta-Llama-3.1-8B",
        "Qwen/Qwen3-8B",
        "allenai/OLMo-1B-hf",
        "allenai/Olmo-3-1025-7B",
    ]
    # checkpoints = []

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

    max_seq_len = 2048
    max_optimization_steps_per_sample = 2500

    for model_checkpoint in checkpoints:
        exp_suffix = f"sl_{max_seq_len}_{model_checkpoint.split('/')[1]}"

        # Build command arguments
        cmd_args = [
            "--remove_unused_columns False",
            "--num_alignment_layers 1",
            "--loss_type cross_entropy",
            f"--max_sequence_length {max_seq_len}",
            "--warmup_steps 100",
            f"--model_checkpoint {model_checkpoint}",
            "--per_device_train_batch_size 1",
            f"--max_optimization_steps_per_sample {max_optimization_steps_per_sample}",
            "--learning_rate 0.01",
            "--progressive_train 1",
            "--embedding_init_method random0.02",
            "--limit_dataset_items 10",
        ]

        # Add optimizer parameters if specified (non-default)
        optim_params = []
        if args.optim is not None:
            cmd_args.append(f"--optim {args.optim}")
            optim_params.append(f"opt_{args.optim}")
        if args.adam_beta1 is not None:
            cmd_args.append(f"--adam_beta1 {args.adam_beta1}")
            optim_params.append(f"b1_{args.adam_beta1}")
        if args.adam_beta2 is not None:
            cmd_args.append(f"--adam_beta2 {args.adam_beta2}")
            optim_params.append(f"b2_{args.adam_beta2}")

        # Update exp_suffix if optimizer parameters are non-default
        if optim_params:
            optim_suffix = "_".join(optim_params)
            exp_suffix = f"{exp_suffix}_{optim_suffix}"

        out_dir_name = f"artifacts/experiments_progressive/{exp_suffix}"
        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        # Add output_dir to command
        cmd_args.append(f"--output_dir {out_dir_name}")
        script = f" cd {workdir} && {python_path} scripts/activation_distillation.py  {' '.join(cmd_args)}"
        job_desc = f"CH: progressive {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

        # Check if job with same description already exists in queue
        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue with description:\033[0m {job_desc}")
            continue

        payload = {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {
                "PYTHONPATH": "./src",
            },
            "instance_type": "a100.1gpu",
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36",
            "n_workers": 1,  # Количество воркеров.
            "processes_per_worker": 1,  # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
        }

        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name)
