import argparse
import os

from mls.manager.job.utils import training_job_api_from_profile

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
        help="Adam beta2 parameter. Default: 0.9.",
    )
    args = parser.parse_args()
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    checkpoints = [
        # "HuggingFaceTB/SmolLM2-1.7B",
        # "unsloth/Llama-3.2-3B",
        # "Qwen/Qwen3-4B",
        # "unsloth/Meta-Llama-3.1-8B",
        # "Qwen/Qwen3-8B",
        # "allenai/OLMo-1B-hf",
        "allenai/Olmo-3-1025-7B",
    ]
    # checkpoints = []

    max_seq_len = 2048
    max_optimization_steps_per_sample = 2500

    # Default values for optimizer parameters
    default_optim = "adamw_torch"
    default_adam_beta1 = 0.9
    default_adam_beta2 = 0.9

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

        # Add optimizer parameters to command if specified
        optim_params = []
        if args.optim is not None:
            cmd_args.append(f"--optim {args.optim}")
            # Only add to output_dir suffix if non-default
            if args.optim != default_optim:
                optim_params.append(f"opt_{args.optim}")
        if args.adam_beta1 is not None:
            cmd_args.append(f"--adam_beta1 {args.adam_beta1}")
            # Only add to output_dir suffix if non-default
            if args.adam_beta1 != default_adam_beta1:
                optim_params.append(f"b1_{args.adam_beta1}")
        if args.adam_beta2 is not None:
            cmd_args.append(f"--adam_beta2 {args.adam_beta2}")
            # Only add to output_dir suffix if non-default
            if args.adam_beta2 != default_adam_beta2:
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
        job_desc = f"CH: progressive {exp_suffix} #{author_name} #multimodal @mrsndmn"

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
