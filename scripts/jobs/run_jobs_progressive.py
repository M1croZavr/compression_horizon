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

    for model_checkpoint in checkpoints:
        exp_suffix = f"sl_{max_seq_len}_{model_checkpoint.split('/')[1]}"
        out_dir_name = f"artifacts/experiments_progressive/{exp_suffix}"
        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        script = f" cd {workdir} && {python_path} scripts/activation_distillation.py  --remove_unused_columns False  --num_alignment_layers 1 --loss_type cross_entropy --max_sequence_length {max_seq_len} --warmup_steps 100 --model_checkpoint {model_checkpoint} --per_device_train_batch_size 1 --max_optimization_steps_per_sample {max_optimization_steps_per_sample} --learning_rate 0.01  --progressive_train 1 --embedding_init_method random0.02 --output_dir {out_dir_name} --limit_dataset_items 10"
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
