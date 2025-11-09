import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    checkpoints = [
        ["random", "ch_cross_entropy_init_random_fmivdb", "ch_cross_entropy_init_random_qjabzg"],
        ["mvnormal", "ch_cross_entropy_init_mvnormal_qghjjl", "ch_cross_entropy_init_mvnormal_kgkutv"],
    ]

    for exp_type, checkpoint1, checkpoint2 in checkpoints:
        result = client.run_job(
            payload={
                "script": f" cd {workdir} && {python_path} scripts/interpolation.py --dataset_path1 artifacts/experiments/{checkpoint1}/compressed_prefixes --dataset_path2 artifacts/experiments/{checkpoint2}/compressed_prefixes --bezier_steps 1000 --bezier_batch_t 100 --bezier_lr 0.1 --bezier_weight_decay 0.0 --bezier_order 2 --output_dir artifacts/interpolations/{exp_type}",
                "job_desc": f"CH: interpolate {checkpoint1} {checkpoint2} #{author_name} #multimodal @mrsndmn",
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
        )
        print(checkpoint1, checkpoint2, result)
