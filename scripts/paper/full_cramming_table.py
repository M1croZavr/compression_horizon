import glob
import os

from scripts.results.results import aggregate_non_progressive, aggregate_progressive, load_dataset_rows
from tqdm.auto import tqdm


def main():

    # Full metrics
    ds_paths_hashes = [
        # Llama-3.2-1B
        "4e378cf3",  # 256
        "af92266e",  # 512
        # Llama-3.2-3B
        "7359e14b",  # 512
        "ef2ea924",  # 1024
        # Llama-3.1-8B
        # '', # 256
        # '', # 512
        # '', # 1568
    ]

    summaries = []
    for ds_paths_hashe in tqdm(ds_paths_hashes, desc="Processing Runs"):

        full_exp_name = glob.glob(f"artifacts/experiments/*{ds_paths_hashe}/")
        assert len(full_exp_name) == 1, f"experiments hashes must be unique: {full_exp_name}"
        full_exp_name = full_exp_name[0]

        rows = load_dataset_rows(os.path.join(full_exp_name, "compressed_prefixes"))
        summary = aggregate_non_progressive(ds_paths_hashe, rows)

        summaries.append(summary)

    # Progressive metrics
    ds_paths_progressive = [
        # Llama-3.2-1B
        "sl_4096_Llama-3.2-1B",
        # Llama-3.2-3B
        "sl_4096_Llama-3.2-3B",
        # Llama-3.1-8B
        "sl_4096_Meta-Llama-3.1-8B",
    ]

    summaries_progressive = []
    for ds_path in tqdm(ds_paths_progressive, desc="Processing Runs"):
        full_ds_path = f"artifacts/experiments_progressive/{ds_path}/progressive_prefixes/"
        rows = load_dataset_rows(full_ds_path)
        summary = aggregate_progressive(full_ds_path, rows)
        summaries_progressive.append(summary)

    # TODO fill the table and print it
    # columns = ["Experiment", "Info Gain", "Max Tokens", "Accuracy"]
    # result_table_rows = []


if __name__ == "__main__":
    main()
