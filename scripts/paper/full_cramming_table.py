import glob
import os

from scripts.results.results import (
    aggregate_non_progressive,
    aggregate_progressive,
    load_dataset_rows,
    to_mean_std_cell,
)
from tabulate import tabulate
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
        full_exp_name = os.path.join(full_exp_name, "compressed_prefixes")

        rows = load_dataset_rows(full_exp_name)
        summary = aggregate_non_progressive(full_exp_name, rows)
        assert summary is not None

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
        assert summary is not None

        summaries_progressive.append(summary)

    columns = ["Experiment", "Info Gain", "Max Tokens", "Accuracy"]

    def format_experiment_label(summary, fallback_label: str) -> str:
        parts = []
        if summary.model_checkpoint:
            parts.append(str(summary.model_checkpoint))
        if summary.max_sequence_length:
            parts.append(f"sl{summary.max_sequence_length}")
        if summary.dataset_type == "progressive_prefixes":
            parts.append("prog")
        label = "-".join(parts).strip()
        if not label:
            label = fallback_label

        return label

    all_summaries = summaries + summaries_progressive

    result_table_rows = []
    for summary in all_summaries:
        experiment = format_experiment_label(summary, fallback_label=str(summary.run_hash or ""))
        info_gain = to_mean_std_cell(
            summary.information_gain_bits_mean,
            summary.information_gain_bits_std,
            use_latex=False,
        )
        if summary.dataset_type != "progressive_prefixes":
            accuracy = to_mean_std_cell(
                summary.final_convergence_mean,
                summary.final_convergence_std,
                use_latex=False,
            )
        else:
            accuracy = "1.0"

        max_tokens = to_mean_std_cell(
            summary.max_sequence_length,
            summary.information_gain_bits_std,
            use_latex=False,
        )
        result_table_rows.append([experiment, info_gain, max_tokens, accuracy])

    print(tabulate(result_table_rows, headers=columns, tablefmt="plain"))


if __name__ == "__main__":
    main()
