import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full cramming results table.")
    parser.add_argument(
        "--tablefmt",
        default="plain",
        help="Tabulate table format (e.g., plain, github, latex, grid).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Full metrics
    ds_paths_hashes = [
        # Llama-3.2-1B
        "4e378cf3",  # 256
        "af92266e",  # 512
        # Llama-3.2-3B
        "7359e14b",  # 512
        "ef2ea924",  # 1024
        # Llama-3.1-8B
        # '', # 1024
        # '', # 1568
        # Pythia 160M
        "dbced9cc",  # 32
        "6a93af63",  # 64
        # Pythia 410M
        "328bdbfb",  # 96
        "22d7b7db",  # 128
        # Pythia 1.7B
        "f3296f56",  # 160
        "a1e58eb5",  # 256
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
        # Llama3
        "sl_4096_Llama-3.2-1B_lr_0.1",
        "sl_4096_Llama-3.2-3B_lr_0.1",
        "sl_4096_Meta-Llama-3.1-8B_lr_0.1",
        # pythia
        "sl_4096_pythia-160m_lr_0.5",
        "sl_4096_pythia-410m_lr_0.5",
        "sl_4096_pythia-1.4b_lr_0.5",
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
            max_tokens = summary.max_sequence_length
        else:
            accuracy = "1.0"
            max_tokens = to_mean_std_cell(
                summary.number_of_compressed_tokens,
                summary.number_of_compressed_tokens_std,
                use_latex=False,
            )

        result_table_rows.append([experiment, info_gain, max_tokens, accuracy])

    print(tabulate(result_table_rows, headers=columns, tablefmt=args.tablefmt))


if __name__ == "__main__":
    main()
