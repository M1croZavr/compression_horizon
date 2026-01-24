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

    experiments_list = [
        # Llama-3.2-1B
        {"train": "full", "id": "4e378cf3"},  # 256
        {"train": "full", "id": "af92266e"},  # 512
        {"train": "progr", "id": "sl_4096_Llama-3.2-1B_lr_0.1"},
        # Llama-3.2-3B
        {"train": "full", "id": "7359e14b"},  # 512
        {"train": "full", "id": "ef2ea924"},  # 1024
        {"train": "progr", "id": "sl_4096_Llama-3.2-3B_lr_0.1"},
        # Llama-3.1-8B
        {"train": "full", "id": "dfbe32b8"},  # 1024
        {"train": "full", "id": "b5aef07e"},  # 1568
        {"train": "progr", "id": "sl_4096_Meta-Llama-3.1-8B_lr_0.1"},
        # Pythia 160M
        {"train": "full", "id": "dbced9cc"},  # 32
        {"train": "full", "id": "6a93af63"},  # 64
        {"train": "progr", "id": "sl_4096_pythia-160m_lr_0.5"},
        # Pythia 410M
        {"train": "full", "id": "328bdbfb"},  # 96
        {"train": "full", "id": "22d7b7db"},  # 128
        {"train": "progr", "id": "sl_4096_pythia-410m_lr_0.5"},
        # Pythia 1.4B
        {"train": "progr", "id": "sl_4096_pythia-1.4b_lr_0.5"},
        # Pythia 1.7B
        {"train": "full", "id": "f3296f56"},  # 160
        {"train": "full", "id": "a1e58eb5"},  # 256
    ]

    columns = ["Exp", "Type", "Tokens", "Info Gain", "Acc"]

    def format_experiment_label(summary, fallback_label: str) -> str:
        parts = []
        if summary.model_checkpoint:
            parts.append(str(summary.model_checkpoint))

        label = "-".join(parts).strip()
        if not label:
            label = fallback_label

        return label

    ordered_summaries = []
    for experiment in tqdm(experiments_list, desc="Processing Runs"):
        rows = None
        summary = None
        if experiment["train"] == "full":
            full_exp_name = glob.glob(f"artifacts/experiments/*{experiment['id']}/")
            assert len(full_exp_name) == 1, f"experiments hashes must be unique: {full_exp_name}"
            full_exp_name = os.path.join(full_exp_name[0], "compressed_prefixes")
            if os.path.isdir(full_exp_name):
                rows = load_dataset_rows(full_exp_name)
                summary = aggregate_non_progressive(full_exp_name, rows)
        elif experiment["train"] == "progr":
            full_ds_path = f"artifacts/experiments_progressive/{experiment['id']}/progressive_prefixes/"
            if os.path.isdir(full_ds_path):
                rows = load_dataset_rows(full_ds_path)
                summary = aggregate_progressive(full_ds_path, rows)
        else:
            raise ValueError(f"Unknown train type: {experiment['train']}")

        if summary is None:
            print("Failed to load:", experiment)
            continue

        ordered_summaries.append(summary)

    result_table_rows = []
    for summary in ordered_summaries:
        experiment = format_experiment_label(summary, fallback_label=str(summary.run_hash or ""))
        info_gain = to_mean_std_cell(
            summary.information_gain_bits_mean,
            summary.information_gain_bits_std,
            use_latex=(args.tablefmt == "latex"),
            float_precision=0,
        )
        is_progressive = summary.dataset_type == "progressive_prefixes"
        train_type = "progr" if is_progressive else "full"
        if not is_progressive:
            accuracy = to_mean_std_cell(
                summary.final_convergence_mean,
                summary.final_convergence_std,
                use_latex=(args.tablefmt == "latex"),
                float_precision=3,
            )
            max_tokens = summary.max_sequence_length
        else:
            accuracy = "1.0"
            max_tokens = to_mean_std_cell(
                summary.number_of_compressed_tokens,
                summary.number_of_compressed_tokens_std,
                use_latex=(args.tablefmt == "latex"),
                float_precision=3,
            )

        result_table_rows.append([experiment, train_type, max_tokens, info_gain, accuracy])

    result = tabulate(result_table_rows, headers=columns, tablefmt=args.tablefmt)
    result = result.replace("\\textbackslash{}", "\\")
    result = result.replace("\$", "$")
    result = result.replace("\\{", "{")
    result = result.replace("\\}", "}")
    print(result)


if __name__ == "__main__":
    main()
