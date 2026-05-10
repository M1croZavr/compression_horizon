"""Compare a saved compressed_prefixes Dataset against paper-expected values."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset


def _load_expected(experiment_name: str) -> dict:
    """Load the paper-expected entry for `experiment_name` from expected.json."""
    expected_path = Path(__file__).parent / "expected.json"
    with open(expected_path) as f:
        all_expected = json.load(f)
    if experiment_name not in all_expected:
        raise KeyError(f"Experiment {experiment_name!r} not found in {expected_path}")
    return all_expected[experiment_name]


def _load_dataset(output_dir: str, trainer_type: str) -> Dataset:
    """Load the saved compressed_prefixes / progressive_prefixes dataset."""
    subdir = "progressive_prefixes" if trainer_type == "progressive" else "compressed_prefixes"
    dataset_path = os.path.join(output_dir, subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Did the training finish?")
    return Dataset.load_from_disk(dataset_path)


def _zscore(value: float, mean: float, std: float) -> float:
    """Normalized distance from `value` to `mean` in units of `std` (or NaN if std==0)."""
    return abs(value - mean) / std if std > 0 else float("nan")


def _verdict(z: float) -> str:
    """Map z-score distance to a 3-tier success indicator."""
    if np.isnan(z):
        return "n/a"
    if z <= 2.0:
        return "OK"
    if z <= 3.0:
        return "WARN"
    return "FAIL"


def _print_row(metric: str, ours: str, paper: str, delta: str, verdict: str) -> None:
    """Print a single comparison row in the OUR | PAPER | DELTA | VERDICT layout."""
    print(f"  {metric:<32}  {ours:>20}   {paper:>20}   {delta:>10}   {verdict}")


def _compare_metric(metric_name: str, ours_values: np.ndarray, expected_spec: dict) -> bool:
    """Compare one metric (mean ± std vs paper) and print a row. Returns True iff OK."""
    if "value" in expected_spec:
        # Fixed-value metric: e.g. compressed_tokens=32 (full cramming budget).
        paper_value = expected_spec["value"]
        tolerance = expected_spec.get("tolerance", 0)
        ours_mean = float(ours_values.mean())
        delta = abs(ours_mean - paper_value)
        verdict = "OK" if delta <= tolerance else "FAIL"
        _print_row(
            metric_name,
            f"{ours_mean:.3f}",
            f"{paper_value}",
            f"{delta:+.3f}",
            verdict,
        )
        return verdict == "OK"

    # Distributional metric: paper gives mean ± std.
    paper_mean = expected_spec["mean"]
    paper_std = expected_spec["std"]
    ours_mean = float(ours_values.mean())
    ours_std = float(ours_values.std())
    z = _zscore(ours_mean, paper_mean, paper_std)
    verdict = _verdict(z)
    _print_row(
        metric_name,
        f"{ours_mean:.3f} ± {ours_std:.3f}",
        f"{paper_mean:.3f} ± {paper_std:.3f}",
        f"z={z:.2f}",
        verdict,
    )
    return verdict == "OK"


def analyze(experiment_name: str, output_dir: str | None = None) -> bool:
    """Print paper-vs-ours comparison for one experiment. Returns True iff every metric is OK."""
    spec = _load_expected(experiment_name)
    output_dir = output_dir or os.path.join("artifacts", "thesis_reproduction", experiment_name)

    ds = _load_dataset(output_dir, spec["trainer_type"])
    rows = list(ds)

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {len(rows)}  (paper: {spec['num_samples']})")
    print()
    _print_row("metric", "ours", "paper", "delta", "verdict")
    print(f"  {'-'*32}  {'-'*20}   {'-'*20}   {'-'*10}   -------")

    metric_extractors = {
        "compressed_tokens": lambda r: r["num_compression_tokens"] * r["num_input_tokens"] / r["num_input_tokens"],
        "information_gain_bits": lambda r: r["information_gain_bits"],
        "final_convergence": lambda r: r["final_convergence"],
    }
    if spec["trainer_type"] == "full":
        # In full cramming the "compressed tokens" metric equals max_sequence_length (fixed budget).
        metric_extractors["compressed_tokens"] = lambda r: r["num_input_tokens"]

    all_ok = True
    for metric_name, expected_spec in spec["expected"].items():
        extractor = metric_extractors.get(metric_name)
        if extractor is None:
            print(f"  {metric_name:<32}  [no extractor]")
            continue
        values = np.array([extractor(r) for r in rows], dtype=np.float64)
        ok = _compare_metric(metric_name, values, expected_spec)
        all_ok = all_ok and ok

    print()
    print("Summary:", "all metrics OK" if all_ok else "some metrics drifted — investigate")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a saved cramming run against paper-expected values.")
    parser.add_argument(
        "--experiment", required=True, help="Experiment key from expected.json (e.g. full_cramming/pythia_160m)."
    )
    parser.add_argument(
        "--output_dir", default=None, help="Override output directory (default: artifacts/thesis_reproduction/<experiment>)."
    )
    args = parser.parse_args()
    analyze(args.experiment, args.output_dir)


if __name__ == "__main__":
    main()
