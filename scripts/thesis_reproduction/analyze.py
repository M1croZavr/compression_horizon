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


def _print_header() -> None:
    """Print the comparison-table header row."""
    _print_row("metric", "ours", "paper", "delta", "verdict")
    print(f"  {'-'*32}  {'-'*20}   {'-'*20}   {'-'*10}   -------")


def _compare_configuration(metric_name: str, ours_values: np.ndarray, expected_spec: dict) -> bool:
    """Sanity-check an input parameter (e.g. fixed token budget). Tautological in full cramming."""
    paper_value = expected_spec["value"]
    tolerance = expected_spec.get("tolerance", 0)
    ours_mean = float(ours_values.mean())
    delta = abs(ours_mean - paper_value)
    verdict = "OK" if delta <= tolerance else "FAIL"
    _print_row(metric_name, f"{ours_mean:.3f}", f"{paper_value}", f"{delta:+.3f}", verdict)
    return verdict == "OK"


def _compare_measured(metric_name: str, ours_values: np.ndarray, expected_spec: dict) -> bool:
    """Compare a measured metric (mean ± std vs paper). Returns True iff within ±2σ."""
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


def _build_metric_extractors(trainer_type: str):
    """Map metric name → callable extracting per-sample value from a row dict."""
    extractors = {
        "information_gain_bits": lambda r: r["information_gain_bits"],
        "final_convergence": lambda r: r["final_convergence"],
    }
    if trainer_type == "full":
        # In full cramming compressed_tokens equals the fixed budget (= max_sequence_length).
        extractors["compressed_tokens"] = lambda r: r["num_input_tokens"]
    elif trainer_type == "progressive":
        # In progressive cramming compressed_tokens is the achieved length per sample.
        extractors["compressed_tokens"] = lambda r: r["stage_seq_len"]
    return extractors


def _split_metrics(spec: dict) -> tuple[dict, dict]:
    """Partition expected.json metrics into (configuration, measured) by their `kind` flag."""
    configuration: dict = {}
    measured: dict = {}
    for metric_name, expected_spec in spec["expected"].items():
        kind = expected_spec.get("kind", "measured")
        target = configuration if kind == "configuration" else measured
        target[metric_name] = expected_spec
    return configuration, measured


def analyze(experiment_name: str, output_dir: str | None = None) -> bool:
    """Print paper-vs-ours comparison for one experiment. Returns True iff every measured metric is OK."""
    spec = _load_expected(experiment_name)
    output_dir = output_dir or os.path.join("artifacts", "thesis_reproduction", experiment_name)

    ds = _load_dataset(output_dir, spec["trainer_type"])
    rows = list(ds)
    extractors = _build_metric_extractors(spec["trainer_type"])
    configuration_metrics, measured_metrics = _split_metrics(spec)

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {len(rows)}  (paper: {spec['num_samples']})")

    if configuration_metrics:
        print()
        print("Configuration (input parameters; tautological match expected):")
        _print_header()
        for metric_name, expected_spec in configuration_metrics.items():
            extractor = extractors.get(metric_name)
            if extractor is None:
                print(f"  {metric_name:<32}  [no extractor]")
                continue
            values = np.array([extractor(r) for r in rows], dtype=np.float64)
            _compare_configuration(metric_name, values, expected_spec)

    print()
    print("Measured metrics (real outputs of the optimization):")
    _print_header()
    all_ok = True
    for metric_name, expected_spec in measured_metrics.items():
        extractor = extractors.get(metric_name)
        if extractor is None:
            print(f"  {metric_name:<32}  [no extractor]")
            continue
        values = np.array([extractor(r) for r in rows], dtype=np.float64)
        ok = _compare_measured(metric_name, values, expected_spec)
        all_ok = all_ok and ok

    print()
    print("Summary:", "all measured metrics OK" if all_ok else "some metrics drifted — investigate")
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
