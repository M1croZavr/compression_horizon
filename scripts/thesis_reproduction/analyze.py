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
        # In progressive cramming compressed_tokens is the achieved prefix length per sample.
        extractors["compressed_tokens"] = lambda r: r["stage_seq_len"]
    return extractors


def _aggregate_rows_per_sample(rows: list[dict], trainer_type: str) -> list[dict]:
    """Reduce per-stage rows to one row per sample (no-op for full cramming).

    Full cramming saves one row per sample already. Progressive cramming saves
    one row per (sample, stage) — we collapse to the row of the *final converged
    stage* per sample (i.e. the largest stage_seq_len with final_convergence == 1.0).
    Samples that never converge contribute the row with the largest reached
    stage_seq_len (kept for diagnostic visibility).
    """
    if trainer_type != "progressive":
        return rows
    by_sample: dict[int, list[dict]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_id"]), []).append(row)
    aggregated: list[dict] = []
    for sample_id in sorted(by_sample):
        sample_rows = by_sample[sample_id]
        converged = [r for r in sample_rows if r.get("final_convergence") == 1.0]
        candidates = converged if converged else sample_rows
        aggregated.append(max(candidates, key=lambda r: r["stage_seq_len"]))
    return aggregated


def _split_metrics(spec: dict) -> tuple[dict, dict]:
    """Partition expected.json metrics into (configuration, measured) by their `kind` flag."""
    configuration: dict = {}
    measured: dict = {}
    for metric_name, expected_spec in spec["expected"].items():
        kind = expected_spec.get("kind", "measured")
        target = configuration if kind == "configuration" else measured
        target[metric_name] = expected_spec
    return configuration, measured


def _analyze_attention_hijacking(experiment_name: str, spec: dict, output_dir: str) -> bool:
    """Compare a saved attention_hijacking.json against paper Table 3 qualitative values.

    Paper Section 5.5 reports compression-mass / BOS-mass / correlation for a
    different model size (SmolLM2-1.7B). We use it as a qualitative reference:
    the experiment passes if (a) compression_mass >= 30% (clear hijacking) and
    (b) correlation >= 0.5 (per-layer profile shape matches BOS pattern).
    """
    json_path = Path(output_dir) / "attention_hijacking.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run scripts/thesis_reproduction/run_attention_hijacking.py first.")
    with open(json_path) as f:
        result = json.load(f)
    summary = result["summary"]
    expected_summary = spec["expected"]

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {summary['num_samples']}  (paper: {spec['num_samples']})")
    print(f"Note:       paper reference row is {spec.get('reference_model', spec['model'])} — qualitative comparison only.")

    print()
    print("Table-3 statistics (qualitative paper comparison):")
    _print_header()

    def _row(metric_name: str, ours: dict, paper: dict) -> None:
        ours_str = f"{ours['mean']:.3f} ± {ours['std']:.3f}"
        paper_str = f"{paper['mean']:.3f} ± {paper['std']:.3f}"
        z = _zscore(ours["mean"], paper["mean"], paper["std"])
        verdict = _verdict(z)
        _print_row(metric_name, ours_str, paper_str, f"z={z:.2f}", verdict)

    _row(
        "compression_mass_pct",
        summary["compression_mass"],
        expected_summary["compression_mass"],
    )
    _row("bos_mass_pct", summary["bos_mass"], expected_summary["bos_mass"])
    _row("correlation", summary["correlation"], expected_summary["correlation"])

    qual = spec.get("qualitative", {"min_compression_mass": 30.0, "min_correlation": 0.5})
    min_mass = float(qual["min_compression_mass"])
    min_corr = float(qual["min_correlation"])
    mass_ok = summary["compression_mass"]["mean"] >= min_mass
    corr_ok = summary["correlation"]["mean"] >= min_corr
    print()
    print(
        f"Qualitative gate: compression_mass ≥ {min_mass:.1f}% "
        f"({'OK' if mass_ok else 'FAIL'} — got {summary['compression_mass']['mean']:.2f}%) "
        f"AND correlation ≥ {min_corr:.2f} "
        f"({'OK' if corr_ok else 'FAIL'} — got {summary['correlation']['mean']:.4f})"
    )
    passed = mass_ok and corr_ok
    print()
    print(
        "Summary:",
        ("attention-hijacking pattern confirmed" if passed else "attention-hijacking pattern NOT confirmed"),
    )
    return passed


def analyze(experiment_name: str, output_dir: str | None = None) -> bool:
    """Print paper-vs-ours comparison for one experiment. Returns True iff every measured metric is OK."""
    spec = _load_expected(experiment_name)
    output_dir = output_dir or os.path.join("artifacts", "thesis_reproduction", experiment_name)

    if spec.get("analyzer") == "attention_hijacking":
        return _analyze_attention_hijacking(experiment_name, spec, output_dir)

    ds = _load_dataset(output_dir, spec["trainer_type"])
    rows = _aggregate_rows_per_sample(list(ds), spec["trainer_type"])
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
    print(
        "Summary:",
        "all measured metrics OK" if all_ok else "some metrics drifted — investigate",
    )
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a saved cramming run against paper-expected values.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment key from expected.json (e.g. full_cramming/pythia_160m).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory (default: artifacts/thesis_reproduction/<experiment>).",
    )
    args = parser.parse_args()
    analyze(args.experiment, args.output_dir)


if __name__ == "__main__":
    main()
