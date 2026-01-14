import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
from tabulate import tabulate
from tqdm.auto import tqdm


def load_progressive_dataset(dataset_path: str) -> Dataset:
    """Load a progressive embeddings dataset from disk."""
    return Dataset.load_from_disk(dataset_path)


def flatten_embedding(row: Dict[str, Any]) -> np.ndarray:
    """Flatten embedding from a dataset row."""
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    return emb.reshape(-1).detach().cpu().numpy()


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter dataset records by sample_id and/or stage_index."""
    rows: List[Dict[str, Any]] = []
    for i in tqdm(range(len(ds)), desc="Filtering records"):
        r = ds[i]
        if sample_id is not None and int(r.get("sample_id", -1)) != int(sample_id):
            continue
        if stage_index is not None and int(r.get("stage_index", -1)) != int(stage_index):
            continue
        rows.append(r)
    return rows


def collate_stages_by_sample(
    rows: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group rows by sample_id and sort by stage_index."""
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("sample_id", -1))
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(r)
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: int(x.get("stage_index", 0)))
    return by_sid


def compute_num_pca_explained_99_var(embeddings: List[np.ndarray]) -> float:
    """Compute cumulative explained variance using PCA with 4 components.

    Args:
        embeddings: List of flattened embedding arrays

    Returns:
        Cumulative explained variance ratio (0.0 to 1.0), or NaN if not computable
    """
    if len(embeddings) < 2:
        return float("nan")

    # Stack embeddings: [n_samples, n_features]
    X = np.stack(embeddings, axis=0)

    # Need at least 2 samples for PCA
    if X.shape[0] < 2:
        return float("nan")

    n_samples, n_features = X.shape

    # Fit PCA with up to 4 components
    max_PCA_components = min(512, n_samples - 1, n_features)
    if max_PCA_components < 1:
        return float("nan")

    pca = PCA(n_components=max_PCA_components, random_state=42)
    pca.fit(X)
    explained_var_ratio = pca.explained_variance_ratio_

    # Return cumulative explained variance
    cumulative_var = np.cumsum(explained_var_ratio)
    num_pca_for99_var = (cumulative_var < 0.99).sum()
    if num_pca_for99_var == max_PCA_components:
        num_pca_for99_var = -1

    return num_pca_for99_var


def extract_trajectory(
    dataset_path: str,
    sample_id: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any], np.ndarray]:
    """Extract embedding trajectory from a dataset.

    Args:
        dataset_path: Path to the progressive embeddings dataset
        sample_id: Optional sample_id to filter. If None, uses first available sample.

    Returns:
        Tuple of (embeddings array [n_stages, n_features], labels list, statistics dict, final_embedding)
        Statistics dict contains: 'num_embeddings', 'total_steps', 'trajectory_length'
        final_embedding is the last embedding in the trajectory
    """
    ds = load_progressive_dataset(dataset_path)
    rows = filter_records(ds, sample_id=sample_id)

    if not rows:
        raise ValueError(f"No records found in {dataset_path}")

    # Group by sample_id
    by_sid = collate_stages_by_sample(rows)

    # If sample_id was specified, use it; otherwise use first available
    if sample_id is not None:
        if sample_id not in by_sid:
            raise ValueError(f"Sample {sample_id} not found in {dataset_path}")
        stages = by_sid[sample_id]
    else:
        # Use first available sample
        first_sid = sorted(by_sid.keys())[0]
        stages = by_sid[first_sid]
        sample_id = first_sid

    # Extract embeddings in order and collect statistics
    embeddings = []
    labels = []
    total_steps = 0
    for stage in stages:
        emb = flatten_embedding(stage)
        embeddings.append(emb)
        stage_seq_len = int(stage.get("stage_seq_len", -1))
        # stage_idx = int(stage.get("stage_index", -1))
        labels.append(f"L{stage_seq_len}")
        # Sum up optimization steps
        steps = int(stage.get("steps_taken", 0))
        total_steps += steps

    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found for sample {sample_id} in {dataset_path}")

    X = np.stack(embeddings, axis=0)
    final_embedding = embeddings[-1]  # Last embedding

    # Compute trajectory length as sum of linear distances between consecutive points
    trajectory_length = 0.0
    if len(embeddings) > 1:
        for i in range(len(embeddings) - 1):
            dist = np.linalg.norm(embeddings[i + 1] - embeddings[i])
            trajectory_length += dist

    num_pca_explained_99_var = compute_num_pca_explained_99_var(embeddings)

    stats = {
        "num_embeddings": len(embeddings),
        "total_steps": total_steps,
        "trajectory_length": trajectory_length,
        "num_pca_for99_var": num_pca_explained_99_var,
    }
    return X, labels, stats, final_embedding


def plot_pca_trajectories(
    trajectories: List[np.ndarray],
    checkpoint_names: List[str],
    outfile: str,
    n_components: int = 2,
    show_labels: bool = False,
    labels_list: Optional[List[List[str]]] = None,
):
    """Plot multiple embedding trajectories on a single PCA plot.

    Args:
        trajectories: List of embedding arrays, each of shape [n_stages, n_features]
        checkpoint_names: List of names for each trajectory (for legend)
        outfile: Output file path
        n_components: Number of PCA components to use (2 or 4)
        show_labels: Whether to show stage labels on points
        labels_list: Optional list of label lists for each trajectory
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Combine all embeddings to fit a single PCA
    all_embeddings = np.vstack(trajectories)
    n_samples, n_features = all_embeddings.shape

    if n_samples < 2 or n_features < 2:
        raise ValueError(f"Insufficient data: {n_samples} samples, {n_features} features")

    n_components = min(n_components, n_samples - 1, n_features)
    if n_components < 2:
        raise ValueError(f"Cannot compute {n_components} components")

    # Fit PCA on all embeddings
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_embeddings)
    explained_var = pca.explained_variance_ratio_

    # Transform each trajectory
    transformed_trajectories = []
    for traj in trajectories:
        traj_transformed = pca.transform(traj)
        transformed_trajectories.append(traj_transformed)

    # Create distinct colors for checkpoints
    # Use a predefined set of highly distinct colors with maximum hue separation
    distinct_colors = [
        "#E6194B",  # bright red
        "#3CB44B",  # bright green
        "#FFE119",  # bright yellow
        "#4363D8",  # bright blue
        "#F58231",  # bright orange
        "#911EB4",  # bright purple
        "#42D4F4",  # bright cyan
        "#F032E6",  # bright magenta
        "#BFEF45",  # lime green
        "#FABED4",  # light pink
        "#469990",  # teal
        "#DCBEFF",  # light purple
        "#9A6324",  # brown
        "#FFFAC8",  # beige
        "#800000",  # maroon
        "#000075",  # navy
        "#A9A9A9",  # gray
        "#000000",  # black
    ]
    # Cycle through distinct colors if we have more trajectories than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(trajectories))]

    if n_components == 2:
        # Single 2D plot
        plt.figure(figsize=(10, 8))
        legend_handles = []
        for idx, (traj_transformed, name, color) in enumerate(zip(transformed_trajectories, checkpoint_names, colors)):
            x_data = traj_transformed[:, 0]
            y_data = traj_transformed[:, 1]

            # Plot trajectory line (without label)
            plt.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1.5, linestyle="--")

            # Plot points
            plt.scatter(x_data, y_data, c=[color], s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

            # Create legend handle with scatter marker
            legend_handles.append(plt.scatter([], [], c=color, s=60, alpha=0.7, edgecolors="black", linewidths=0.5, label=name))

            # Add labels if requested
            if show_labels and labels_list is not None and idx < len(labels_list):
                labels = labels_list[idx]
                labeled_positions = []
                for k, lab in enumerate(labels):
                    if k >= len(x_data):
                        continue
                    # Check if there's already a labeled point within distance < 0.5
                    should_label = True
                    for labeled_pos in labeled_positions:
                        dist = np.linalg.norm([x_data[k] - labeled_pos[0], y_data[k] - labeled_pos[1]])
                        if dist < 0.5:
                            should_label = False
                            break
                    if should_label:
                        plt.text(x_data[k], y_data[k], lab, fontsize=12, ha="left", va="bottom", color=color)
                        labeled_positions.append([x_data[k], y_data[k]])

            # Mark start and end points
            if len(x_data) > 0:
                plt.scatter(x_data[0], y_data[0], c=[color], s=150, marker="o", edgecolors="black", linewidths=2, zorder=5)
                plt.scatter(x_data[-1], y_data[-1], c=[color], s=150, marker="s", edgecolors="black", linewidths=2, zorder=5)

        plt.xlabel(f"PC1 ({explained_var[0]:.4f})", fontsize=18)
        plt.ylabel(f"PC2 ({explained_var[1]:.4f})", fontsize=18)
        plt.title(
            f"PCA Trajectories Comparison\nCumulative variance: {explained_var.sum():.4f}",
            fontsize=20,
        )
        plt.legend(handles=legend_handles, loc="best", fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved 2D PCA plot to: {outfile}")

    elif n_components == 4:
        # Multiple subplots for 4 components (similar to plot_pca_4_components)
        pairs = [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]
        n_pairs = len(pairs)

        n_cols = 3
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        legend_handles = []
        for pair_idx, (i, j) in enumerate(pairs):
            ax = axes[pair_idx]

            for idx, (traj_transformed, name, color) in enumerate(zip(transformed_trajectories, checkpoint_names, colors)):
                x_data = traj_transformed[:, i]
                y_data = traj_transformed[:, j]

                # Plot trajectory line (without label)
                ax.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1.5, linestyle="--")

                # Plot points
                ax.scatter(x_data, y_data, c=[color], s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

                # Create legend handle with scatter marker (only for first subplot)
                if pair_idx == 0:
                    legend_handles.append(
                        ax.scatter([], [], c=color, s=60, alpha=0.7, edgecolors="black", linewidths=0.5, label=name)
                    )

                # Mark start and end points
                if len(x_data) > 0:
                    ax.scatter(x_data[0], y_data[0], c=[color], s=150, marker="o", edgecolors="black", linewidths=2, zorder=5)
                    ax.scatter(x_data[-1], y_data[-1], c=[color], s=150, marker="s", edgecolors="black", linewidths=2, zorder=5)

            ax.set_xlabel(f"PC{i+1} ({explained_var[i]:.3f})", fontsize=14)
            ax.set_ylabel(f"PC{j+1} ({explained_var[j]:.3f})", fontsize=14)
            ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            if pair_idx == 0:
                ax.legend(handles=legend_handles, loc="best", fontsize=16)

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"PCA Trajectories Comparison (4 components, cumulative variance: {explained_var.sum():.4f})",
            fontsize=18,
        )
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved 4-component PCA plot to: {outfile}")
    else:
        raise ValueError(f"n_components must be 2 or 4, got {n_components}")


def compute_pairwise_distances(final_embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise distances between final embeddings.

    Args:
        final_embeddings: List of final embedding arrays

    Returns:
        Tuple of (l2_distances, l1_distances, cosine_distances) matrices
    """
    n = len(final_embeddings)
    if n < 2:
        return np.array([]), np.array([]), np.array([])

    # Stack embeddings
    X = np.stack(final_embeddings, axis=0)  # [n_experiments, n_features]

    # Compute L2 distances
    diffs = X[:, None, :] - X[None, :, :]
    l2_distances = np.linalg.norm(diffs, axis=-1)

    # Compute L1 distances
    l1_distances = np.linalg.norm(diffs, ord=1, axis=-1)

    # Compute cosine distances
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos_sim = (Xn @ Xn.T).clip(-1.0, 1.0)
    cosine_distances = 1.0 - cos_sim

    return l2_distances, l1_distances, cosine_distances


def print_statistics_table(
    checkpoint_names: List[str],
    statistics: List[Dict[str, Any]],
):
    """Print a statistics table using tabulate.

    Args:
        checkpoint_names: List of experiment labels
        statistics: List of statistics dicts, each containing 'num_embeddings' and 'total_steps'
    """
    if len(checkpoint_names) == 0 or len(statistics) == 0:
        return

    # Prepare table data
    table_data = []
    for name, stats in zip(checkpoint_names, statistics):
        table_data.append(
            [
                name,
                stats.get("num_embeddings", 0),
                f"{stats.get('trajectory_length', 0.0):.4f}",
                f"{stats.get('num_pca_for99_var', 0.0)}",
            ]
        )

    headers = ["Experiment", "# Compr. Tok", "Traj. Len", "# PCA expl 99% var"]
    table = tabulate(table_data, headers=headers, tablefmt="grid", numalign="right", stralign="left")

    print("\n" + "=" * 80)
    print("Progressive Embeddings Statistics")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")


def print_pairwise_distances_table(
    checkpoint_names: List[str],
    l2_distances: np.ndarray,
    l1_distances: np.ndarray,
    cosine_distances: np.ndarray,
):
    """Print pairwise distances tables using tabulate.

    Args:
        checkpoint_names: List of experiment labels
        l2_distances: L2 distance matrix [n_experiments, n_experiments]
        l1_distances: L1 distance matrix [n_experiments, n_experiments]
        cosine_distances: Cosine distance matrix [n_experiments, n_experiments]
    """
    if len(checkpoint_names) < 2 or l2_distances.size == 0:
        return

    n = len(checkpoint_names)

    # L2 distances table
    print("\n" + "=" * 80)
    print("Pairwise L2 Distances Between Final Embeddings")
    print("=" * 80)
    l2_table_data = []
    for i in range(n):
        row = [checkpoint_names[i]]
        for j in range(n):
            if i == j:
                row.append("0.000")
            else:
                row.append(f"{l2_distances[i, j]:.4f}")
        l2_table_data.append(row)
    l2_headers = ["Experiment"] + checkpoint_names
    l2_table = tabulate(l2_table_data, headers=l2_headers, tablefmt="grid", numalign="right", stralign="left")
    print(l2_table)

    # L1 distances table
    print("\n" + "=" * 80)
    print("Pairwise L1 Distances Between Final Embeddings")
    print("=" * 80)
    l1_table_data = []
    for i in range(n):
        row = [checkpoint_names[i]]
        for j in range(n):
            if i == j:
                row.append("0.000")
            else:
                row.append(f"{l1_distances[i, j]:.4f}")
        l1_table_data.append(row)
    l1_headers = ["Experiment"] + checkpoint_names
    l1_table = tabulate(l1_table_data, headers=l1_headers, tablefmt="grid", numalign="right", stralign="left")
    print(l1_table)

    # Cosine distances table
    print("\n" + "=" * 80)
    print("Pairwise Cosine Distances Between Final Embeddings")
    print("=" * 80)
    cos_table_data = []
    for i in range(n):
        row = [checkpoint_names[i]]
        for j in range(n):
            if i == j:
                row.append("0.000")
            else:
                row.append(f"{cosine_distances[i, j]:.4f}")
        cos_table_data.append(row)
    cos_headers = ["Experiment"] + checkpoint_names
    cos_table = tabulate(cos_table_data, headers=cos_headers, tablefmt="grid", numalign="right", stralign="left")
    print(cos_table)
    print("=" * 80 + "\n")


def parse_names_mapping(names_str: Optional[str]) -> Tuple[Dict[str, str], Optional[List[str]]]:
    """Parse names mapping from string.

    Supports two formats:
    1. Path-based: 'path1:name1,path2:name2' (returns dict, None)
    2. Positional list: 'name1,name2,name3' (returns empty dict, list of names)

    Returns:
        Tuple of (path_mapping_dict, positional_names_list)
    """
    if names_str is None:
        return {}, None

    # Check if it contains colons (path-based mapping)
    if ":" in names_str:
        mapping = {}
        for pair in names_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                mapping[key.strip()] = value.strip()
        return mapping, None
    else:
        # Positional list format
        names = [name.strip() for name in names_str.split(",") if name.strip()]
        return {}, names if names else None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multiple progressive embeddings training trajectories on one PCA plot"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to progressive embeddings datasets (checkpoints)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the plot",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Sample ID to visualize (default: first available sample)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        choices=[2, 4],
        help="Number of PCA components (2 or 4)",
    )
    parser.add_argument(
        "--show_labels",
        action="store_true",
        help="Show stage labels on trajectory points",
    )
    parser.add_argument(
        "--names_mapping",
        type=str,
        default=None,
        help="Optional mapping of checkpoint paths to display names. "
        "Two formats supported: 1) Path-based: 'path1:name1,path2:name2' "
        "2) Positional list: 'name1,name2,name3' (corresponds to --checkpoints order)",
    )

    args = parser.parse_args()

    # Parse names mapping
    path_mapping, positional_names = parse_names_mapping(args.names_mapping)

    # Validate positional names length if provided
    if positional_names is not None and len(positional_names) != len(args.checkpoints):
        raise ValueError(
            f"Number of names in --names_mapping ({len(positional_names)}) "
            f"does not match number of checkpoints ({len(args.checkpoints)})"
        )

    # Extract trajectories from each checkpoint
    trajectories = []
    checkpoint_names = []
    labels_list = []
    statistics_list = []
    final_embeddings = []

    for idx, checkpoint_path in enumerate(args.checkpoints):
        traj, labels, stats, final_emb = extract_trajectory(checkpoint_path, sample_id=args.sample_id)
        trajectories.append(traj)
        labels_list.append(labels)
        statistics_list.append(stats)
        final_embeddings.append(final_emb)

        # Determine name for this checkpoint
        if positional_names is not None:
            # Use positional mapping
            checkpoint_names.append(positional_names[idx])
        elif checkpoint_path in path_mapping:
            # Use path-based mapping
            checkpoint_names.append(path_mapping[checkpoint_path])
        else:
            # Extract a short name from the path
            name = os.path.basename(os.path.dirname(checkpoint_path))
            if not name or name == ".":
                name = os.path.basename(checkpoint_path)
            checkpoint_names.append(name)

        print(f"Loaded trajectory from {checkpoint_path}: {traj.shape[0]} stages, {traj.shape[1]} features")

    if len(trajectories) == 0:
        raise ValueError("No valid trajectories loaded")

    # Print statistics table
    if len(statistics_list) > 0:
        print_statistics_table(checkpoint_names, statistics_list)

    # Compute and print pairwise distances
    # if len(final_embeddings) >= 2:
    #     l2_distances, l1_distances, cosine_distances = compute_pairwise_distances(final_embeddings)
    #     print_pairwise_distances_table(checkpoint_names, l2_distances, l1_distances, cosine_distances)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Plot trajectories
    plot_pca_trajectories(
        trajectories=trajectories,
        checkpoint_names=checkpoint_names,
        outfile=args.output,
        n_components=args.n_components,
        show_labels=args.show_labels,
        labels_list=labels_list if args.show_labels else None,
    )

    print(f"Visualization complete. Saved to: {args.output}")


if __name__ == "__main__":
    main()
