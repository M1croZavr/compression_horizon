import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
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


def extract_trajectory(
    dataset_path: str,
    sample_id: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract embedding trajectory from a dataset.

    Args:
        dataset_path: Path to the progressive embeddings dataset
        sample_id: Optional sample_id to filter. If None, uses first available sample.

    Returns:
        Tuple of (embeddings array [n_stages, n_features], labels list)
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

    # Extract embeddings in order
    embeddings = []
    labels = []
    for stage in stages:
        emb = flatten_embedding(stage)
        embeddings.append(emb)
        stage_seq_len = int(stage.get("stage_seq_len", -1))
        int(stage.get("stage_index", -1))
        labels.append(f"L{stage_seq_len}")

    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found for sample {sample_id} in {dataset_path}")

    X = np.stack(embeddings, axis=0)
    return X, labels


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

    # Create color map for checkpoints
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    if n_components == 2:
        # Single 2D plot
        plt.figure(figsize=(10, 8))
        for idx, (traj_transformed, name, color) in enumerate(zip(transformed_trajectories, checkpoint_names, colors)):
            x_data = traj_transformed[:, 0]
            y_data = traj_transformed[:, 1]

            # Plot trajectory line
            plt.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1.5, linestyle="--", label=name)

            # Plot points
            plt.scatter(x_data, y_data, c=[color], s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

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
                        plt.text(x_data[k], y_data[k], lab, fontsize=8, ha="left", va="bottom", color=color)
                        labeled_positions.append([x_data[k], y_data[k]])

            # Mark start and end points
            if len(x_data) > 0:
                plt.scatter(x_data[0], y_data[0], c=[color], s=150, marker="o", edgecolors="black", linewidths=2, zorder=5)
                plt.scatter(x_data[-1], y_data[-1], c=[color], s=150, marker="s", edgecolors="black", linewidths=2, zorder=5)

        plt.xlabel(f"PC1 ({explained_var[0]:.4f})", fontsize=14)
        plt.ylabel(f"PC2 ({explained_var[1]:.4f})", fontsize=14)
        plt.title(
            f"PCA Trajectories Comparison\nCumulative variance: {explained_var.sum():.4f}",
            fontsize=16,
        )
        plt.legend(loc="best", fontsize=10)
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

        for pair_idx, (i, j) in enumerate(pairs):
            ax = axes[pair_idx]

            for idx, (traj_transformed, name, color) in enumerate(zip(transformed_trajectories, checkpoint_names, colors)):
                x_data = traj_transformed[:, i]
                y_data = traj_transformed[:, j]

                # Plot trajectory line
                ax.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1.5, linestyle="--", label=name)

                # Plot points
                ax.scatter(x_data, y_data, c=[color], s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

                # Mark start and end points
                if len(x_data) > 0:
                    ax.scatter(x_data[0], y_data[0], c=[color], s=150, marker="o", edgecolors="black", linewidths=2, zorder=5)
                    ax.scatter(x_data[-1], y_data[-1], c=[color], s=150, marker="s", edgecolors="black", linewidths=2, zorder=5)

            ax.set_xlabel(f"PC{i+1} ({explained_var[i]:.3f})", fontsize=10)
            ax.set_ylabel(f"PC{j+1} ({explained_var[j]:.3f})", fontsize=10)
            ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            if pair_idx == 0:
                ax.legend(loc="best", fontsize=8)

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"PCA Trajectories Comparison (4 components, cumulative variance: {explained_var.sum():.4f})",
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved 4-component PCA plot to: {outfile}")
    else:
        raise ValueError(f"n_components must be 2 or 4, got {n_components}")


def parse_names_mapping(names_str: Optional[str]) -> Dict[str, str]:
    """Parse names mapping from string format 'key1:value1,key2:value2'."""
    if names_str is None:
        return {}
    mapping = {}
    for pair in names_str.split(","):
        if ":" in pair:
            key, value = pair.split(":", 1)
            mapping[key.strip()] = value.strip()
    return mapping


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
        help="Optional mapping of checkpoint paths to display names. Format: 'path1:name1,path2:name2'",
    )

    args = parser.parse_args()

    # Parse names mapping
    names_mapping = parse_names_mapping(args.names_mapping)

    # Extract trajectories from each checkpoint
    trajectories = []
    checkpoint_names = []
    labels_list = []

    for checkpoint_path in args.checkpoints:
        try:
            traj, labels = extract_trajectory(checkpoint_path, sample_id=args.sample_id)
            trajectories.append(traj)
            labels_list.append(labels)

            # Use mapping if available, otherwise use checkpoint path
            if checkpoint_path in names_mapping:
                checkpoint_names.append(names_mapping[checkpoint_path])
            else:
                # Extract a short name from the path
                name = os.path.basename(os.path.dirname(checkpoint_path))
                if not name or name == ".":
                    name = os.path.basename(checkpoint_path)
                checkpoint_names.append(name)

            print(f"Loaded trajectory from {checkpoint_path}: {traj.shape[0]} stages, {traj.shape[1]} features")
        except Exception as e:
            print(f"Warning: Failed to load {checkpoint_path}: {e}")
            continue

    if len(trajectories) == 0:
        raise ValueError("No valid trajectories loaded")

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
