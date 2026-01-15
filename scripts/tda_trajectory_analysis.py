"""
Topological Data Analysis (TDA) of Trajectory Space for Progressive Training Artifacts.

This script analyzes the topological structure of the space of compression trajectories
using persistent homology. Each trajectory is a curve in high-dimensional embedding space
(typically 2048D), representing the evolution of compression tokens across progressive stages.

Questions addressed:
- Do trajectories for similar semantics live in the same homology class?
- Are there forbidden topological structures?
- What are the fundamental constraints on how compression works?
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from persim import plot_diagrams
from ripser import ripser
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


def load_progressive_dataset(dataset_path: str) -> Dataset:
    """Load progressive training artifacts dataset."""
    return Dataset.load_from_disk(dataset_path)


def extract_trajectories(ds: Dataset) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Extract trajectories from progressive artifacts.

    Each trajectory is a sequence of embeddings across progressive stages for a given sample.

    Returns:
        trajectories: Array of shape (n_samples, max_stages, embedding_dim)
        sample_to_stages: Mapping from sample_id to list of stage indices
    """
    # Group by sample_id
    by_sample: Dict[int, List[Dict]] = {}
    for i in tqdm(range(len(ds)), desc="Grouping by sample"):
        row = ds[i]
        sample_id = int(row.get("sample_id", -1))
        if sample_id not in by_sample:
            by_sample[sample_id] = []
        by_sample[sample_id].append(row)

    # Sort by stage_index for each sample
    for sample_id in by_sample:
        by_sample[sample_id].sort(key=lambda x: int(x.get("stage_index", 0)))

    # Extract trajectories
    trajectories = []
    sample_to_stages = {}
    embedding_dim = None

    for sample_id, rows in sorted(by_sample.items()):
        stages = []
        stage_indices = []
        for row in rows:
            emb = torch.tensor(row["embedding"], dtype=torch.float32)
            emb_flat = emb.reshape(-1).detach().cpu().numpy()
            if embedding_dim is None:
                embedding_dim = len(emb_flat)
            stages.append(emb_flat)
            stage_indices.append(int(row.get("stage_index", 0)))

        if len(stages) > 0:
            # Pad trajectories to same length if needed
            trajectories.append(np.array(stages))
            sample_to_stages[sample_id] = stage_indices

    # Find max length
    max_len = max(len(traj) for traj in trajectories) if trajectories else 0

    # Pad all trajectories to same length
    padded_trajectories = []
    for traj in trajectories:
        if len(traj) < max_len:
            # Repeat last point to pad
            padding = np.tile(traj[-1:], (max_len - len(traj), 1))
            padded = np.vstack([traj, padding])
        else:
            padded = traj
        padded_trajectories.append(padded)

    return np.array(padded_trajectories), sample_to_stages


def compute_trajectory_distance_matrix(trajectories: np.ndarray, metric: str = "frechet") -> np.ndarray:
    """
    Compute pairwise distance matrix between trajectories.

    Args:
        trajectories: Array of shape (n_samples, n_stages, embedding_dim)
        metric: Distance metric to use. Options:
            - "frechet": Discrete Fréchet distance (approximate)
            - "dtw": Dynamic Time Warping distance
            - "hausdorff": Hausdorff distance
            - "l2_mean": Mean L2 distance between corresponding points
            - "l2_max": Maximum L2 distance between corresponding points

    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    n_samples = len(trajectories)
    dist_matrix = np.zeros((n_samples, n_samples))

    if metric == "frechet":
        # Approximate Fréchet distance using dynamic programming
        for i in tqdm(range(n_samples), desc="Computing Fréchet distances"):
            for j in range(i + 1, n_samples):
                dist = _discrete_frechet_distance(trajectories[i], trajectories[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    elif metric == "dtw":
        # Dynamic Time Warping
        for i in tqdm(range(n_samples), desc="Computing DTW distances"):
            for j in range(i + 1, n_samples):
                dist = _dtw_distance(trajectories[i], trajectories[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    elif metric == "hausdorff":
        # Hausdorff distance
        for i in tqdm(range(n_samples), desc="Computing Hausdorff distances"):
            for j in range(i + 1, n_samples):
                dist = _hausdorff_distance(trajectories[i], trajectories[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    elif metric == "l2_mean":
        # Mean L2 distance
        for i in tqdm(range(n_samples), desc="Computing mean L2 distances"):
            for j in range(i + 1, n_samples):
                dists = np.linalg.norm(trajectories[i] - trajectories[j], axis=1)
                dist = np.mean(dists)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    elif metric == "l2_max":
        # Maximum L2 distance
        for i in tqdm(range(n_samples), desc="Computing max L2 distances"):
            for j in range(i + 1, n_samples):
                dists = np.linalg.norm(trajectories[i] - trajectories[j], axis=1)
                dist = np.max(dists)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist_matrix


def _discrete_frechet_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute discrete Fréchet distance between two trajectories."""
    n, m = len(traj1), len(traj2)
    # Dynamic programming table
    dp = np.full((n, m), np.inf)

    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    dp[0, 0] = dist(traj1[0], traj2[0])

    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], dist(traj1[i], traj2[0]))

    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], dist(traj1[0], traj2[j]))

    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                dist(traj1[i], traj2[j]),
            )

    return dp[n - 1, m - 1]


def _dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two trajectories."""
    n, m = len(traj1), len(traj2)
    # Dynamic programming table
    dp = np.full((n, m), np.inf)

    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    dp[0, 0] = dist(traj1[0], traj2[0])

    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + dist(traj1[i], traj2[0])

    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + dist(traj1[0], traj2[j])

    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = dist(traj1[i], traj2[j]) + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return dp[n - 1, m - 1]


def _hausdorff_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Hausdorff distance between two trajectories."""
    # For each point in traj1, find minimum distance to traj2
    dists_1_to_2 = []
    for p1 in traj1:
        min_dist = min(np.linalg.norm(p1 - p2) for p2 in traj2)
        dists_1_to_2.append(min_dist)

    # For each point in traj2, find minimum distance to traj1
    dists_2_to_1 = []
    for p2 in traj2:
        min_dist = min(np.linalg.norm(p2 - p1) for p1 in traj1)
        dists_2_to_1.append(min_dist)

    # Hausdorff distance is the maximum of the two suprema
    return max(max(dists_1_to_2), max(dists_2_to_1))


def compute_persistent_homology(distance_matrix: np.ndarray, max_dim: int = 2) -> Dict:
    """
    Compute persistent homology from distance matrix.

    Args:
        distance_matrix: Pairwise distance matrix
        max_dim: Maximum homology dimension to compute (0=connected components, 1=loops, 2=voids)

    Returns:
        Dictionary with persistence diagrams for each dimension
    """
    print(f"Computing persistent homology (max_dim={max_dim})...")
    result = ripser(distance_matrix, maxdim=max_dim, metric="precomputed")
    diagrams = result["dgms"]

    return {
        "diagrams": diagrams,
        "cocycles": result.get("cocycles", []),
        "num_points": len(distance_matrix),
    }


def plot_persistence_diagram(
    diagrams: List[np.ndarray],
    output_path: str,
    title: str = "Persistence Diagram",
):
    """Plot persistence diagrams for all dimensions."""
    fig, axes = plt.subplots(1, len(diagrams), figsize=(6 * len(diagrams), 6))
    if len(diagrams) == 1:
        axes = [axes]

    for dim, (diagram, ax) in enumerate(zip(diagrams, axes)):
        if len(diagram) > 0:
            plot_diagrams([diagram], ax=ax, show=False)
            ax.set_title(f"H{dim} - {title}")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
        else:
            ax.set_title(f"H{dim} - {title} (no features)")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_persistence_barcode(diagrams: List[np.ndarray], output_path: str, title: str = "Persistence Barcode"):
    """Plot persistence barcodes for all dimensions."""
    n_dims = len(diagrams)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims))
    if n_dims == 1:
        axes = [axes]

    for dim, (diagram, ax) in enumerate(zip(diagrams, axes)):
        if len(diagram) > 0:
            # Sort by persistence (death - birth)
            persistence = diagram[:, 1] - diagram[:, 0]
            sorted_indices = np.argsort(persistence)[::-1]
            sorted_diagram = diagram[sorted_indices]

            # Plot barcodes
            for i, (birth, death) in enumerate(sorted_diagram):
                ax.plot([birth, death], [i, i], "b-", linewidth=2)
                ax.scatter([birth, death], [i, i], c="b", s=30, zorder=3)

            ax.set_xlabel("Filtration Value")
            ax.set_ylabel("Feature Index")
            ax.set_title(f"H{dim} Barcode - {title} ({len(diagram)} features)")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"H{dim} Barcode - {title} (no features)")
            ax.set_xlabel("Filtration Value")
            ax.set_ylabel("Feature Index")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_topological_features(diagrams: List[np.ndarray]) -> Dict:
    """Analyze and summarize topological features."""
    analysis = {}

    for dim, diagram in enumerate(diagrams):
        if len(diagram) == 0:
            analysis[f"H{dim}"] = {
                "num_features": 0,
                "max_persistence": 0.0,
                "mean_persistence": 0.0,
                "total_persistence": 0.0,
            }
            continue

        # Filter out infinite features (death = inf)
        finite_diagram = diagram[diagram[:, 1] != np.inf]
        infinite_count = len(diagram) - len(finite_diagram)

        if len(finite_diagram) > 0:
            persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
            analysis[f"H{dim}"] = {
                "num_features": len(diagram),
                "num_finite": len(finite_diagram),
                "num_infinite": infinite_count,
                "max_persistence": float(np.max(persistence)),
                "mean_persistence": float(np.mean(persistence)),
                "total_persistence": float(np.sum(persistence)),
                "std_persistence": float(np.std(persistence)),
            }
        else:
            analysis[f"H{dim}"] = {
                "num_features": len(diagram),
                "num_finite": 0,
                "num_infinite": infinite_count,
                "max_persistence": 0.0,
                "mean_persistence": 0.0,
                "total_persistence": 0.0,
                "std_persistence": 0.0,
            }

    return analysis


def plot_trajectory_space_embedding(
    distance_matrix: np.ndarray,
    output_path: str,
    method: str = "tsne",
    title: str = "Trajectory Space Embedding",
):
    """
    Visualize trajectory space using dimensionality reduction.

    Args:
        distance_matrix: Pairwise distance matrix
        output_path: Path to save figure
        method: "tsne" or "pca"
        title: Plot title
    """
    print(f"Computing {method.upper()} embedding...")

    if method == "tsne":
        # Use distance matrix for t-SNE
        # Note: init="random" is required when using metric="precomputed"
        embedding = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=42,
            perplexity=min(30, len(distance_matrix) - 1),
        ).fit_transform(distance_matrix)
    elif method == "pca":
        # Convert distance matrix to coordinates using MDS-like approach
        # Use PCA on a kernel matrix derived from distances
        n = len(distance_matrix)
        # Center the distance matrix
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (distance_matrix**2) @ J
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(B)
        # Sort by eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        # Take top 2 components
        embedding = eigenvecs[:, :2] * np.sqrt(eigenvals[:2])
    else:
        raise ValueError(f"Unknown method: {method}")

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=50)
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Topological Data Analysis of trajectory space from progressive training")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to progressive_prefixes dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save TDA results and visualizations",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="frechet",
        choices=["frechet", "dtw", "hausdorff", "l2_mean", "l2_max"],
        help="Distance metric for comparing trajectories",
    )
    parser.add_argument(
        "--max_dim",
        type=int,
        default=2,
        help="Maximum homology dimension to compute (0=components, 1=loops, 2=voids)",
    )
    parser.add_argument(
        "--embedding_method",
        type=str,
        default="tsne",
        choices=["tsne", "pca"],
        help="Method for visualizing trajectory space",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.dataset_path), "tda_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Topological Data Analysis of Trajectory Space")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading progressive artifacts...")
    ds = load_progressive_dataset(args.dataset_path)
    print(f"   Loaded {len(ds)} records")

    # Extract trajectories
    print("\n2. Extracting trajectories...")
    trajectories, sample_to_stages = extract_trajectories(ds)
    print(f"   Extracted {len(trajectories)} trajectories")
    print(f"   Trajectory shape: {trajectories.shape}")
    print(f"   Embedding dimension: {trajectories.shape[2]}")

    # Compute distance matrix
    print(f"\n3. Computing distance matrix (metric: {args.distance_metric})...")
    distance_matrix = compute_trajectory_distance_matrix(trajectories, metric=args.distance_metric)
    print(f"   Distance matrix shape: {distance_matrix.shape}")
    print(f"   Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
    print(f"   Mean distance: {distance_matrix.mean():.4f}")

    # Save distance matrix
    np.save(
        os.path.join(args.output_dir, f"distance_matrix_{args.distance_metric}.npy"),
        distance_matrix,
    )

    # Compute persistent homology
    print(f"\n4. Computing persistent homology (max_dim={args.max_dim})...")
    ph_result = compute_persistent_homology(distance_matrix, max_dim=args.max_dim)
    diagrams = ph_result["diagrams"]

    # Analyze topological features
    print("\n5. Analyzing topological features...")
    analysis = analyze_topological_features(diagrams)
    for dim, stats in analysis.items():
        print(f"\n   {dim}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.4f}")
            else:
                print(f"     {key}: {value}")

    # Save analysis
    import json

    with open(os.path.join(args.output_dir, "topological_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Visualize persistence diagrams
    print("\n6. Visualizing persistence diagrams...")
    plot_persistence_diagram(
        diagrams,
        os.path.join(args.output_dir, "persistence_diagram.png"),
        title=f"Trajectory Space ({args.distance_metric})",
    )

    # Visualize persistence barcodes
    print("7. Visualizing persistence barcodes...")
    plot_persistence_barcode(
        diagrams,
        os.path.join(args.output_dir, "persistence_barcode.png"),
        title=f"Trajectory Space ({args.distance_metric})",
    )

    # Visualize trajectory space embedding
    print(f"\n8. Visualizing trajectory space ({args.embedding_method})...")
    plot_trajectory_space_embedding(
        distance_matrix,
        os.path.join(args.output_dir, f"trajectory_space_{args.embedding_method}.png"),
        method=args.embedding_method,
        title=f"Trajectory Space Embedding ({args.distance_metric})",
    )

    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
