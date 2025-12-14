import argparse
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_progressive_dataset(dataset_path: str) -> Dataset:
    return Dataset.load_from_disk(dataset_path)


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("sample_id", -1))
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(r)
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: int(x.get("stage_index", 0)))
    return by_sid


def flatten_embedding(row: Dict[str, Any]) -> np.ndarray:
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    return emb.reshape(-1).detach().cpu().numpy()


def compute_pairwise_similarities(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diffs = X[:, None, :] - X[None, :, :]
    l2 = np.linalg.norm(diffs, axis=-1)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn @ Xn.T).clip(-1.0, 1.0)
    cos_dist = 1.0 - cos
    return l2, cos_dist


def plot_heatmap(matrix: np.ndarray, labels: List[str], title: str, outfile: str):
    plt.figure(figsize=(0.7 * max(4, len(labels)), 0.7 * max(4, len(labels))))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=False,
        square=True,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_pca(X: np.ndarray, labels: List[str], outfile: str):
    if X.shape[0] < 2 or X.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    print(
        f"PCA explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}, Cumulative={explained_var.sum():.4f}"
    )
    # Check dispersion (std dev) and swap if needed to ensure x-axis has more dispersion
    pc1_disp = np.std(xy[:, 0])
    pc2_disp = np.std(xy[:, 1])
    if pc2_disp > pc1_disp:
        # Swap PC1 and PC2
        xy = xy[:, [1, 0]]
        explained_var = explained_var[[1, 0]]
        xlabel = "PC2"
        ylabel = "PC1"
    else:
        xlabel = "PC1"
        ylabel = "PC2"
    # Calculate appropriate figure size for 1:1 aspect ratio
    # x_range = np.max(xy[:, 0]) - np.min(xy[:, 0])
    # y_range = np.max(xy[:, 1]) - np.min(xy[:, 1])

    # plt.figure(figsize=(x_range, y_range))
    plt.figure(figsize=(8.8, 7))
    labeled_positions = []
    for i, lab in enumerate(labels):
        plt.scatter(xy[i, 0], xy[i, 1], s=60)
        # Check if there's already a labeled point within distance < 0.5
        should_label = True
        for labeled_pos in labeled_positions:
            dist = np.linalg.norm(xy[i] - labeled_pos)
            if dist < 0.5:
                should_label = False
                break
        if should_label:
            plt.text(xy[i, 0], xy[i, 1], lab, fontsize=18, ha="left", va="bottom")
            labeled_positions.append(xy[i])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.axis("equal")
    plt.title(
        f"PCA of progressive embeddings (flattened)\n{xlabel}: {explained_var[0]:.4f}, {ylabel}: {explained_var[1]:.4f}, Cumulative: {explained_var.sum():.4f}",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print("plot_pca", outfile)
    plt.close()


def plot_pca_4_components(X: np.ndarray, labels: List[str], outfile: str):
    """Plot all pairs of the 4 main PCA components in subplots.

    Args:
        X: Input data array [n_samples, n_features]
        labels: List of labels for each sample
        outfile: Output file path
    """
    if X.shape[0] < 2 or X.shape[1] < 2:
        return

    n_components = min(4, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        return

    pca = PCA(n_components=n_components, random_state=42)
    pca_data = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_

    # Create all pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    pairs = [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]
    n_pairs = len(pairs)

    # Arrange subplots in a grid: 2 rows x 3 columns for 6 pairs
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        x_data = pca_data[:, i]
        y_data = pca_data[:, j]

        # Plot scatter points
        ax.scatter(x_data, y_data, s=60)

        # Add labels with collision detection
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
                ax.text(x_data[k], y_data[k], lab, fontsize=10, ha="left", va="bottom")
                labeled_positions.append([x_data[k], y_data[k]])

        ax.set_xlabel(f"PC{i+1} ({explained_var[i]:.3f})", fontsize=10)
        ax.set_ylabel(f"PC{j+1} ({explained_var[j]:.3f})", fontsize=10)
        ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"PCA: All Component Pairs (4 components, cumulative variance: {explained_var.sum():.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"plot_pca_4_components: {outfile}")


def plot_cumulative_explained_variance(X: np.ndarray, title: str, outfile: str, max_components: Optional[int] = None):
    """Plot cumulative explained variance as a function of number of PCA components.

    Args:
        X: Input data array [n_samples, n_features]
        title: Plot title
        outfile: Output file path
        max_components: Maximum number of components to compute (default: min(n_samples, n_features))
    """
    if X.shape[0] < 2 or X.shape[1] < 2:
        return

    n_samples, n_features = X.shape
    max_comp = max_components if max_components is not None else min(n_samples - 1, n_features)
    max_comp = min(max_comp, n_samples - 1, n_features)

    if max_comp < 1:
        return

    pca = PCA(n_components=max_comp, random_state=42)
    pca.fit(X)
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    n_components = np.arange(1, len(cumulative_var) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(n_components, cumulative_var, marker="o", linewidth=2, markersize=4)
    plt.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95% variance")
    plt.axhline(y=0.99, color="g", linestyle="--", alpha=0.7, label="99% variance")
    plt.xlabel("Number of PCA Components", fontsize=14)
    plt.ylabel("Cumulative Explained Variance", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(left=0)
    # plt.ylim(bottom=0, top=1.05)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print("plot_cumulative_explained_variance", outfile)
    plt.close()

    # Print summary statistics
    n_95 = np.argmax(cumulative_var >= 0.95) + 1 if np.any(cumulative_var >= 0.95) else len(cumulative_var)
    n_99 = np.argmax(cumulative_var >= 0.99) + 1 if np.any(cumulative_var >= 0.99) else len(cumulative_var)
    print(f"Cumulative explained variance: {n_95} components explain 95%, {n_99} components explain 99%")


def compute_pca_components_for_sample(
    stages: List[Dict[str, Any]],
    target_seq_lengths: List[int] = [4, 16, 32, 48, 64, 96, 128],
) -> Dict[int, Optional[int]]:
    """Compute number of PCA components explaining 99% variance for each sequence length for a sample.

    Args:
        stages: List of stage records for a sample
        target_seq_lengths: List of sequence lengths to analyze

    Returns:
        Dictionary mapping sequence length to number of components (or None if not computable)
    """
    # Group stages by sequence length
    stages_by_seq_len: Dict[int, List[Dict[str, Any]]] = {}
    for tsl in target_seq_lengths:
        stages_by_seq_len[tsl] = []

    max_seq_len = 0
    for stage in stages:
        seq_len = int(stage.get("stage_seq_len", -1))
        max_seq_len = max(max_seq_len, seq_len)
        for tsl in target_seq_lengths:
            if seq_len <= tsl:
                stages_by_seq_len[tsl].append(stage)

    for tsl in list(stages_by_seq_len.keys()):
        if tsl > max_seq_len:
            del stages_by_seq_len[tsl]

    results: Dict[int, Optional[int]] = {}

    for seq_len in sorted(target_seq_lengths):
        if seq_len not in stages_by_seq_len:
            results[seq_len] = None
            continue

        stage_list = stages_by_seq_len[seq_len]
        if len(stage_list) < 1:
            results[seq_len] = None
            continue

        # Extract and flatten embeddings for this sequence length
        embeddings_list = [flatten_embedding(s) for s in stage_list]
        if len(embeddings_list) == 0:
            results[seq_len] = None
            continue

        # Stack embeddings: [n_stages, n_features]
        X = np.stack(embeddings_list, axis=0)

        # Need at least 2 samples for PCA
        if X.shape[0] < 2:
            results[seq_len] = None
            continue

        # Compute PCA with maximum possible components
        n_samples, n_features = X.shape
        max_comp = min(n_samples - 1, n_features)
        if max_comp < 1:
            results[seq_len] = None
            continue

        pca = PCA(n_components=max_comp, random_state=42)
        pca.fit(X)
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)

        # Find number of components explaining 99% variance
        n_99 = np.argmax(cumulative_var >= 0.99) + 1 if np.any(cumulative_var >= 0.99) else len(cumulative_var)
        results[seq_len] = n_99

    return results


def plot_pca_components_vs_sequence_length(
    stages: List[Dict[str, Any]],
    sample_id: int,
    outfile: str,
    target_seq_lengths: List[int] = [4, 16, 32, 48, 64, 96, 128],
):
    """Plot number of PCA components explaining 99% variance vs sequence length for a sample.

    Args:
        stages: List of stage records for a sample
        sample_id: Sample ID for title
        outfile: Output file path
        target_seq_lengths: List of sequence lengths to analyze
    """
    results = compute_pca_components_for_sample(stages, target_seq_lengths)

    seq_lengths: List[int] = []
    n_components_99: List[int] = []

    for seq_len in sorted(target_seq_lengths):
        if results.get(seq_len) is not None:
            seq_lengths.append(seq_len)
            n_components_99.append(results[seq_len])

    if len(seq_lengths) == 0:
        print(f"No valid data points for sample {sample_id}")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, n_components_99, marker="o", linewidth=2, markersize=8, label="99% variance")
    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Number of PCA Components", fontsize=14)
    plt.title(f"Sample {sample_id}: PCA Components Explaining 99% Variance vs Sequence Length", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"plot_pca_components_vs_sequence_length: {outfile}")
    plt.close()


def plot_pca_components_vs_sequence_length_aggregate(
    by_sid: Dict[int, List[Dict[str, Any]]],
    outfile: str,
    target_seq_lengths: List[int] = [4, 16, 32, 48, 64, 96, 128],
):
    """Plot number of PCA components explaining 99% variance vs sequence length aggregated across all samples.

    Shows quantiles (10th-90th percentile, 25th-75th IQR) and mean across samples.

    Args:
        by_sid: Dictionary mapping sample_id to list of stage records
        outfile: Output file path
        target_seq_lengths: List of sequence lengths to analyze
    """
    # Collect PCA component counts for each sequence length across all samples
    components_by_seq_len: Dict[int, List[int]] = {}
    for seq_len in target_seq_lengths:
        components_by_seq_len[seq_len] = []

    for sid, stages in tqdm(by_sid.items(), desc="Computing PCA components per sample"):
        results = compute_pca_components_for_sample(stages, target_seq_lengths)
        for seq_len, n_components in results.items():
            if n_components is not None:
                components_by_seq_len[seq_len].append(n_components)

    # Filter out sequence lengths with no data
    seq_lengths: List[int] = []
    all_components_per_seq_len: List[List[int]] = []

    for seq_len in sorted(target_seq_lengths):
        if seq_len in components_by_seq_len and len(components_by_seq_len[seq_len]) > 0:
            seq_lengths.append(seq_len)
            all_components_per_seq_len.append(components_by_seq_len[seq_len])

    if len(seq_lengths) == 0:
        print("No valid data points for aggregate PCA components vs sequence length")
        return

    # Compute statistics for plotting
    mean_components = [np.mean(comps) for comps in all_components_per_seq_len]
    q25_components = [np.percentile(comps, 25) for comps in all_components_per_seq_len]
    q75_components = [np.percentile(comps, 75) for comps in all_components_per_seq_len]
    q10_components = [np.percentile(comps, 10) for comps in all_components_per_seq_len]
    q90_components = [np.percentile(comps, 90) for comps in all_components_per_seq_len]

    plt.figure(figsize=(10, 7))
    # Plot shaded regions showing distribution
    # Outer region: 10th-90th percentile
    plt.fill_between(
        seq_lengths,
        q10_components,
        q90_components,
        alpha=0.15,
        color="blue",
        label="10th-90th percentile",
    )
    # Inner region: 25th-75th percentile (IQR)
    plt.fill_between(
        seq_lengths,
        q25_components,
        q75_components,
        alpha=0.3,
        color="blue",
        label="Interquartile range (25th-75th)",
    )
    # Plot mean line
    plt.plot(
        seq_lengths,
        mean_components,
        marker="o",
        linewidth=2.5,
        markersize=6,
        color="darkblue",
        label="Mean",
    )
    # Plot individual points with transparency to show density
    for seq_len, comps in zip(seq_lengths, all_components_per_seq_len):
        plt.scatter([seq_len] * len(comps), comps, alpha=0.2, s=20, color="blue", zorder=0)

    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Number of PCA Components", fontsize=14)
    plt.title("PCA Components Explaining 99% Variance vs Sequence Length (All Samples)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"plot_pca_components_vs_sequence_length_aggregate: {outfile}")
    plt.close()


def plot_pca_reconstruction_accuracy(
    rows: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    title: str,
    outfile: str,
    max_components: Optional[int] = None,
):
    """Plot reconstruction accuracy (with teacher forcing) vs number of PCA components.

    For each number of PCA components, reconstructs compression embeddings from PCA and computes
    token prediction accuracy using model forward pass. Only computes accuracy once on the max seq length per unique sample_id.
    PCA is applied separately for each sample.

    Args:
        rows: List of dataset rows containing 'embedding', 'text', and 'sample_id' fields
        model: Language model for forward pass
        tokenizer: Tokenizer for text processing
        device: Device to run computations on
        title: Plot title
        outfile: Output file path
        max_components: Maximum number of components to compute
    """
    if len(rows) == 0:
        return

    # Group rows by sample_id
    rows_by_sample_id: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        sample_id = row.get("sample_id", None)
        if sample_id is None:
            continue
        sample_id = int(sample_id)
        if sample_id not in rows_by_sample_id:
            rows_by_sample_id[sample_id] = []
        rows_by_sample_id[sample_id].append(row)

    # Prepare data structures for each sample
    # For each sample: collect all embeddings for PCA, and identify longest sequence row for accuracy
    sample_data: List[Dict[str, Any]] = []
    for sample_id, sample_rows in rows_by_sample_id.items():
        # Find max sequence length for this sample and collect all embeddings
        max_seq_len = -1
        longest_row = None
        longest_row_idx = -1
        all_embeddings_for_sample = []
        all_rows_ordered = []

        for idx, row in enumerate(sample_rows):
            seq_len = int(row.get("stage_seq_len", -1))
            if seq_len > max_seq_len:
                max_seq_len = seq_len
                longest_row = row
                longest_row_idx = idx

            # Collect all embeddings for this sample (for PCA)
            emb = torch.tensor(row["embedding"], dtype=torch.float32)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            flattened_embedding = emb.reshape(-1).detach().cpu().numpy()
            all_embeddings_for_sample.append(flattened_embedding)
            all_rows_ordered.append(row)

        if longest_row is None or len(all_embeddings_for_sample) < 2:
            # Need at least 2 embeddings for PCA
            continue

        text = longest_row.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            continue

        emb_longest = torch.tensor(longest_row["embedding"], dtype=torch.float32)
        if emb_longest.ndim == 1:
            emb_longest = emb_longest.unsqueeze(0)
        original_shape = emb_longest.shape  # [num_compression_tokens, hidden_dim]

        # Get the embedding corresponding to the longest sequence
        longest_embedding = all_embeddings_for_sample[longest_row_idx]

        sample_data.append(
            {
                "sample_id": sample_id,
                "text": text,
                "original_shape": original_shape,
                "all_embeddings": np.stack(all_embeddings_for_sample, axis=0),  # All embeddings for PCA
                "longest_embedding": longest_embedding,  # Embedding from longest sequence
            }
        )

    if len(sample_data) == 0:
        return

    print(f"Processing {len(sample_data)} samples, each with PCA learned on all its embeddings")

    model.eval()
    input_embeddings_layer = model.get_input_embeddings()

    n_components_list = []
    all_accuracies_per_component = []  # Store all accuracies for each component count
    all_first_error_indices_per_component = []  # Store first error indices for each component count

    # Determine max components across all samples
    max_comp_global = 0
    for sample_info in sample_data:
        all_emb = sample_info["all_embeddings"]
        n_samples_for_pca, n_features = all_emb.shape
        max_comp_for_sample = min(n_samples_for_pca - 1, n_features)
        max_comp_global = max(max_comp_global, max_comp_for_sample)

    if max_components is not None:
        max_comp_global = min(max_comp_global, max_components)

    if max_comp_global < 1:
        return

    for n_comp in tqdm(range(1, max_comp_global + 1, 2), desc="pca_reconstruction_accuracy"):
        accuracies_per_sample = []
        first_error_indices_per_sample = []

        for sample_info in sample_data:
            all_emb = sample_info["all_embeddings"]
            n_samples_for_pca, n_features = all_emb.shape
            max_comp_for_sample = min(n_samples_for_pca - 1, n_features)

            if n_comp > max_comp_for_sample:
                continue

            # Fit PCA with n_comp components on all embeddings for this sample
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(all_emb)

            # Transform the longest sequence embedding
            longest_emb = sample_info["longest_embedding"].reshape(1, -1)
            longest_transformed = pca.transform(longest_emb)
            longest_reconstructed = pca.inverse_transform(longest_transformed)

            with torch.no_grad():
                # Reconstruct compression embedding and reshape to original shape
                comp_emb_flat = torch.tensor(longest_reconstructed[0], dtype=torch.float32, device=device)
                compression_embedding = comp_emb_flat.reshape(sample_info["original_shape"]).to(device)

                text = sample_info["text"]

                # Tokenize text
                enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Get input text embeddings
                input_text_embeds = input_embeddings_layer(input_ids)

                # Concatenate compression embedding with input text embeddings
                num_compression_tokens = compression_embedding.shape[0]
                input_embeds = torch.cat([compression_embedding.unsqueeze(0), input_text_embeds], dim=1)

                # Extend attention mask to include compression tokens
                comp_attention = torch.ones(
                    (attention_mask.shape[0], num_compression_tokens), device=device, dtype=attention_mask.dtype
                )
                extended_attention_mask = torch.cat([comp_attention, attention_mask], dim=1)

                # Forward pass
                compression_outputs = model(inputs_embeds=input_embeds, attention_mask=extended_attention_mask)

                # Compute accuracy: compare predicted tokens with input_ids
                # logits[:, num_compression_tokens - 1 : -1] corresponds to predictions for input tokens
                pred_logits = compression_outputs.logits[:, num_compression_tokens - 1 : -1]
                pred_tokens = pred_logits.argmax(dim=-1)

                # Compare with input_ids (full sequence, as logits predict next token)
                # The logits at position num_compression_tokens - 1 predict input_ids[0], etc.
                convergence_numerator = (pred_tokens == input_ids).sum(dim=-1)
                convergence_per_sample = convergence_numerator.float() / attention_mask.sum(dim=-1).float()

                if attention_mask.sum().item() > 0:
                    accuracy = convergence_per_sample.item()
                    accuracies_per_sample.append(accuracy)

                    # Find first error index: first position where prediction is wrong
                    # Compare pred_tokens[0, i] with input_ids[0, i] for valid positions
                    seq_len = int(attention_mask.sum().item())
                    first_error_idx = seq_len  # Default: no error (error at sequence length)
                    for i in range(seq_len):
                        if pred_tokens[0, i].item() != input_ids[0, i].item():
                            first_error_idx = i
                            break
                    first_error_indices_per_sample.append(first_error_idx)

        if len(accuracies_per_sample) > 0:
            n_components_list.append(n_comp)
            all_accuracies_per_component.append(accuracies_per_sample)
            all_first_error_indices_per_component.append(first_error_indices_per_sample)

    breakpoint()

    if len(n_components_list) == 0:
        print("len(n_components_list) == 0")
        raise ValueError("len(n_components_list) == 0")
        # return

    # Compute statistics for plotting
    mean_accuracies = [np.mean(accs) for accs in all_accuracies_per_component]
    # std_accuracies = [np.std(accs) for accs in all_accuracies_per_component]
    q25_accuracies = [np.percentile(accs, 25) for accs in all_accuracies_per_component]
    q75_accuracies = [np.percentile(accs, 75) for accs in all_accuracies_per_component]
    q10_accuracies = [np.percentile(accs, 10) for accs in all_accuracies_per_component]
    q90_accuracies = [np.percentile(accs, 90) for accs in all_accuracies_per_component]

    plt.figure(figsize=(10, 7))
    # Plot shaded regions showing distribution
    # Outer region: 10th-90th percentile
    plt.fill_between(
        n_components_list,
        q10_accuracies,
        q90_accuracies,
        alpha=0.15,
        color="blue",
        label="10th-90th percentile",
    )
    # Inner region: 25th-75th percentile (IQR)
    plt.fill_between(
        n_components_list,
        q25_accuracies,
        q75_accuracies,
        alpha=0.3,
        color="blue",
        label="Interquartile range (25th-75th)",
    )
    # Plot mean line
    plt.plot(
        n_components_list, mean_accuracies, marker="o", linewidth=2.5, markersize=6, color="darkblue", label="Mean accuracy"
    )
    # Plot individual points with transparency to show density
    for n_comp, accs in zip(n_components_list, all_accuracies_per_component):
        plt.scatter([n_comp] * len(accs), accs, alpha=0.2, s=20, color="blue", zorder=0)

    plt.xlabel("Number of PCA Components", fontsize=14)
    plt.ylabel("Reconstruction Accuracy (Token Prediction)", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1.05)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"plot_pca_reconstruction_accuracy: {outfile}")
    plt.close()

    # Plot first error index vs number of PCA components
    if len(all_first_error_indices_per_component) > 0:
        # Compute statistics for first error indices
        mean_first_error_indices = [np.mean(indices) for indices in all_first_error_indices_per_component]
        q25_first_error_indices = [np.percentile(indices, 25) for indices in all_first_error_indices_per_component]
        q75_first_error_indices = [np.percentile(indices, 75) for indices in all_first_error_indices_per_component]
        q10_first_error_indices = [np.percentile(indices, 10) for indices in all_first_error_indices_per_component]
        q90_first_error_indices = [np.percentile(indices, 90) for indices in all_first_error_indices_per_component]

        plt.figure(figsize=(10, 7))
        # Plot shaded regions showing distribution
        # Outer region: 10th-90th percentile
        plt.fill_between(
            n_components_list,
            q10_first_error_indices,
            q90_first_error_indices,
            alpha=0.15,
            color="red",
            label="10th-90th percentile",
        )
        # Inner region: 25th-75th percentile (IQR)
        plt.fill_between(
            n_components_list,
            q25_first_error_indices,
            q75_first_error_indices,
            alpha=0.3,
            color="red",
            label="Interquartile range (25th-75th)",
        )
        # Plot mean line
        plt.plot(
            n_components_list,
            mean_first_error_indices,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color="darkred",
            label="Mean first error index",
        )
        # Plot individual points with transparency to show density
        for n_comp, indices in zip(n_components_list, all_first_error_indices_per_component):
            plt.scatter([n_comp] * len(indices), indices, alpha=0.2, s=20, color="red", zorder=0)

        plt.xlabel("Number of PCA Components", fontsize=14)
        plt.ylabel("First Error Index (Sequence Position)", fontsize=14)
        plt.title(f"{title} - First Error Index", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=11)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tight_layout()
        error_index_outfile = outfile.replace(".png", "_first_error_index.png")
        plt.savefig(error_index_outfile, dpi=150)
        print(f"plot_pca_reconstruction_accuracy (first error index): {error_index_outfile}")
        plt.close()


def plot_pca_components_similarity_across_samples(
    by_sid: Dict[int, List[Dict[str, Any]]],
    outfile: str,
    n_components: int = 4,
):
    """Visualize how much PCA components differ across different samples.

    For each sample, fits PCA individually and compares principal components across samples.
    Shows similarity/difference between PC vectors from different samples.

    Args:
        by_sid: Dictionary mapping sample_id to list of stage records
        outfile: Output file path
        n_components: Number of principal components to compare
    """
    if len(by_sid) < 2:
        print("Need at least 2 samples to compare PCA components")
        return

    # Extract embeddings for each sample (use all stages to fit PCA)
    sample_embeddings_list = {}
    sample_labels = []
    for sid, stages in by_sid.items():
        # Use all stages from this sample to fit PCA
        embeddings_list = [flatten_embedding(s) for s in stages]
        if len(embeddings_list) == 0:
            continue
        sample_embeddings_list[sid] = embeddings_list
        sample_labels.append(f"Sample {sid}")

    if len(sample_embeddings_list) < 2:
        return

    # Fit PCA on each sample individually (using all stages)
    pca_models = {}
    pca_components = {}
    sample_ids = sorted(sample_embeddings_list.keys())

    for sid in tqdm(sample_ids, desc="Fitting PCA per sample"):
        embeddings_list = sample_embeddings_list[sid]
        # Stack all stage embeddings for this sample
        emb_2d = np.stack(embeddings_list, axis=0)  # [n_stages, n_features]

        n_comp = min(n_components, emb_2d.shape[1], emb_2d.shape[0] - 1)
        if n_comp < 1:
            continue

        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(emb_2d)
        pca_models[sid] = pca
        # Store components (each row is a PC)
        pca_components[sid] = pca.components_  # [n_components, n_features]

    if len(pca_models) < 2:
        return

    # Normalize all PC vectors for cosine similarity computation
    normalized_pcs = {}
    for sid in sample_ids:
        if sid in pca_components:
            normalized_pcs[sid] = []
            for pc_idx in range(pca_components[sid].shape[0]):
                pc_vec = pca_components[sid][pc_idx]
                pc_vec_norm = pc_vec / (np.linalg.norm(pc_vec) + 1e-12)
                normalized_pcs[sid].append(pc_vec_norm)
            normalized_pcs[sid] = np.array(normalized_pcs[sid])  # [n_components, n_features]

    # Get the maximum number of components across all samples
    max_n_comp = max([len(normalized_pcs[sid]) for sid in sample_ids])
    n_comp_actual = min(max_n_comp, n_components)

    # Create comprehensive similarity matrices: for each pair of samples, compare ALL their PCs
    n_samples = len(sample_ids)
    # _ = n_samples * (n_samples - 1) // 2 + n_samples  # Include self-comparisons

    # Create figure with subplots for each sample pair
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for plot_idx, sid_i in enumerate(sample_ids):
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]

        # For sample sid_i, compare its PCs with PCs from all other samples
        if sid_i not in normalized_pcs:
            ax.axis("off")
            continue

        pcs_i = normalized_pcs[sid_i]  # [n_comp_i, n_features]
        n_comp_i = min(pcs_i.shape[0], n_comp_actual)

        # Build similarity matrix: rows = PCs of sample i, cols = PCs of all other samples
        all_pcs_other = []
        other_labels = []
        for sid_j in sample_ids:
            if sid_j in normalized_pcs:
                pcs_j = normalized_pcs[sid_j]
                n_comp_j = min(pcs_j.shape[0], n_comp_actual)
                for pc_j_idx in range(n_comp_j):
                    all_pcs_other.append(pcs_j[pc_j_idx])
                    other_labels.append(f"S{sid_j}-PC{pc_j_idx + 1}")

        if len(all_pcs_other) == 0:
            ax.axis("off")
            continue

        all_pcs_other = np.array(all_pcs_other)  # [total_other_pcs, n_features]

        # Compute similarity matrix: [n_comp_i, total_other_pcs]
        similarity_matrix = np.zeros((n_comp_i, len(all_pcs_other)))
        for i in range(n_comp_i):
            for j in range(len(all_pcs_other)):
                sim = np.clip(np.dot(pcs_i[i], all_pcs_other[j]), -1.0, 1.0)
                similarity_matrix[i, j] = sim

        # Plot heatmap
        im = ax.imshow(similarity_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(all_pcs_other)))
        ax.set_yticks(np.arange(n_comp_i))
        ax.set_xticklabels(other_labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels([f"PC{i+1}" for i in range(n_comp_i)], fontsize=10)
        ax.set_title(f"Sample {sid_i} PCs vs All Other PCs", fontsize=11, fontweight="bold")
        ax.set_xlabel("Other Sample PCs", fontsize=9)
        ax.set_ylabel(f"Sample {sid_i} PCs", fontsize=9)

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # Hide unused subplots
    for idx in range(len(sample_ids), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Cross-Sample PCA Component Similarity\n(Each subplot: one sample's PCs vs all other samples' PCs)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"plot_pca_components_similarity_across_samples: {outfile}")
    plt.close()

    # Create a comprehensive cross-component similarity matrix
    # For each sample pair, create a matrix showing similarity between all their PCs
    # n_pairs_plot = min(n_samples * (n_samples - 1) // 2, 6)  # Limit to 6 pairs for readability
    pair_idx = 0
    _, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    axes2 = axes2.flatten()

    for i, sid_i in enumerate(sample_ids):
        if pair_idx >= len(axes2):
            break
        for j, sid_j in enumerate(sample_ids):
            if i >= j or pair_idx >= len(axes2):
                continue

            ax = axes2[pair_idx]

            if sid_i not in normalized_pcs or sid_j not in normalized_pcs:
                ax.axis("off")
                pair_idx += 1
                continue

            pcs_i = normalized_pcs[sid_i]
            pcs_j = normalized_pcs[sid_j]
            n_comp_i = min(pcs_i.shape[0], n_comp_actual)
            n_comp_j = min(pcs_j.shape[0], n_comp_actual)

            # Compute similarity matrix: [n_comp_i, n_comp_j]
            similarity_matrix = np.zeros((n_comp_i, n_comp_j))
            for pc_i_idx in range(n_comp_i):
                for pc_j_idx in range(n_comp_j):
                    sim = np.clip(np.dot(pcs_i[pc_i_idx], pcs_j[pc_j_idx]), -1.0, 1.0)
                    similarity_matrix[pc_i_idx, pc_j_idx] = sim

            # Plot heatmap
            im = ax.imshow(similarity_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(np.arange(n_comp_j))
            ax.set_yticks(np.arange(n_comp_i))
            ax.set_xticklabels([f"PC{k+1}" for k in range(n_comp_j)], fontsize=9)
            ax.set_yticklabels([f"PC{k+1}" for k in range(n_comp_i)], fontsize=9)
            ax.set_xlabel(f"Sample {sid_j} PCs", fontsize=10)
            ax.set_ylabel(f"Sample {sid_i} PCs", fontsize=10)
            ax.set_title(f"S{sid_i} vs S{sid_j}", fontsize=11, fontweight="bold")

            # Add text annotations (only if matrix is small enough)
            if n_comp_i <= 8 and n_comp_j <= 8:
                for pi in range(n_comp_i):
                    for pj in range(n_comp_j):
                        _ = ax.text(
                            pj, pi, f"{similarity_matrix[pi, pj]:.2f}", ha="center", va="center", color="black", fontsize=8
                        )

            plt.colorbar(im, ax=ax, label="Cosine Similarity")
            pair_idx += 1

    # Hide unused subplots
    for idx in range(pair_idx, len(axes2)):
        axes2[idx].axis("off")

    plt.suptitle(
        "Pairwise Cross-Component Similarity\n(All PCs of sample A vs all PCs of sample B)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    pairwise_outfile = outfile.replace(".png", "_pairwise.png")
    plt.savefig(pairwise_outfile, dpi=150)
    print(f"plot_pca_components_similarity_across_samples (pairwise): {pairwise_outfile}")
    plt.close()

    # Create summary: average similarity between matching and non-matching PC indices
    matching_similarities = []
    non_matching_similarities = []

    for i, sid_i in enumerate(sample_ids):
        if sid_i not in normalized_pcs:
            continue
        for j, sid_j in enumerate(sample_ids):
            if i >= j or sid_j not in normalized_pcs:
                continue

            pcs_i = normalized_pcs[sid_i]
            pcs_j = normalized_pcs[sid_j]
            n_comp_i = min(pcs_i.shape[0], n_comp_actual)
            n_comp_j = min(pcs_j.shape[0], n_comp_actual)

            for pc_i_idx in range(n_comp_i):
                for pc_j_idx in range(n_comp_j):
                    sim = np.clip(np.dot(pcs_i[pc_i_idx], pcs_j[pc_j_idx]), -1.0, 1.0)
                    if pc_i_idx == pc_j_idx:
                        matching_similarities.append(sim)
                    else:
                        non_matching_similarities.append(sim)

    # Plot summary comparison
    summary_outfile = outfile.replace(".png", "_summary.png")
    plt.figure(figsize=(10, 6))
    categories = ["Matching PC indices\n(e.g., PC1 vs PC1)", "Non-matching PC indices\n(e.g., PC1 vs PC2)"]
    means = [
        np.mean(matching_similarities) if matching_similarities else 0.0,
        np.mean(non_matching_similarities) if non_matching_similarities else 0.0,
    ]
    stds = [
        np.std(matching_similarities) if matching_similarities else 0.0,
        np.std(non_matching_similarities) if non_matching_similarities else 0.0,
    ]

    x_pos = np.arange(len(categories))
    _ = plt.bar(x_pos, means, yerr=stds, alpha=0.7, color=["steelblue", "coral"], edgecolor="black", linewidth=1.5, capsize=5)
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("Comparison Type", fontsize=14)
    plt.ylabel("Average Cosine Similarity", fontsize=14)
    plt.title("PCA Component Similarity: Matching vs Non-Matching Indices", fontsize=14, fontweight="bold")
    plt.xticks(x_pos, categories, fontsize=11)
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(-1, 1)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(
            i,
            mean + std + 0.05 if mean >= 0 else mean - std - 0.05,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(summary_outfile, dpi=150)
    print(f"plot_pca_components_similarity_across_samples (summary): {summary_outfile}")
    plt.close()


def plot_correlation(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    outfile: str,
    label_y_threshold: Optional[float] = None,
    point_labels: Optional[List[str]] = None,
):
    # Ensure x and y have the same length
    if len(x) != len(y):
        print(f"Warning: Skipping {title} - x and y have different lengths ({len(x)} vs {len(y)})")
        return

    plt.figure(figsize=(6, 4))
    # Create gradient colors based on position (first to last)
    n_points = len(x)
    if n_points > 0:
        positions = np.arange(n_points)
        # Normalize positions to [0, 1] for colormap
        # norm_positions = positions / max(positions.max(), 1.0) if positions.max() > 0 else positions
        # colors = plt.cm.viridis(norm_positions)
        # Create scatter plot with gradient colors
        scatter = plt.scatter(x, y, s=20, alpha=0.5, c=positions, cmap="viridis")
        # Add colorbar to show gradient meaning
        cbar = plt.colorbar(scatter, ax=plt.gca())
        cbar.set_label("position", rotation=270, labelpad=15)
    else:
        plt.scatter(x, y, s=20, alpha=0.5)
    # Add regression line
    sns.regplot(x=x, y=y, scatter=False, line_kws={"color": "red"})
    corr = np.corrcoef(x, y)[0, 1] if x.size > 1 and y.size > 1 else np.nan
    plt.title(f"{title} (r={corr:.3f})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add labels for points meeting the threshold condition
    if label_y_threshold is not None:
        mask = y > label_y_threshold
        if np.any(mask):
            for i in np.where(mask)[0]:
                label_text = (
                    point_labels[i] if point_labels is not None and i < len(point_labels) else f"({x[i]:.1f}, {y[i]:.1f})"
                )
                plt.annotate(
                    label_text,
                    (x[i], y[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved correlation plot to: {outfile}")


def plot_norms_over_stages(
    labels: List[str], mean_vals: List[float], max_vals: List[float], ylabel: str, title: str, outfile: str
):
    if len(mean_vals) == 0:
        return
    plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
    x = np.arange(len(labels))
    plt.plot(x, mean_vals, marker="o", label="mean")
    if len(max_vals) == len(mean_vals):
        plt.plot(x, max_vals, marker="s", label="max")
    plt.xticks(x, labels, rotation=0)
    plt.xlabel("stages")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def estimate_token_perplexity(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    # logits: [B, T, V], labels: [B, T], mask: [B, T]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = labels[:, 1:]
    m = mask[:, 1:].bool()
    nll = -log_probs.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    nll = nll[m]
    if nll.numel() == 0:
        return float("nan")
    ppl = torch.exp(nll.mean()).item()
    return float(ppl)


def compute_ppl_for_text(model: AutoModelForCausalLM, tok: AutoTokenizer, device: torch.device, text: str) -> Tuple[int, float]:
    with torch.no_grad():
        enc = tok(text, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        ppl = estimate_token_perplexity(out.logits, input_ids, attn)
        seq_len = int(attn.sum().item())
    return seq_len, ppl


def compute_distance_metrics(X: np.ndarray) -> float:
    # Returns (initial_final_l2, trajectory_length_l2)
    if X.shape[0] < 2:
        return 0.0, 0.0
    init_final = float(np.linalg.norm(X[-1] - X[0]))
    diffs = X[1:, :] - X[:-1, :]
    traj_len = float(np.linalg.norm(diffs, axis=1).sum())
    return init_final, traj_len


def compute_token_norm_stats_from_row(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (l1_per_token, l2_per_token) across all tokens in the embedding
    # Accepts embeddings of shape [..., hidden_dim]; flattens leading dims to tokens
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    hidden_dim = emb.shape[-1]
    emb2d = emb.reshape(-1, hidden_dim)
    l2 = torch.linalg.norm(emb2d, ord=2, dim=-1).detach().cpu().numpy()
    l1 = torch.linalg.norm(emb2d, ord=1, dim=-1).detach().cpu().numpy()
    return l1, l2


def maybe_compute_perplexity(
    rows: List[Dict[str, Any]],
    model_name: Optional[str],
    max_eval_samples: int,
) -> Tuple[List[int], List[float]]:
    if model_name is None or len(rows) == 0 or max_eval_samples <= 0:
        return [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tok = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    except Exception:
        return [], []

    seq_lens: List[int] = []
    ppls: List[float] = []
    with torch.no_grad():
        for r in tqdm(rows[:max_eval_samples], desc="Computing perplexity"):
            text = r.get("text", "")
            if not isinstance(text, str) or text.strip() == "":
                continue
            enc = tok(text, truncation=True, padding=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            ppl = estimate_token_perplexity(out.logits, input_ids, attn)
            seq_lens.append(int(attn.sum().item()))
            ppls.append(float(ppl))
    return seq_lens, ppls


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze progressive_train artifacts")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to progressive_prefixes dataset",
    )
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample_id filter")
    parser.add_argument("--stage_index", type=int, default=None, help="Optional stage filter")
    parser.add_argument("--process_samples", default=False, action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures and metrics",
    )
    parser.add_argument(
        "--perplexity_model",
        type=str,
        default=None,
        help="HF model name to compute token-level perplexity of sample texts",
    )
    parser.add_argument(
        "--perplexity_max_samples",
        type=int,
        default=64,
        help="Max rows to use for perplexity estimation",
    )

    args = parser.parse_args()

    os.makedirs("artifacts/visualizations", exist_ok=True)
    out_dir = args.output_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("artifacts/visualizations", f"progressive_embeddings_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_progressive_dataset(args.dataset_path)
    rows = filter_records(ds, sample_id=args.sample_id, stage_index=args.stage_index)
    if not rows:
        raise ValueError("No records found with given filters.")

    # Group by sample and build stage-wise matrices
    by_sid = collate_stages_by_sample(rows)

    # For each sample: compute pairwise distances between stages and PCA
    sns.set(style="whitegrid")
    summary_steps: List[int] = []
    summary_conv: List[float] = []
    summary_seq_len: List[int] = []

    # Prepare optional perplexity model once
    model_for_ppl: Optional[str] = args.perplexity_model
    if model_for_ppl is None:
        names = [str(r.get("model_checkpoint", "")).strip() for r in rows]
        names = [n for n in names if n]
        if names:
            uniq = {}
            for n in names:
                uniq[n] = uniq.get(n, 0) + 1
            model_for_ppl = max(uniq.items(), key=lambda kv: kv[1])[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tok = None
    if model_for_ppl is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_for_ppl).to(device)
            model.eval()
            tok = AutoTokenizer.from_pretrained(model_for_ppl)
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        except Exception:
            model = None
            tok = None

    # Holders for cross-sample correlation analyses
    dist_l1_all: List[float] = []
    dist_l2_all: List[float] = []
    dist_cosine_all: List[float] = []
    ppl_all: List[float] = []
    seq_len_all: List[int] = []
    sid_all: List[int] = []
    length_vs_steps_labels: List[str] = []
    all_compression_embeddings: List[np.ndarray] = []
    all_rows_for_pca: List[Dict[str, Any]] = []

    for sid, stages in tqdm(by_sid.items(), desc="Processing samples for aggragate"):
        # Collect compression embeddings for aggregate analysis
        for s in stages:
            all_compression_embeddings.append(flatten_embedding(s))
            all_rows_for_pca.append(s)

    if args.process_samples:
        for sid, stages in tqdm(by_sid.items(), desc="Processing samples"):
            labels = [f"L{int(s.get('stage_seq_len', -1))}" for s in stages]
            X = np.stack([flatten_embedding(s) for s in stages], axis=0)
            l2, cos_d = compute_pairwise_similarities(X)
            plot_heatmap(
                l2,
                labels,
                title=f"Sample {sid}: L2 by stage",
                outfile=os.path.join(out_dir, f"sid{sid}_l2.png"),
            )
            plot_heatmap(
                cos_d,
                labels,
                title=f"Sample {sid}: cosine distance by stage",
                outfile=os.path.join(out_dir, f"sid{sid}_cosine.png"),
            )
            plot_pca(X, labels, outfile=os.path.join(out_dir, f"sid{sid}_pca.png"))
            plot_pca_4_components(X, labels, outfile=os.path.join(out_dir, f"sid{sid}_pca4.png"))
            plot_cumulative_explained_variance(
                X,
                max_components=16,
                title=f"Sample {sid}: Cumulative Explained Variance",
                outfile=os.path.join(out_dir, f"sid{sid}_cumulative_variance.png"),
            )
            # plot_pca_components_vs_sequence_length(
            #     stages,
            #     sample_id=sid,
            #     outfile=os.path.join(out_dir, f"sid{sid}_pca_components_vs_seq_len.png"),
            #     target_seq_lengths=[4, 16, 32, 48, 64, 96, 128],
            # )
            # if model is not None and tok is not None:
            #     plot_pca_reconstruction_accuracy(
            #         stages,
            #         model,
            #         tok,
            #         device,
            #         title=f"Sample {sid}: PCA Reconstruction Accuracy",
            #         outfile=os.path.join(out_dir, f"sid{sid}_pca_reconstruction_accuracy.png"),
            #         max_components=16,
            #     )

            # Collect per-stage stats
            for s in stages:
                steps = int(s.get("steps_taken", 0))
                conv = float(s.get("final_convergence", np.nan)) if s.get("final_convergence") is not None else np.nan
                seql = int(s.get("stage_seq_len", -1))
                # stage_idx = int(s.get("stage_index", -1))
                summary_steps.append(steps)
                summary_conv.append(conv)
                summary_seq_len.append(seql)
                length_vs_steps_labels.append(f"L{seql}")

            # Per-sample distance metrics
            for i in range(X.shape[0] - 1):
                # Compute L1 distance
                l1_dist = float(np.linalg.norm(X[i + 1] - X[i], ord=1))
                dist_l1_all.append(l1_dist)
                # Compute L2 distance
                l2_dist = float(np.linalg.norm(X[i + 1] - X[i], ord=2))
                dist_l2_all.append(l2_dist)
                # Compute cosine distance: 1 - cosine_similarity
                v1 = X[i + 1] / (np.linalg.norm(X[i + 1]) + 1e-12)
                v2 = X[i] / (np.linalg.norm(X[i]) + 1e-12)
                cos_sim = np.clip(np.dot(v1, v2), -1.0, 1.0)
                cos_dist = 1.0 - cos_sim
                dist_cosine_all.append(float(cos_dist))

            # Per-sample perplexity (optional)
            if model is not None and tok is not None:
                sample_text = None
                for s in stages:
                    sample_text = s.get("text", None)
                    if sample_text is not None:
                        seql, ppl = compute_ppl_for_text(model, tok, device, sample_text)
                        if math.isnan(ppl):
                            continue

                        seq_len_all.append(seql)
                        ppl_all.append(float(ppl))
                        sid_all.append(int(sid))

            # Per-sample token norm trajectories across stages
            mean_l2_by_stage: List[float] = []
            max_l2_by_stage: List[float] = []
            mean_l1_by_stage: List[float] = []
            max_l1_by_stage: List[float] = []
            for s in stages:
                try:
                    l1_tok, l2_tok = compute_token_norm_stats_from_row(s)
                    if l1_tok.size == 0 or l2_tok.size == 0:
                        mean_l1_by_stage.append(float("nan"))
                        max_l1_by_stage.append(float("nan"))
                        mean_l2_by_stage.append(float("nan"))
                        max_l2_by_stage.append(float("nan"))
                    else:
                        mean_l1_by_stage.append(float(np.mean(l1_tok)))
                        max_l1_by_stage.append(float(np.max(l1_tok)))
                        mean_l2_by_stage.append(float(np.mean(l2_tok)))
                        max_l2_by_stage.append(float(np.max(l2_tok)))
                except Exception:
                    mean_l1_by_stage.append(float("nan"))
                    max_l1_by_stage.append(float("nan"))
                    mean_l2_by_stage.append(float("nan"))
                    max_l2_by_stage.append(float("nan"))

            # Plot L2 and L1 norm trajectories for this sample
            plot_norms_over_stages(
                labels,
                mean_l2_by_stage,
                max_l2_by_stage,
                ylabel="token L2 norm",
                title=f"Sample {sid}: token L2 norms across stages",
                outfile=os.path.join(out_dir, f"sid{sid}_token_norms_l2.png"),
            )
            plot_norms_over_stages(
                labels,
                mean_l1_by_stage,
                max_l1_by_stage,
                ylabel="token L1 norm",
                title=f"Sample {sid}: token L1 norms across stages",
                outfile=os.path.join(out_dir, f"sid{sid}_token_norms_l1.png"),
            )

    # Aggregate PCA reconstruction accuracy across all compression embeddings
    if len(all_rows_for_pca) > 0 and model is not None and tok is not None:
        plot_pca_reconstruction_accuracy(
            all_rows_for_pca,
            model,
            tok,
            device,
            title="Aggregate: PCA Reconstruction Accuracy (All Compression Embeddings)",
            outfile=os.path.join(out_dir, "aggregate_pca_reconstruction_accuracy.png"),
            max_components=32,
        )

    # Aggregate PCA components vs sequence length across all samples
    if len(by_sid) > 0:
        plot_pca_components_vs_sequence_length_aggregate(
            by_sid,
            outfile=os.path.join(out_dir, "aggregate_pca_components_vs_seq_len.png"),
            target_seq_lengths=[4, 16, 32, 48, 64, 96, 128],
        )

    # Visualize PCA component similarity across samples
    if len(by_sid) >= 2:
        plot_pca_components_similarity_across_samples(
            by_sid,
            outfile=os.path.join(out_dir, "pca_components_similarity_across_samples.png"),
            n_components=8,
        )

    # Correlation plots across all stages
    if len(summary_steps) > 1 and len(summary_conv) == len(summary_steps):
        plot_correlation(
            np.array(summary_steps),
            np.array(summary_conv),
            xlabel="steps_taken",
            ylabel="final_convergence",
            title="Steps vs Convergence",
            outfile=os.path.join(out_dir, "steps_vs_convergence.png"),
        )
    if len(summary_seq_len) > 1 and len(summary_steps) == len(summary_seq_len):
        plot_correlation(
            np.array(summary_seq_len),
            np.array(summary_steps),
            xlabel="stage_seq_len",
            ylabel="steps_taken",
            title="Length vs Steps",
            outfile=os.path.join(out_dir, "length_vs_steps.png"),
            label_y_threshold=50,
            point_labels=length_vs_steps_labels if len(length_vs_steps_labels) == len(summary_steps) else None,
        )

    # Note: ppl_all is per-sample, summary_steps is per-stage
    # They have different structures, so we can only plot if lengths happen to match
    if len(ppl_all) > 1 and len(ppl_all) == len(summary_steps):
        plot_correlation(
            np.array(ppl_all),
            np.array(summary_steps),
            xlabel="ppl",
            ylabel="steps_taken",
            title="PPL vs Steps",
            outfile=os.path.join(out_dir, "ppl_vs_steps.png"),
        )

    # Optional: plots leveraging per-sample perplexities (if available)
    if len(ppl_all) > 1 and len(ppl_all) == len(dist_l1_all):
        plot_correlation(
            np.array(dist_l1_all),
            np.array(ppl_all),
            xlabel="L1 distance",
            ylabel="perplexity",
            title="Comp Embeddings L1 Distance vs Perplexity",
            outfile=os.path.join(out_dir, "l1_dist_vs_perplexity.png"),
        )
    if len(ppl_all) > 1 and len(ppl_all) == len(dist_l2_all):
        plot_correlation(
            np.array(dist_l2_all),
            np.array(ppl_all),
            xlabel="L2 distance",
            ylabel="perplexity",
            title="Comp Embeddings L2 Distance vs Perplexity",
            outfile=os.path.join(out_dir, "l2_dist_vs_perplexity.png"),
        )
    if len(ppl_all) > 1 and len(ppl_all) == len(dist_cosine_all):
        plot_correlation(
            np.array(dist_cosine_all),
            np.array(ppl_all),
            xlabel="cosine distance",
            ylabel="perplexity",
            title="Comp Embeddings Cosine Distance vs Perplexity",
            outfile=os.path.join(out_dir, "cosine_dist_vs_perplexity.png"),
        )
    if len(seq_len_all) > 1 and len(seq_len_all) == len(ppl_all):
        plot_correlation(
            np.array(seq_len_all),
            np.array(ppl_all),
            xlabel="sequence_length",
            ylabel="perplexity",
            title="Length vs Perplexity",
            outfile=os.path.join(out_dir, "length_vs_perplexity.png"),
        )

    # Save a summary CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("sample_id,stage_index,stage_seq_len,steps_taken,final_convergence\n")
        for sid, stages in by_sid.items():
            for s in stages:
                f.write(
                    f"{sid},{int(s.get('stage_index', -1))},{int(s.get('stage_seq_len', -1))},{int(s.get('steps_taken', 0))},{float(s.get('final_convergence', np.nan))}\n"
                )

    print(f"Saved progressive figures and metrics to: {out_dir}")


if __name__ == "__main__":
    main()
