import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.visualize_progressive_embeddings import (
    collate_stages_by_sample,
    compute_pca_components_for_sample,
    filter_records,
    load_progressive_dataset,
)
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_pca_components_sequence_length_stats(
    by_sid: Dict[int, List[Dict[str, Any]]],
    target_seq_lengths: List[int],
) -> Optional[Dict[str, Any]]:
    components_by_seq_len: Dict[int, List[int]] = {seq_len: [] for seq_len in target_seq_lengths}

    for sid, stages in tqdm(by_sid.items(), desc="Computing PCA components per sample"):
        results = compute_pca_components_for_sample(stages, target_seq_lengths)
        for seq_len, n_components in results.items():
            if n_components is not None:
                components_by_seq_len[seq_len].append(n_components)

    seq_lengths: List[int] = []
    all_components_per_seq_len: List[List[int]] = []

    for seq_len in sorted(target_seq_lengths):
        if seq_len in components_by_seq_len and len(components_by_seq_len[seq_len]) > 0:
            seq_lengths.append(seq_len)
            all_components_per_seq_len.append(components_by_seq_len[seq_len])

    if len(seq_lengths) == 0:
        return None

    mean_components = [np.mean(comps) for comps in all_components_per_seq_len]
    q25_components = [np.percentile(comps, 25) for comps in all_components_per_seq_len]
    q75_components = [np.percentile(comps, 75) for comps in all_components_per_seq_len]
    q10_components = [np.percentile(comps, 10) for comps in all_components_per_seq_len]
    q90_components = [np.percentile(comps, 90) for comps in all_components_per_seq_len]

    return {
        "seq_lengths": seq_lengths,
        "all_components": all_components_per_seq_len,
        "mean": mean_components,
        "q25": q25_components,
        "q75": q75_components,
        "q10": q10_components,
        "q90": q90_components,
    }


def infer_checkpoint_label(dataset_path: str) -> str:
    label = os.path.basename(os.path.dirname(dataset_path))
    label = label.removeprefix("sl_4096_")
    if "lr_" not in label:
        label = label + "_lr_0.01"

    label = label.replace("_lr_", " lr=")
    label = label.replace("Meta-", "")
    label = label.replace("Llama-3.1-8B ", "")

    return label


def get_experiment_dir(dataset_path: str) -> str:
    return os.path.dirname(dataset_path)


def make_cache_key(
    dataset_path: str,
    stage_index: Optional[int],
    target_seq_lengths: List[int],
) -> Dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "stage_index": stage_index,
        "target_seq_lengths": target_seq_lengths,
    }


def load_cached_entry(cache_path: str, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
    except Exception:
        return None
    if not isinstance(cache_data, dict):
        return None
    entries = cache_data.get("entries", [])
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if entry.get("key") == cache_key:
            if isinstance(entry, dict):
                return entry
    return None


def save_cached_stats(
    cache_path: str,
    cache_key: Dict[str, Any],
    stats: Dict[str, Any],
    label: str,
) -> None:
    cache_data = {"version": 1, "entries": []}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                cache_data = existing
        except Exception:
            cache_data = {"version": 1, "entries": []}
    cache_data["entries"] = [entry for entry in cache_data.get("entries", []) if entry.get("key") != cache_key]
    cache_data["entries"].append({"key": cache_key, "label": label, "stats": make_json_safe(stats)})
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def plot_pca_components_vs_sequence_length_multi(
    stats_by_label: List[Tuple[str, Dict[str, Any]]],
    outfile: str,
):
    if len(stats_by_label) == 0:
        print("No valid data points for aggregate PCA components vs sequence length")
        return

    plt.figure(figsize=(10, 6))
    fint_size = 25
    matplotlib.rcParams.update({"font.size": fint_size})

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["blue"])

    for idx, (label, stats) in enumerate(stats_by_label):
        color = color_cycle[idx % len(color_cycle)]
        seq_lengths = stats["seq_lengths"]
        mean_components = stats["mean"]
        q25_components = stats["q25"]
        q75_components = stats["q75"]

        plt.fill_between(
            seq_lengths,
            q25_components,
            q75_components,
            alpha=0.15,
            color=color,
        )
        plt.plot(
            seq_lengths,
            mean_components,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=color,
            label=label,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("PCA Components")
    # plt.title(" vs Sequence Length (All Samples)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"plot_pca_components_vs_sequence_length_multi: {outfile}")
    outfile_png = outfile.replace(".pdf", ".png")
    plt.savefig(outfile_png, dpi=150)
    print(f"plot_pca_components_vs_sequence_length_multi: {outfile_png}")

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
                compression_outputs = model(
                    inputs_embeds=input_embeds.to(torch.bfloat16), attention_mask=extended_attention_mask
                )

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

    # breakpoint()

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
    plt.ylabel("Accuracy (teacher-forcing)", fontsize=14)
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


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze progressive_train artifacts")
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to progressive_prefixes dataset",
    )
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample_id filter")
    parser.add_argument("--stage_index", type=int, default=None, help="Optional stage filter")
    parser.add_argument("--process_samples", default=False, action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/paper/",
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
    parser.add_argument(
        "--draw-landscape",
        default=False,
        action="store_true",
        help="Draw loss landscape for PCA component pairs",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=2.0,
        help="Maximum radius for neighborhood loss computation in PCA space",
    )
    parser.add_argument(
        "--draw-landscape-points-step",
        type=int,
        default=1,
        help="Compute landscape only for every Nth point (default: 1, compute for all points)",
    )
    parser.add_argument(
        "--draw-landscape-points-limit",
        type=int,
        default=None,
        help="Limit number of points for GIF visualization (default: None, use all points)",
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=40,
        help="Resolution of the mesh grid for loss landscape computation (default: 40)",
    )
    parser.add_argument(
        "--landscape_pairs_limit",
        type=int,
        default=2,
        help="Limit number of PCA component pairs to compute landscape for (default: 2)",
    )
    parser.add_argument(
        "--target_seq_lengths",
        type=int,
        nargs="+",
        default=[4, 16, 32, 48, 64, 96, 128, 256, 512, 1024],
        help="Sequence lengths to analyze",
    )
    parser.add_argument(
        "--out_file_suffix",
        type=str,
        default="",
    )

    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset_paths = args.dataset_path
    stats_by_label: List[Tuple[str, Dict[str, Any]]] = []

    for dataset_path in dataset_paths:
        experiment_dir = get_experiment_dir(dataset_path)
        cache_path = os.path.join(experiment_dir, "pca_seq_len_cache.json")
        cache_key = make_cache_key(dataset_path, args.stage_index, args.target_seq_lengths)
        cached_entry = load_cached_entry(cache_path, cache_key)
        stats: Optional[Dict[str, Any]] = None
        label: Optional[str] = None
        rows: List[Dict[str, Any]] = []
        if cached_entry is None:
            ds = load_progressive_dataset(dataset_path)
            rows = filter_records(ds, stage_index=args.stage_index)
            if not rows:
                raise ValueError(f"No records found with given filters in {dataset_path}.")

            by_sid = collate_stages_by_sample(rows)
            stats = compute_pca_components_sequence_length_stats(
                by_sid,
                target_seq_lengths=args.target_seq_lengths,
            )
            if stats is None:
                print(f"No valid data points for {dataset_path}")
                continue
            save_cached_stats(cache_path, cache_key, stats, label)
        else:
            stats = cached_entry.get("stats")

        if stats is None:
            print(f"No valid data points for {dataset_path}")
            continue

        label = infer_checkpoint_label(dataset_path)
        stats_by_label.append((label, stats))

    if len(stats_by_label) == 0:
        raise ValueError("No valid data points for any dataset path.")

    out_file_suffix = args.out_file_suffix
    if out_file_suffix != "":
        out_file_suffix = "_" + out_file_suffix
    outfile = os.path.join(out_dir, f"aggregate_pca_components_vs_seq_len{out_file_suffix}.pdf")

    plot_pca_components_vs_sequence_length_multi(stats_by_label, outfile=outfile)


if __name__ == "__main__":
    main()
