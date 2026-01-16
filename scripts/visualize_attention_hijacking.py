import argparse
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_progressive_dataset(dataset_path: str) -> Dataset:
    """Load progressive checkpoint dataset."""
    return Dataset.load_from_disk(dataset_path)


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter records by sample_id and/or stage_index."""
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


def extract_attention_mass_for_all_seq_lengths(
    attentions: tuple,
    num_compression_tokens: int,
    target_seq_lengths: List[int],
    block_size: int = 16,
    block_threshold: int = 0,
) -> Dict[int, Dict[int, float]]:
    """
    Extract attention mass percent from full attention map for all target sequence lengths at once.

    For very long sequences (target_seq_len > block_threshold), attention mass is averaged in
    non-overlapping blocks over sequence length with block size = block_size. The returned dict will
    contain one entry per block (keyed by the block's max target_seq_len). For
    target_seq_len <= block_threshold, values are returned for each individual length.

    Args:
        attentions: Tuple of attention tensors, one per layer [batch_size, num_heads, seq_len, seq_len]
        num_compression_tokens: Number of compression tokens
        target_seq_lengths: List of target sequence lengths (input tokens, excluding compression tokens)

    Returns:
        Dictionary mapping target_seq_len to layer_index to attention_mass_percent
    """
    if not attentions:
        return {}
    if num_compression_tokens < 1:
        raise ValueError("num_compression_tokens must be >= 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_threshold < 0:
        raise ValueError("block_threshold must be >= 0")

    num_layers = len(attentions)
    total_seq_len = attentions[0].shape[-1]
    if num_compression_tokens > total_seq_len:
        raise ValueError(f"num_compression_tokens ({num_compression_tokens}) exceeds total_seq_len ({total_seq_len})")

    # Compute per-layer compression attention per query position without materializing
    # the full [layers, seq_len, seq_len] tensor.
    #
    # For each layer, we compute:
    #   comp_attn[q] = sum_k attn(q -> k) for k in [0, num_compression_tokens)
    # resulting in [seq_len] per layer.
    compression_attention_per_layer = torch.stack(
        [attn_layer.mean(dim=1)[0, :, :num_compression_tokens].sum(dim=-1) for attn_layer in attentions],
        dim=0,
    )  # [num_layers, seq_len]

    # Prefix means let us answer all effective lengths in O(1) each.
    # prefix_mean[:, e] = mean(compression_attention_per_layer[:, : e + 1], dim=-1)
    prefix_sums = compression_attention_per_layer.cumsum(dim=-1).cpu()  # [num_layers, seq_len]

    results: Dict[int, Dict[int, float]] = {}
    # Build processing items: individual lengths up to block_threshold, then block_size-length blocks thereafter.
    # Blocks are keyed by their max target_seq_len to keep x-axis monotonic for plotting.
    lengths = sorted(set(int(x) for x in target_seq_lengths))
    block_start_min = block_threshold + 1

    blocks: Dict[int, List[int]] = {}
    items: List[Dict[str, Any]] = []
    for t in lengths:
        if t <= block_threshold:
            items.append({"kind": "single", "lengths": [t], "key": t})
            continue
        block_start = block_start_min + ((t - block_start_min) // block_size) * block_size
        blocks.setdefault(block_start, []).append(t)

    for block_start in sorted(blocks.keys()):
        members = blocks[block_start]
        items.append({"kind": "block", "lengths": members, "key": max(members)})

    for item in tqdm(items, desc="Save results attention mass all seq lengths"):
        members = item["lengths"]
        effective_lengths = [num_compression_tokens + t for t in members]
        effective_lengths = [e for e in effective_lengths if 1 <= e <= total_seq_len]
        if not effective_lengths:
            continue

        # Compute per-length means: prefix_sums[:, e-1] / e, then average across lengths.
        idx = torch.tensor([e - 1 for e in effective_lengths], dtype=torch.long)
        denom = torch.tensor(effective_lengths, dtype=prefix_sums.dtype).unsqueeze(0)
        per_length_means = prefix_sums[:, idx] / denom  # [num_layers, num_lengths]
        mean_comp_attn = per_length_means.mean(dim=1)  # [num_layers]

        layer_vals = (mean_comp_attn * 100.0).tolist()
        results[int(item["key"])] = {layer_idx: layer_vals[layer_idx] for layer_idx in range(num_layers)}

    return results


def compute_attention_mass_for_stages(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stages: List[Dict[str, Any]],
    device: torch.device,
    attention_block_size: int = 16,
    target_seq_lengths_override: Optional[List[int]] = None,
) -> Dict[int, Dict[int, float]]:
    """
    Compute attention mass percent for all stages.

    Uses the longest sequence to compute attention once, then extracts attention mass
    for each stage from the full attention map (due to causal attention mask).

    Args:
        model: Language model
        tokenizer: Tokenizer
        stages: List of stage records, sorted by stage_index
        device: Device to run on

    Returns:
        Dictionary mapping stage_seq_len to layer_index to attention_mass_percent
    """
    if not stages:
        return {}

    # Find the longest sequence and get its compression embeddings and text
    longest_stage = max(stages, key=lambda s: int(s.get("stage_seq_len", 0)))
    max_seq_len = int(longest_stage.get("stage_seq_len", -1))

    if max_seq_len < 1:
        return {}

    # Extract compression embeddings for longest sequence
    embedding = longest_stage.get("embedding")
    if embedding is None:
        return {}

    # Convert to tensor
    if isinstance(embedding, list):
        compression_embeddings = torch.tensor(embedding, dtype=torch.float32)
    else:
        compression_embeddings = torch.tensor(embedding, dtype=torch.float32)

    # Get number of compression tokens
    num_compression_tokens = int(longest_stage.get("num_compression_tokens", 1))

    # Get text
    text = longest_stage.get("text", "")
    if not isinstance(text, str) or text.strip() == "":
        return {}

    # Compute attention once for the longest sequence
    print(f"Computing attention for longest sequence (length={max_seq_len})...")
    model.eval()
    with torch.no_grad():
        # Tokenize text
        enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Get input embeddings
        input_embeddings_layer = model.get_input_embeddings()
        input_text_embeds = input_embeddings_layer(input_ids)

        # Concatenate compression embeddings with input text embeddings
        compression_embeddings = compression_embeddings.to(device).to(input_text_embeds.dtype)
        # Add batch dimension: [1, num_compression_tokens, hidden_size]
        compression_embeddings = compression_embeddings.unsqueeze(0)
        input_embeds = torch.cat([compression_embeddings, input_text_embeds], dim=1)

        # Extend attention mask to include compression tokens
        comp_attention = torch.ones(
            (attention_mask.shape[0], num_compression_tokens), device=device, dtype=attention_mask.dtype
        )
        extended_attention_mask = torch.cat([comp_attention, attention_mask], dim=1)

        # Forward pass with attention outputs
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=extended_attention_mask,
            output_attentions=True,
        )

        # Extract attention weights
        attentions = outputs.attentions

        # Check if attentions are available
        if attentions is None:
            raise ValueError(
                "Attention weights are None. The model may not support output_attentions. "
                "Try setting model.set_attn_implementation('eager') before loading."
            )

    if target_seq_lengths_override is not None:
        target_seq_lengths = list(target_seq_lengths_override)
    else:
        # Get all unique sequence lengths from stages
        target_seq_lengths = sorted(set(int(s.get("stage_seq_len", -1)) for s in stages if int(s.get("stage_seq_len", -1)) > 0))

    # Extract attention mass for all sequence lengths at once
    print(f"Extracting attention mass for {len(target_seq_lengths)} sequence lengths...")
    results = extract_attention_mass_for_all_seq_lengths(
        attentions=attentions,
        num_compression_tokens=num_compression_tokens,
        target_seq_lengths=target_seq_lengths,
        block_size=attention_block_size,
    )

    return results


def plot_attention_hijacking_heatmap(
    results: Dict[int, Dict[int, float]],
    sample_id: Optional[int],
    output_path: str,
):
    """
    Plot heatmap of attention mass percent vs sequence length vs layer.

    Args:
        results: Dictionary mapping stage_seq_len to layer_index to attention_mass_percent
        sample_id: Optional sample ID for title
        output_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return

    # Collect all sequence lengths and layer indices
    seq_lengths = sorted(results.keys())
    all_layer_indices = set()
    for seq_len_data in results.values():
        all_layer_indices.update(seq_len_data.keys())
    layer_indices = sorted(all_layer_indices)

    # Build heatmap matrix: rows = layers, cols = sequence lengths
    heatmap_data = np.zeros((len(layer_indices), len(seq_lengths)))

    for col_idx, seq_len in enumerate(seq_lengths):
        layer_data = results[seq_len]
        for row_idx, layer_idx in enumerate(layer_indices):
            if layer_idx in layer_data:
                heatmap_data[row_idx, col_idx] = layer_data[layer_idx]

    # Create heatmap
    plt.figure(figsize=(max(8, len(seq_lengths) * 0.8), max(6, len(layer_indices) * 0.5)))
    ax = sns.heatmap(
        heatmap_data,
        xticklabels=seq_lengths,
        yticklabels=[f"Layer {idx}" for idx in layer_indices],
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Attention Mass % on Compression Tokens"},
        vmin=0,
        vmax=100,
    )
    # Flip layers axis: lower layer indices appear at the bottom.
    ax.invert_yaxis()
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Layer", fontsize=12)
    title = "Attention Hijacking: Compression Token Attention Mass %"
    if sample_id is not None:
        title += f" (Sample {sample_id})"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def average_attention_mass_results(
    results_list: List[Dict[int, Dict[int, float]]],
) -> Dict[int, Dict[int, float]]:
    if not results_list:
        return {}

    sums: Dict[int, Dict[int, float]] = {}
    counts: Dict[int, Dict[int, int]] = {}
    for res in results_list:
        for seq_len, layer_map in res.items():
            for layer_idx, val in layer_map.items():
                sums.setdefault(seq_len, {}).setdefault(layer_idx, 0.0)
                counts.setdefault(seq_len, {}).setdefault(layer_idx, 0)
                sums[seq_len][layer_idx] += float(val)
                counts[seq_len][layer_idx] += 1

    out: Dict[int, Dict[int, float]] = {}
    for seq_len in sorted(sums.keys()):
        out[seq_len] = {}
        for layer_idx in sorted(sums[seq_len].keys()):
            c = counts[seq_len][layer_idx]
            if c > 0:
                out[seq_len][layer_idx] = sums[seq_len][layer_idx] / c
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize attention hijacking with compression tokens")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to progressive_prefixes dataset",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (if not provided, will try to infer from dataset)",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Optional sample_id filter (if not provided, will process all samples)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures (default: inferred from dataset_path)",
    )
    parser.add_argument(
        "--min_seq_length",
        type=int,
        default=1,
        help="Filter out samples whose max stage_seq_len is < this value.",
    )
    parser.add_argument(
        "--attention_block_size",
        type=int,
        default=16,
        help="Block size for averaging attention for long sequences (target_seq_len > 100).",
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        # Try to infer from dataset path
        dataset_path = args.dataset_path
        if "artifacts/experiments" in dataset_path or "artifacts/experiments_progressive" in dataset_path:
            exp_dir = os.path.dirname(dataset_path)
            output_dir = os.path.join(exp_dir, "attention_visualizations")
        else:
            output_dir = "attention_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    ds = load_progressive_dataset(args.dataset_path)

    # Filter records
    rows = filter_records(ds, sample_id=args.sample_id)
    if not rows:
        raise ValueError("No records found with given filters.")

    # Group by sample
    by_sid = collate_stages_by_sample(rows)

    # Determine model checkpoint
    model_checkpoint = args.model_checkpoint
    if model_checkpoint is None:
        # Try to infer from dataset
        if rows:
            model_checkpoint = rows[0].get("model_checkpoint", "")
            if not model_checkpoint:
                raise ValueError(
                    "model_checkpoint not provided and cannot be inferred from dataset. " "Please provide --model_checkpoint."
                )
        else:
            raise ValueError("No rows found to infer model_checkpoint from.")

    print(f"Using model checkpoint: {model_checkpoint}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    # Set attention implementation to 'eager' to enable output_attentions
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            attn_implementation="eager",
        ).to(device)
    except TypeError:
        # Fallback for older transformers versions that don't support attn_implementation
        print("Warning: attn_implementation parameter not supported, loading model without it...")
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
        # Try to set attention implementation after loading
        try:
            model.set_attn_implementation("eager")
        except (AttributeError, ValueError):
            print("Warning: Could not set attention implementation to 'eager'. Attention outputs may not be available.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process each sample
    if args.min_seq_length < 1:
        raise ValueError("--min_seq_length must be >= 1")

    if args.sample_id is None:
        # Average heatmaps over all samples, limiting to the minimum max sequence length across samples.
        eligible_by_sid: Dict[int, List[Dict[str, Any]]] = {}
        per_sample_max = []
        for _sid, stages in by_sid.items():
            max_len = max((int(s.get("stage_seq_len", -1)) for s in stages), default=-1)
            if max_len >= args.min_seq_length:
                eligible_by_sid[_sid] = stages
                per_sample_max.append(max_len)
        if not per_sample_max:
            raise ValueError(
                f"No samples with max stage_seq_len >= {args.min_seq_length} found. "
                "Lower --min_seq_length or check the dataset."
            )

        min_max_len = min(per_sample_max)
        print(
            f"\nAveraging over {len(eligible_by_sid)} samples; "
            f"using target_seq_len in [{args.min_seq_length}, {min_max_len}]"
        )
        target_seq_lengths_override = list(range(args.min_seq_length, min_max_len + 1))

        all_results: List[Dict[int, Dict[int, float]]] = []
        for sample_id, stages in eligible_by_sid.items():
            print(f"\nProcessing sample {sample_id} with {len(stages)} stages...")
            results = compute_attention_mass_for_stages(
                model=model,
                tokenizer=tokenizer,
                stages=stages,
                device=device,
                attention_block_size=args.attention_block_size,
                target_seq_lengths_override=target_seq_lengths_override,
            )
            if results:
                all_results.append(results)

        avg_results = average_attention_mass_results(all_results)
        output_path = os.path.join(output_dir, "attention_hijacking_avg.png")
        plot_attention_hijacking_heatmap(
            results=avg_results,
            sample_id=None,
            output_path=output_path,
        )
    else:
        for sample_id, stages in by_sid.items():
            max_len = max((int(s.get("stage_seq_len", -1)) for s in stages), default=-1)
            if max_len < args.min_seq_length:
                print(f"\nSkipping sample {sample_id}: max stage_seq_len={max_len} < min_seq_length={args.min_seq_length}")
                continue
            print(f"\nProcessing sample {sample_id} with {len(stages)} stages...")
            results = compute_attention_mass_for_stages(
                model=model,
                tokenizer=tokenizer,
                stages=stages,
                device=device,
                attention_block_size=args.attention_block_size,
            )
            output_path = os.path.join(output_dir, f"attention_hijacking_sample_{sample_id}.png")
            plot_attention_hijacking_heatmap(
                results=results,
                sample_id=sample_id,
                output_path=output_path,
            )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
