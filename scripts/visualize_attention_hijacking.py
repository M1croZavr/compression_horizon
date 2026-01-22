import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_dataset_type(ds: Dataset) -> str:
    """
    Detect dataset type by checking for key fields.

    Returns:
        "progressive" if embedding and stage_index exist
        "prefix_tuning" if prefix_embedding exists
        "unknown" otherwise
    """
    if len(ds) == 0:
        return "unknown"
    # Check first record for field presence
    first_record = ds[0]
    if "prefix_embedding" in first_record:
        return "prefix_tuning"
    elif "embedding" in first_record and "stage_index" in first_record:
        return "progressive"
    return "unknown"


def load_dataset(dataset_path: str) -> Tuple[Dataset, str]:
    """
    Load dataset and detect its type.

    Returns:
        Tuple of (dataset, dataset_type) where dataset_type is "progressive" or "prefix_tuning"
    """
    ds = Dataset.load_from_disk(dataset_path)
    dataset_type = detect_dataset_type(ds)
    return ds, dataset_type


def load_progressive_dataset(dataset_path: str) -> Dataset:
    """Load progressive checkpoint dataset (deprecated, use load_dataset instead)."""
    ds, _ = load_dataset(dataset_path)
    return ds


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
    dataset_type: str = "progressive",
) -> List[Dict[str, Any]]:
    """
    Filter records by sample_id and/or stage_index.

    Args:
        ds: Dataset to filter
        sample_id: Optional sample_id filter
        stage_index: Optional stage_index filter (ignored for prefix_tuning datasets)
        dataset_type: Type of dataset ("progressive" or "prefix_tuning")
    """
    # Remove columns that may not exist in all dataset types
    columns_to_remove = []
    for col in ["orig_embedding", "initialization_embedding", "initialization_prefix_embedding"]:
        if col in ds.column_names:
            columns_to_remove.append(col)
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)

    rows: List[Dict[str, Any]] = []
    for i in tqdm(range(len(ds)), desc="Filtering records"):
        r = ds[i]
        if sample_id is not None and int(r.get("sample_id", -1)) != int(sample_id):
            continue
        # Only filter by stage_index for progressive datasets
        if dataset_type == "progressive" and stage_index is not None and int(r.get("stage_index", -1)) != int(stage_index):
            continue
        rows.append(r)
    return rows


def collate_stages_by_sample(
    rows: List[Dict[str, Any]],
    dataset_type: str = "progressive",
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group rows by sample_id and sort by stage_index (for progressive) or keep single entry (for prefix_tuning).

    For prefix_tuning datasets, each sample has only one entry, so we create a single-item list.
    """
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("sample_id", -1))
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(r)
    # For progressive datasets, sort by stage_index
    # For prefix_tuning, each sample should have only one entry (no stages)
    if dataset_type == "progressive":
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
        # Filter to valid lengths: effective_lengths are 1-indexed (total sequence length including prefix)
        # indices are 0-indexed, so max valid effective_length is total_seq_len, max index is total_seq_len - 1
        # But we need to be careful: if e = total_seq_len, then idx = total_seq_len - 1 is valid
        # So we allow e in [1, total_seq_len] which gives idx in [0, total_seq_len - 1]
        max_valid_idx = prefix_sums.shape[1] - 1  # 0-indexed max index
        effective_lengths = [e for e in effective_lengths if 1 <= e <= max_valid_idx + 1]
        if not effective_lengths:
            continue

        # Compute per-length means: prefix_sums[:, e-1] / e, then average across lengths.
        # e is 1-indexed effective length, so idx = e - 1 is 0-indexed
        # Ensure all indices are valid: idx in [0, max_valid_idx]
        idx_list = []
        valid_effective_lengths = []
        for e in effective_lengths:
            idx_val = e - 1
            if 0 <= idx_val <= max_valid_idx:
                idx_list.append(idx_val)
                valid_effective_lengths.append(e)

        if not idx_list:
            continue

        idx = torch.tensor(idx_list, dtype=torch.long)
        denom = torch.tensor(valid_effective_lengths, dtype=prefix_sums.dtype).unsqueeze(0)
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
    dataset_type: str = "progressive",
) -> Dict[int, Dict[int, float]]:
    """
    Compute attention mass percent for all stages.

    Uses the longest sequence to compute attention once, then extracts attention mass
    for each stage from the full attention map (due to causal attention mask).

    For prefix_tuning datasets, computes attention across different sequence lengths
    using the same prefix embedding.

    Args:
        model: Language model
        tokenizer: Tokenizer
        stages: List of stage records, sorted by stage_index (progressive) or single entry (prefix_tuning)
        device: Device to run on
        dataset_type: Type of dataset ("progressive" or "prefix_tuning")

    Returns:
        Dictionary mapping seq_len to layer_index to attention_mass_percent
    """
    if not stages:
        return {}

    # Get the stage record (for prefix_tuning, there's only one; for progressive, use the longest)
    if dataset_type == "prefix_tuning":
        stage_record = stages[0]
        # For prefix tuning, extract sequence length from tokenized text
        text = stage_record.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            return {}
        # Tokenize to get actual sequence length
        enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
        max_seq_len = enc["input_ids"].shape[1]
        # Extract prefix embedding
        embedding = stage_record.get("prefix_embedding")
        if embedding is None:
            return {}
        # Get number of virtual tokens
        num_compression_tokens = int(stage_record.get("num_virtual_tokens", 1))
    else:
        # Progressive: find the longest sequence
        longest_stage = max(stages, key=lambda s: int(s.get("stage_seq_len", 0)))
        max_seq_len = int(longest_stage.get("stage_seq_len", -1))
        if max_seq_len < 1:
            return {}
        # Extract compression embeddings
        embedding = longest_stage.get("embedding")
        if embedding is None:
            return {}
        # Get number of compression tokens
        num_compression_tokens = int(longest_stage.get("num_compression_tokens", 1))
        # Get text
        text = longest_stage.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            return {}

    # Convert to tensor
    if isinstance(embedding, list):
        compression_embeddings = torch.tensor(embedding, dtype=torch.float32)
    else:
        compression_embeddings = torch.tensor(embedding, dtype=torch.float32)

    # Get model's hidden size for validation
    model_hidden_size = model.config.hidden_size

    # For prefix tuning, embeddings are stored as PEFT module state and may need special handling
    if dataset_type == "prefix_tuning":
        # Prefix tuning embeddings are stored as PEFT parameters which may have different shapes
        # We need to use PEFT to properly convert them to the format needed for attention computation
        try:
            from peft import PrefixTuningConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("peft is required for prefix tuning visualization. Install it (e.g. `uv add peft`).")

        # Create a PEFT model to properly handle prefix embeddings
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_compression_tokens,
        )
        peft_model = get_peft_model(model, peft_config).to(device)

        # Find the prefix embedding parameter in the PEFT model
        prefix_param_name = None
        for name, param in peft_model.named_parameters():
            if param.requires_grad and param.ndim == 2 and param.shape[0] == num_compression_tokens:
                prefix_param_name = name
                break

        if prefix_param_name is None:
            raise ValueError("Could not find prefix embedding parameter in PEFT model.")

        # Load the saved prefix embedding into the PEFT model parameter
        original_shape = compression_embeddings.shape
        print(f"Original prefix embedding shape: {original_shape}, num_virtual_tokens: {num_compression_tokens}")

        # Reshape to match PEFT parameter shape (which may be different from model hidden_size)
        target_param = dict(peft_model.named_parameters())[prefix_param_name]
        target_shape = target_param.shape
        print(f"PEFT parameter shape: {target_shape}")

        # Reshape the embedding to match the PEFT parameter shape
        if compression_embeddings.shape != target_shape:
            total_elements = compression_embeddings.numel()
            if total_elements == target_param.numel():
                compression_embeddings = compression_embeddings.reshape(target_shape)
                print(f"Reshaped prefix embedding from {original_shape} to {target_shape}")
            else:
                raise ValueError(
                    f"Cannot reshape prefix embedding from {original_shape} (total: {total_elements}) "
                    f"to PEFT parameter shape {target_shape} (total: {target_param.numel()}). "
                    f"Element counts don't match."
                )

        # Set the PEFT parameter
        with torch.no_grad():
            target_param.data = compression_embeddings.to(device).to(target_param.dtype)

        # Now we'll use the PEFT model for forward pass, which will properly handle the prefix embeddings
        use_peft_model = True
    else:
        # Progressive: reshape compression embeddings to correct shape: [num_compression_tokens, hidden_size]
        # Handle different possible shapes the embedding might be stored in
        original_shape = compression_embeddings.shape
        print(
            f"Original embedding shape: {original_shape}, model hidden_size: {model_hidden_size}, num_compression_tokens: {num_compression_tokens}"
        )
        use_peft_model = False

        # Progressive training: handle shape reshaping
        if len(original_shape) == 1:
            # Flattened: reshape to [num_compression_tokens, hidden_size]
            total_elements = compression_embeddings.numel()
            if total_elements % model_hidden_size == 0:
                num_tokens_from_shape = total_elements // model_hidden_size
                compression_embeddings = compression_embeddings.reshape(num_tokens_from_shape, model_hidden_size)
                if num_tokens_from_shape != num_compression_tokens:
                    print(
                        f"Warning: Reshaped embedding from {original_shape} to [{num_tokens_from_shape}, {model_hidden_size}], but expected {num_compression_tokens} tokens. Using {num_tokens_from_shape}."
                    )
                    num_compression_tokens = num_tokens_from_shape
            else:
                raise ValueError(
                    f"Cannot reshape embedding from shape {original_shape} (total elements: {total_elements}) "
                    f"to [num_tokens, hidden_size={model_hidden_size}]. Total elements must be divisible by hidden_size."
                )
        elif len(original_shape) == 2:
            # Should be [num_compression_tokens, hidden_size] or [hidden_size, num_compression_tokens]
            if original_shape[1] == model_hidden_size:
                # Already correct shape [num_compression_tokens, hidden_size]
                if original_shape[0] != num_compression_tokens:
                    print(
                        f"Warning: Embedding has {original_shape[0]} tokens but expected {num_compression_tokens}. Using {original_shape[0]}."
                    )
                    num_compression_tokens = original_shape[0]
            elif original_shape[0] == model_hidden_size:
                # Transpose if it's [hidden_size, num_compression_tokens]
                compression_embeddings = compression_embeddings.transpose(0, 1)
                if compression_embeddings.shape[0] != num_compression_tokens:
                    print(
                        f"Warning: After transpose, embedding has {compression_embeddings.shape[0]} tokens but expected {num_compression_tokens}. Using {compression_embeddings.shape[0]}."
                    )
                    num_compression_tokens = compression_embeddings.shape[0]
            elif original_shape[0] == num_compression_tokens and original_shape[1] != model_hidden_size:
                # [num_compression_tokens, wrong_hidden_size] - model mismatch
                raise ValueError(
                    f"Model hidden size mismatch: embedding has hidden_size={original_shape[1]}, "
                    f"but model has hidden_size={model_hidden_size}. "
                    f"This embedding was created with a different model. "
                    f"Please use the same model checkpoint that was used to create the embeddings."
                )
            elif original_shape[1] == num_compression_tokens and original_shape[0] != model_hidden_size:
                # [wrong_hidden_size, num_compression_tokens] - transpose and check
                compression_embeddings = compression_embeddings.transpose(0, 1)
                if compression_embeddings.shape[1] != model_hidden_size:
                    raise ValueError(
                        f"Model hidden size mismatch: embedding has hidden_size={original_shape[0]}, "
                        f"but model has hidden_size={model_hidden_size}. "
                        f"This embedding was created with a different model. "
                        f"Please use the same model checkpoint that was used to create the embeddings."
                    )
            elif original_shape[0] * original_shape[1] == num_compression_tokens * model_hidden_size:
                # Reshape if dimensions are swapped but total elements match
                compression_embeddings = compression_embeddings.reshape(num_compression_tokens, model_hidden_size)
            else:
                # Try to reshape based on total elements
                total_elements = compression_embeddings.numel()
                if total_elements % model_hidden_size == 0:
                    num_tokens_from_shape = total_elements // model_hidden_size
                    compression_embeddings = compression_embeddings.reshape(num_tokens_from_shape, model_hidden_size)
                    if num_tokens_from_shape != num_compression_tokens:
                        print(
                            f"Warning: Reshaped embedding from {original_shape} to [{num_tokens_from_shape}, {model_hidden_size}], but expected {num_compression_tokens} tokens. Using {num_tokens_from_shape}."
                        )
                        num_compression_tokens = num_tokens_from_shape
                else:
                    raise ValueError(
                        f"Cannot reshape embedding from shape {original_shape} to match model hidden_size={model_hidden_size}. "
                        f"Expected shape [num_compression_tokens={num_compression_tokens}, hidden_size={model_hidden_size}] "
                        f"or total elements divisible by {model_hidden_size}. "
                        f"This may indicate the embedding was created with a different model."
                    )
        elif len(original_shape) == 3:
            # Remove batch dimension if present: [1, num_tokens, hidden_size] or [1, hidden_size, num_tokens]
            if original_shape[0] == 1:
                compression_embeddings = compression_embeddings.squeeze(0)
                # Recursively handle the 2D case
                if compression_embeddings.shape[1] == model_hidden_size:
                    pass  # Already correct
                elif compression_embeddings.shape[0] == model_hidden_size:
                    compression_embeddings = compression_embeddings.transpose(0, 1)
                else:
                    total_elements = compression_embeddings.numel()
                    if total_elements % model_hidden_size == 0:
                        num_tokens_from_shape = total_elements // model_hidden_size
                        compression_embeddings = compression_embeddings.reshape(num_tokens_from_shape, model_hidden_size)
                        if num_tokens_from_shape != num_compression_tokens:
                            print(
                                f"Warning: Reshaped embedding from {original_shape} to [{num_tokens_from_shape}, {model_hidden_size}], but expected {num_compression_tokens} tokens. Using {num_tokens_from_shape}."
                            )
                            num_compression_tokens = num_tokens_from_shape
                    else:
                        raise ValueError(
                            f"Cannot reshape embedding from shape {original_shape} to match model hidden_size={model_hidden_size}."
                        )
            else:
                raise ValueError(f"Unexpected 3D embedding shape: {original_shape}. Expected [1, num_tokens, hidden_size].")
        else:
            raise ValueError(f"Unexpected embedding shape: {original_shape}. Expected 1D, 2D, or 3D tensor.")

        # Final validation for progressive: ensure shape is [num_compression_tokens, hidden_size]
        if compression_embeddings.shape != (num_compression_tokens, model_hidden_size):
            # Check if this might be a model mismatch
            if compression_embeddings.shape[1] != model_hidden_size:
                raise ValueError(
                    f"Embedding hidden size mismatch: embedding has hidden_size={compression_embeddings.shape[1]}, "
                    f"but model has hidden_size={model_hidden_size}. "
                    f"This suggests the embedding was created with a different model. "
                    f"Please use the same model checkpoint that was used to create the embeddings. "
                    f"Embedding shape: {compression_embeddings.shape}, expected: [{num_compression_tokens}, {model_hidden_size}], "
                    f"original shape: {original_shape}."
                )
            else:
                raise ValueError(
                    f"After reshaping, embedding shape is {compression_embeddings.shape}, "
                    f"but expected [{num_compression_tokens}, {model_hidden_size}]. "
                    f"Original shape was {original_shape}."
                )

    # Compute attention once for the longest sequence
    print(f"Computing attention for longest sequence (length={max_seq_len})...")
    if use_peft_model:
        print(f"Using PEFT model for prefix tuning with {num_compression_tokens} virtual tokens")
    else:
        print(f"Compression embeddings shape: {compression_embeddings.shape}, num_compression_tokens: {num_compression_tokens}")

    model_to_use = peft_model if use_peft_model else model
    model_to_use.eval()
    with torch.no_grad():
        # Tokenize text
        enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if use_peft_model:
            # For PEFT, we can use input_ids directly - PEFT will handle prefix embeddings internally
            # Forward pass with attention outputs
            outputs = model_to_use(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        else:
            # Progressive: manually concatenate compression embeddings
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
            outputs = model_to_use(
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
        if dataset_type == "prefix_tuning":
            # For prefix tuning, use sequence lengths from 1 to max_seq_len
            target_seq_lengths = list(range(1, max_seq_len + 1))
        else:
            # Get all unique sequence lengths from stages
            target_seq_lengths = sorted(
                set(int(s.get("stage_seq_len", -1)) for s in stages if int(s.get("stage_seq_len", -1)) > 0)
            )

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
        help="Path to progressive_prefixes or prefix_tuning_prefixes dataset",
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
        help="Filter out samples whose max sequence length is < this value.",
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
        if (
            "artifacts/experiments" in dataset_path
            or "artifacts/experiments_progressive" in dataset_path
            or "artifacts/experiments_prefix_tuning" in dataset_path
        ):
            exp_dir = os.path.dirname(dataset_path)
            output_dir = os.path.join(exp_dir, "attention_visualizations")
        else:
            output_dir = "attention_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and detect type
    print(f"Loading dataset from: {args.dataset_path}")
    ds, dataset_type = load_dataset(args.dataset_path)
    print(f"Detected dataset type: {dataset_type}")

    if dataset_type == "unknown":
        raise ValueError("Could not detect dataset type. Expected 'progressive' or 'prefix_tuning' dataset.")

    # Filter records
    rows = filter_records(ds, sample_id=args.sample_id, dataset_type=dataset_type)
    if not rows:
        raise ValueError("No records found with given filters.")

    # Group by sample
    by_sid = collate_stages_by_sample(rows, dataset_type=dataset_type)

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
            if dataset_type == "prefix_tuning":
                # For prefix tuning, get sequence length from tokenized text
                stage_record = stages[0]
                text = stage_record.get("text", "")
                if not isinstance(text, str) or text.strip() == "":
                    continue
                enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
                max_len = enc["input_ids"].shape[1]
            else:
                # Progressive: use stage_seq_len
                max_len = max((int(s.get("stage_seq_len", -1)) for s in stages), default=-1)
            if max_len >= args.min_seq_length:
                eligible_by_sid[_sid] = stages
                per_sample_max.append(max_len)
        if not per_sample_max:
            raise ValueError(
                f"No samples with max sequence length >= {args.min_seq_length} found. "
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
            stage_count = len(stages)
            stage_label = "stages" if dataset_type == "progressive" else "entry"
            print(f"\nProcessing sample {sample_id} with {stage_count} {stage_label}...")
            results = compute_attention_mass_for_stages(
                model=model,
                tokenizer=tokenizer,
                stages=stages,
                device=device,
                attention_block_size=args.attention_block_size,
                target_seq_lengths_override=target_seq_lengths_override,
                dataset_type=dataset_type,
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
            if dataset_type == "prefix_tuning":
                # For prefix tuning, get sequence length from tokenized text
                stage_record = stages[0]
                text = stage_record.get("text", "")
                if not isinstance(text, str) or text.strip() == "":
                    print(f"\nSkipping sample {sample_id}: empty text")
                    continue
                enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
                max_len = enc["input_ids"].shape[1]
            else:
                # Progressive: use stage_seq_len
                max_len = max((int(s.get("stage_seq_len", -1)) for s in stages), default=-1)
            if max_len < args.min_seq_length:
                print(f"\nSkipping sample {sample_id}: max sequence length={max_len} < min_seq_length={args.min_seq_length}")
                continue
            stage_count = len(stages)
            stage_label = "stages" if dataset_type == "progressive" else "entry"
            print(f"\nProcessing sample {sample_id} with {stage_count} {stage_label}...")
            results = compute_attention_mass_for_stages(
                model=model,
                tokenizer=tokenizer,
                stages=stages,
                device=device,
                attention_block_size=args.attention_block_size,
                dataset_type=dataset_type,
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
