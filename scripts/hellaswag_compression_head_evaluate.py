"""Evaluate compression head model on HellaSwag benchmark.

This script:
1. Loads a LlamaForCausalLMCompressionHead model from checkpoint
2. Evaluates by computing PPL for all endings with and without compression
3. Dumps evaluation results
"""

import argparse
import json
import math
import os
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from compression_horizon.models.llama_compression_head import LlamaForCausalLMCompressionHead
from compression_horizon.utils.launch import get_device, set_launch_seed


def estimate_token_perplexity(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute perplexity from logits, labels, and attention mask."""
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


@torch.no_grad()
def compute_ppl_baseline_batch(
    model: LlamaForCausalLMCompressionHead,
    tokenizer: AutoTokenizer,
    contexts: list[str],
    endings: list[str],
    device: Optional[torch.device] = None,
) -> list[float]:
    """Compute perplexity of context + ending pairs without compression (baseline).

    Args:
        model: The language model with compression head
        tokenizer: Tokenizer
        contexts: List of context texts
        endings: List of ending texts (same length as contexts)
        device: Device to use

    Returns:
        List of perplexity scores
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if len(contexts) == 0:
        return []

    # Combine contexts and endings
    full_texts = [ctx + end for ctx, end in zip(contexts, endings)]

    # Tokenize with padding
    encoded = tokenizer(full_texts, truncation=True, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Forward pass without compression (no prefix_lengths)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Compute PPL for each sample
    ppls = []
    for i in range(len(contexts)):
        seq_len = attention_mask[i].sum().item()
        sample_logits = outputs.logits[i : i + 1, :seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)

    return ppls


@torch.no_grad()
def compute_ppl_with_compression_batch(
    model: LlamaForCausalLMCompressionHead,
    tokenizer: AutoTokenizer,
    contexts: list[str],
    endings: list[str],
    prefix_lengths: Optional[torch.LongTensor] = None,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Compute perplexity of context + ending pairs using compression head.

    Args:
        model: The language model with compression head
        tokenizer: Tokenizer
        contexts: List of context texts
        endings: List of ending texts (same length as contexts)
        prefix_lengths: Optional tensor [B] with prefix lengths to compress. If None, uses context length.
        device: Device to use

    Returns:
        List of perplexity scores
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if len(contexts) == 0:
        return []

    # Tokenize contexts to determine prefix lengths
    encoded_contexts = tokenizer(contexts, truncation=True, padding=True, return_tensors="pt")
    context_input_ids = encoded_contexts["input_ids"].to(device)  # [batch_size, seq_len]
    context_attention_mask = encoded_contexts["attention_mask"].to(device)  # [batch_size, seq_len]

    # Determine prefix lengths (number of tokens in context to compress)
    if prefix_lengths is None:
        prefix_lengths = context_attention_mask.sum(dim=1).to(torch.long).clamp_min(1)  # [B]

    # First forward pass: get compression embeddings for contexts
    outputs1 = model(
        input_ids=context_input_ids,
        attention_mask=context_attention_mask,
        prefix_lengths=prefix_lengths,
        output_hidden_states=True,
        return_dict=True,
    )

    if outputs1.compression_embeds is None:
        raise RuntimeError("Model did not return compression embeddings")

    compression_embeds = outputs1.compression_embeds  # [B, 1, H]

    # Combine contexts and endings for full sequences
    full_texts = [ctx + end for ctx, end in zip(contexts, endings)]
    encoded_full = tokenizer(full_texts, truncation=True, padding=True, return_tensors="pt")
    full_input_ids = encoded_full["input_ids"].to(device)  # [batch_size, seq_len]
    full_attention_mask = encoded_full["attention_mask"].to(device)  # [batch_size, seq_len]

    # Get token embeddings for full sequences
    token_embeddings = model.get_input_embeddings()(full_input_ids)  # [batch_size, seq_len, hidden]

    # Build compressed inputs: [compression_embed] + suffix tokens
    batch_size, seq_len, hidden = token_embeddings.shape
    lengths = full_attention_mask.sum(dim=1).to(torch.long).clamp_min(1)  # [B]
    p = prefix_lengths.to(device=device).to(torch.long)
    p = torch.clamp(p, min=0)
    max_prefix = (lengths - 1).clamp_min(0)
    p = torch.minimum(p, max_prefix)  # [B]

    # Build fixed-length outputs: [compression] + up to T suffix tokens
    out_len = 1 + seq_len
    inputs_embeds_new = torch.zeros((batch_size, out_len, hidden), device=device, dtype=token_embeddings.dtype)
    attention_mask_new = torch.zeros((batch_size, out_len), device=device, dtype=full_attention_mask.dtype)
    labels_new = torch.full((batch_size, out_len), fill_value=-100, device=device, dtype=full_input_ids.dtype)

    # Place compression embedding at position 0
    inputs_embeds_new[:, 0:1, :] = compression_embeds
    attention_mask_new[:, 0] = 1

    # Gather suffix tokens starting at p for each batch item
    ar = torch.arange(seq_len, device=device, dtype=torch.long)  # [T]
    src_idx = p.unsqueeze(1) + ar.unsqueeze(0)  # [B, T]
    valid = src_idx < lengths.unsqueeze(1)  # [B, T]
    src_idx_safe = torch.clamp(src_idx, max=seq_len - 1)

    gathered_embeds = token_embeddings.gather(1, src_idx_safe.unsqueeze(-1).expand(-1, -1, hidden))
    gathered_ids = full_input_ids.gather(1, src_idx_safe)

    if valid.dtype != torch.bool:
        valid = valid.to(torch.bool)

    inputs_embeds_new[:, 1:, :] = gathered_embeds * valid.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    attention_mask_new[:, 1:] = valid.to(dtype=full_attention_mask.dtype)
    labels_new[:, 1:] = torch.where(valid, gathered_ids, torch.full_like(gathered_ids, -100))

    # Second forward pass: evaluate with compression
    outputs2 = model(inputs_embeds=inputs_embeds_new, attention_mask=attention_mask_new)

    # Compute PPL for each sample (only on suffix tokens, excluding compression position)
    ppls = []
    for i in range(len(contexts)):
        seq_len_full = full_attention_mask[i].sum().item()
        p_i = int(p[i].item())
        # Logits start at position 1 (after compression token), align with full_input_ids
        # We want to evaluate on the suffix part
        suffix_len = max(seq_len_full - p_i, 1)
        sample_logits = outputs2.logits[i : i + 1, 1 : 1 + suffix_len]  # Skip compression position
        sample_input_ids = full_input_ids[i : i + 1, p_i : p_i + suffix_len]
        sample_attention = torch.ones((1, suffix_len), device=device, dtype=full_attention_mask.dtype)
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)

    return ppls


def main():
    parser = argparse.ArgumentParser(description="Evaluate compression head model on HellaSwag benchmark")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Base model checkpoint to use (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--compression_head_checkpoint",
        type=str,
        required=True,
        help="Path to compression_head.pt checkpoint file",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=100,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/hellaswag_compression_head_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--no_evaluate_baseline",
        action="store_false",
        dest="evaluate_baseline",
        default=True,
        help="Skip baseline evaluation. Default: evaluate baseline.",
    )
    parser.add_argument(
        "--no_evaluate_compressed",
        action="store_false",
        dest="evaluate_compressed",
        default=True,
        help="Skip compressed evaluation. Default: evaluate compressed.",
    )

    args = parser.parse_args()

    # Set random seed
    set_launch_seed(args.random_seed)

    # Resolve dtype
    def _resolve_torch_dtype(dtype_str: str):
        s = (dtype_str or "").lower()
        if s in {"auto"}:
            return "auto"
        if s in {"float32", "fp32"}:
            return torch.float32
        if s in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if s in {"float16", "fp16"}:
            return torch.float16
        return torch.float32

    torch_dtype = _resolve_torch_dtype(args.dtype)
    device = get_device()

    # Load base model
    print(f"Loading base model from {args.model_checkpoint}...")
    model = LlamaForCausalLMCompressionHead.from_pretrained(
        args.model_checkpoint, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
    )

    # Load compression head checkpoint
    print(f"Loading compression head from {args.compression_head_checkpoint}...")
    checkpoint = torch.load(args.compression_head_checkpoint, map_location=device)
    if "compression_head" in checkpoint and checkpoint["compression_head"] is not None:
        model.compression_head.load_state_dict(checkpoint["compression_head"])
        print("Loaded compression_head state dict")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load HellaSwag dataset
    print("Loading HellaSwag dataset...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    if args.limit_samples:
        dataset = dataset.select(range(args.limit_samples))
    print(f"Evaluating on {len(dataset)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluation results
    results = []
    correct_predictions_compressed = 0
    correct_predictions_baseline = 0
    total_predictions = 0

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start_idx, end_idx)]
        actual_batch_size = len(batch_items)

        # Extract batch data
        batch_contexts = [item["ctx"] for item in batch_items]
        batch_endings_list = [item["endings"] for item in batch_items]
        batch_labels = [int(item["label"]) for item in batch_items]

        # Compute baseline PPL for all endings (if enabled)
        batch_baseline_ppls = []
        if args.evaluate_baseline:
            for sample_idx in range(actual_batch_size):
                sample_ppls = []
                contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
                try:
                    ppls = compute_ppl_baseline_batch(
                        model=model,
                        tokenizer=tokenizer,
                        contexts=contexts_for_batch,
                        endings=batch_endings_list[sample_idx],
                        device=device,
                    )
                    sample_ppls = ppls
                except Exception as e:
                    print(f"Error computing baseline PPL for sample {start_idx + sample_idx}: {e}")
                    sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                batch_baseline_ppls.append(sample_ppls)
        else:
            batch_baseline_ppls = [[float("inf")] * len(ends) for ends in batch_endings_list]

        # Compute compressed PPL for all endings (if enabled)
        batch_compressed_ppls = []
        if args.evaluate_compressed:
            for sample_idx in range(actual_batch_size):
                sample_ppls = []
                try:
                    # Prepare contexts and endings for this sample
                    contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
                    # Always compress full prefix (prefix_lengths=None uses full context length)
                    ppls = compute_ppl_with_compression_batch(
                        model=model,
                        tokenizer=tokenizer,
                        contexts=contexts_for_batch,
                        endings=batch_endings_list[sample_idx],
                        prefix_lengths=None,
                        device=device,
                    )
                    sample_ppls = ppls
                except Exception as e:
                    print(f"Error computing compressed PPL for sample {start_idx + sample_idx}: {e}")
                    sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                batch_compressed_ppls.append(sample_ppls)
        else:
            batch_compressed_ppls = [[float("inf")] * len(ends) for ends in batch_endings_list]

        # Process results for this batch
        for sample_idx in range(actual_batch_size):
            idx = start_idx + sample_idx
            context = batch_contexts[sample_idx]
            endings = batch_endings_list[sample_idx]
            label = batch_labels[sample_idx]

            baseline_ppls = batch_baseline_ppls[sample_idx]
            baseline_predicted_label = None
            baseline_is_correct = None
            if args.evaluate_baseline and baseline_ppls and all(not math.isinf(p) for p in baseline_ppls):
                baseline_predicted_label = int(torch.tensor(baseline_ppls).argmin().item())
                baseline_is_correct = baseline_predicted_label == label
                if baseline_is_correct:
                    correct_predictions_baseline += 1

            compressed_ppls = batch_compressed_ppls[sample_idx]
            compressed_predicted_label = None
            compressed_is_correct = None
            if args.evaluate_compressed and compressed_ppls and all(not math.isinf(p) for p in compressed_ppls):
                compressed_predicted_label = int(torch.tensor(compressed_ppls).argmin().item())
                compressed_is_correct = compressed_predicted_label == label
                if compressed_is_correct:
                    correct_predictions_compressed += 1

            total_predictions += 1

            # Store result
            result = {
                "sample_id": idx,
                "context": context,
                "endings": endings,
                "label": label,
                "baseline": {
                    "predicted_label": baseline_predicted_label,
                    "is_correct": baseline_is_correct,
                    "ppls": baseline_ppls,
                },
                "compressed": {
                    "predicted_label": compressed_predicted_label,
                    "is_correct": compressed_is_correct,
                    "ppls": compressed_ppls,
                },
            }
            results.append(result)

        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
            baseline_accuracy = (
                correct_predictions_baseline / total_predictions if total_predictions > 0 and args.evaluate_baseline else 0.0
            )
            compressed_accuracy = (
                correct_predictions_compressed / total_predictions
                if total_predictions > 0 and args.evaluate_compressed
                else 0.0
            )
            print(
                f"Progress: {total_predictions}/{len(dataset)}, Baseline Accuracy: {baseline_accuracy:.4f}, "
                f"Compressed Accuracy: {compressed_accuracy:.4f}"
            )

    # Compute final accuracies
    baseline_accuracy = (
        correct_predictions_baseline / total_predictions if total_predictions > 0 and args.evaluate_baseline else 0.0
    )
    compressed_accuracy = (
        correct_predictions_compressed / total_predictions if total_predictions > 0 and args.evaluate_compressed else 0.0
    )

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "baseline": {
                    "accuracy": baseline_accuracy,
                    "correct_predictions": correct_predictions_baseline,
                    "total_predictions": total_predictions if args.evaluate_baseline else 0,
                },
                "compressed": {
                    "accuracy": compressed_accuracy,
                    "correct_predictions": correct_predictions_compressed,
                    "total_predictions": total_predictions if args.evaluate_compressed else 0,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total samples: {total_predictions}")
    if args.evaluate_baseline:
        print("\nBaseline (without compression):")
        print(f"  Correct predictions: {correct_predictions_baseline}")
        print(f"  Accuracy: {baseline_accuracy:.4f}")
    if args.evaluate_compressed:
        print("\nCompressed (with compression head):")
        print(f"  Correct predictions: {correct_predictions_compressed}")
        print(f"  Accuracy: {compressed_accuracy:.4f}")
    if args.evaluate_baseline and args.evaluate_compressed:
        print(f"\nDifference: {compressed_accuracy - baseline_accuracy:+.4f}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
