"""Evaluate compression tokens on HellaSwag benchmark.

This script:
1. Downloads HellaSwag dataset
2. Compresses the context (prefix) for each item
3. Evaluates by computing PPL for all endings and choosing the one with lowest PPL
4. Dumps evaluation results
"""

import argparse
import json
import math
import os
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from compression_horizon.train.loss import compute_hybrid_cross_entropy_and_alignment_loss
from compression_horizon.utils.launch import freeze_model_parameters, get_device, resolve_torch_dtype, set_launch_seed
from compression_horizon.utils.tokens import count_text_tokens, count_text_characters
from compression_horizon.metric import estimate_token_perplexity


def compress_prefixes_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    num_compression_tokens: int = 1,
    max_steps: int = 1000,
    learning_rate: float = 1e-2,
    loss_type: str = "cross_entropy",
    hybrid_alpha: Optional[float] = None,
    num_alignment_layers: int = 0,
    inverted_alignment: bool = False,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[dict[str, torch.Tensor | float]]:
    """Compress multiple text prefixes into compression tokens (batched).

    Args:
        model: The language model (frozen)
        tokenizer: Tokenizer
        texts: List of text prefixes to compress
        num_compression_tokens: Number of compression tokens
        max_steps: Maximum optimization steps
        learning_rate: Learning rate for optimization
        device: Device to use

    Returns:
        List of compression token embeddings, each [num_compression_tokens, hidden_size]
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    freeze_model_parameters(model)

    batch_size = len(texts)
    if batch_size == 0:
        return []

    loss_type = (loss_type or "cross_entropy").lower()
    use_alignment = hybrid_alpha is not None and loss_type != "cross_entropy"

    # Tokenize all texts with padding
    # max_length=training_args.max_sequence_length
    encoded = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    hidden_size = model.config.hidden_size

    # Get token embeddings to determine dtype
    with torch.no_grad():
        token_embeddings = model.model.embed_tokens(input_ids)  # [batch_size, seq_len, hidden]

    # Get dtype from model embeddings
    embedding_dtype = token_embeddings.dtype
    compression_dtype = embedding_dtype

    # Initialize compression tokens for all samples
    compression_token_embeddings = torch.nn.Parameter(
        torch.randn([batch_size, num_compression_tokens, hidden_size], dtype=compression_dtype, device=device) * 0.02
    )  # [batch_size, num_compression_tokens, hidden]

    # Optimizer
    optimizer = AdamW([compression_token_embeddings], lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    model.train()

    # Training loop
    for _ in range(max_steps):
        optimizer.zero_grad()

        # Concatenate compression tokens with input embeddings
        # Expand compression tokens to match sequence lengths
        united_token_embeddings_list = []
        united_attention_mask_list = []
        labels_list = []
        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            # Get embeddings for this sample (only non-padded tokens)
            sample_token_embeddings = token_embeddings[i : i + 1, :seq_len]  # [1, seq_len, hidden]
            sample_attention_mask = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
            sample_compression_token_embeddings = compression_token_embeddings[i : i + 1]  # [1, num_compression_tokens, hidden]
            # Concatenate
            united_token_embeddings = torch.cat([sample_compression_token_embeddings, sample_token_embeddings], dim=1)  # [1, mem+seq, hidden]
            united_attention_mask = torch.cat(
                [
                    torch.ones((1, num_compression_tokens), dtype=sample_attention_mask.dtype, device=device),
                    sample_attention_mask,
                ],
                dim=1,
            )  # [1, mem+seq]
            united_token_embeddings_list.append(united_token_embeddings)
            united_attention_mask_list.append(united_attention_mask)
            # Labels for this sample
            sample_input_ids = input_ids[i : i + 1, :seq_len]  # [1, seq_len]
            sample_labels = sample_input_ids.clone()
            sample_labels[sample_attention_mask == 0] = -100
            labels_list.append(sample_labels)

        # Pad to maximum length and gather batches
        max_len = max(item.shape[1] for item in united_token_embeddings_list)
        batch_embeddings = []
        batch_attention = []
        batch_labels = []
        for i in range(batch_size):
            united_token_embeddings = united_token_embeddings_list[i]  # [1, mem+seq, hidden]
            united_attention_mask = united_attention_mask_list[i]  # [1, mem+seq]
            labels = labels_list[i]  # [1, seq_len]
            current_len = united_token_embeddings.shape[1]
            if current_len < max_len:
                pad_len = max_len - current_len
                united_token_embeddings = torch.cat(
                    [
                        united_token_embeddings,
                        torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device)
                    ],
                    dim=1,
                )  # [1, max_len, hidden]
                united_attention_mask = torch.cat(
                    [
                        united_attention_mask,
                        torch.zeros(1, pad_len, dtype=united_attention_mask.dtype, device=device),
                    ],
                    dim=1,
                )  # [1, max_len]
                labels = torch.cat(
                    [labels, torch.full((1, pad_len), -100, dtype=labels.dtype, device=device)],
                    dim=1,
                )  # [1, max_len - mem]
            batch_embeddings.append(united_token_embeddings)
            batch_attention.append(united_attention_mask)
            batch_labels.append(labels)
        batch_embeddings = torch.cat(batch_embeddings, dim=0)  # [batch_size, max_len, hidden]
        batch_attention = torch.cat(batch_attention, dim=0)  # [batch_size, max_len]
        batch_labels = torch.cat(batch_labels, dim=0)  # [batch_size, max_len - mem]

        # Forward pass without compression tokens and gradient capturing
        target_outputs = None
        if use_alignment:
            with torch.no_grad():
                target_outputs = model(
                    inputs_embeds=token_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        # Forward pass with compression tokens and gradient capturing
        compression_outputs = model(
            inputs_embeds=batch_embeddings,
            attention_mask=batch_attention,
            output_hidden_states=use_alignment,
        )

        # Compute loss per sample
        total_loss = 0.0
        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            sample_logits = compression_outputs.logits[i : i + 1, : num_compression_tokens + seq_len]
            sample_input_ids = input_ids[i : i + 1, :seq_len]
            sample_attention_mask = attention_mask[i : i + 1, :seq_len]
            if use_alignment:
                assert target_outputs is not None
                sample_compression_hidden_states = tuple(
                    hs_layer[i : i + 1, : num_compression_tokens + seq_len] for hs_layer in compression_outputs.hidden_states
                )
                sample_target_hidden_states = tuple(hs_layer[i : i + 1, :seq_len] for hs_layer in target_outputs.hidden_states)
                sample_loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                    logits=sample_logits,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    num_prefix_tokens=num_compression_tokens,
                    target_hidden_states=sample_target_hidden_states,
                    compression_hidden_states=sample_compression_hidden_states,
                    num_alignment_layers=num_alignment_layers,
                    inverted_alignment=inverted_alignment,
                    loss_type=loss_type,
                    hybrid_alpha=hybrid_alpha,
                )
            else:
                sample_loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                    logits=sample_logits,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    num_prefix_tokens=num_compression_tokens,
                    num_alignment_layers=num_alignment_layers,
                    inverted_alignment=inverted_alignment,
                    loss_type=loss_type,
                    hybrid_alpha=hybrid_alpha,
                )
            total_loss = total_loss + sample_loss
        loss = total_loss / batch_size
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    model.eval()
    # Return compression tokens (remove batch dimension for each)
    convergences = calculate_convergence(model, batch_embeddings, batch_attention, batch_labels, num_compression_tokens)
    print("Batch convergences:", convergences)
    return [
        {
            "compression_embedding": compression_token_embeddings[i].detach(),
            "convergence": convergences[i],
        }
        for i in range(batch_size)
    ]


@torch.no_grad()
def calculate_convergence(
    model: AutoModelForCausalLM, batch_embeddings: torch.Tensor, batch_attention: torch.Tensor, batch_labels: torch.Tensor, num_compression_tokens: int
) -> list[float]:
    outputs = model(
        inputs_embeds=batch_embeddings,
        attention_mask=batch_attention,
    )
    batch_size = batch_embeddings.shape[0]
    convergences = []
    for i in range(batch_size):
        seq_len = batch_attention[i].sum().item()  # Considering compression tokens
        sample_logits = outputs.logits[i : i + 1, : seq_len]  # [1, mem+sequence, vocabulary]
        sample_predicted_tokens = sample_logits[:, num_compression_tokens - 1 : -1].argmax(dim=-1).squeeze(dim=0)  # [sequence]
        sample_labels = batch_labels[i, : seq_len - num_compression_tokens].clone()  # [sequence]
        # print(sample_predicted_tokens, sample_labels)
        convergence = torch.mean((sample_predicted_tokens == sample_labels).to(torch.float)).item()
        convergences.append(convergence)
    return convergences


@torch.no_grad()
def compute_ppl_baseline_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contexts: list[str],
    endings: list[str],
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute perplexity of context + ending pairs without compression tokens (batched).

    Args:
        model: The language model
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
    full_texts = [f"{ctx} {end}" for ctx, end in zip(contexts, endings)]

    # Tokenize with padding
    encoded = tokenizer(full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Compute PPL for each sample
    ppls = []
    for i in range(len(full_texts)):
        seq_len = attention_mask[i].sum().item()
        sample_logits = outputs.logits[i : i + 1, :seq_len]  # [1, seq_len, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :seq_len]  # [1, seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)
    return ppls


@torch.no_grad()
def compute_ppl_with_compression_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_token_embeddings: list[torch.Tensor],
    contexts: list[str],
    endings: list[str],
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute perplexity of context + ending pairs using compression tokens (batched).

    Args:
        model: The language model
        tokenizer: Tokenizer
        compression_tokens_list: List of compression token embeddings, each [num_compression_tokens, hidden_size]
        contexts: List of context texts
        endings: List of ending texts (same length as contexts and compression_tokens_list)
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
    full_texts = [f"{ctx} {end}" for ctx, end in zip(contexts, endings)]

    # Tokenize with padding
    encoded = tokenizer(full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Get token embeddings for all texts
    token_embeddings = model.model.embed_tokens(input_ids)  # [batch_size, seq_len, hidden]

    # Prepare batched inputs with compression tokens
    united_token_embeddings_list = []
    united_attention_mask_list = []
    num_compression_tokens = compression_token_embeddings[0].shape[0]
    for i in range(len(full_texts)):
        seq_len = attention_mask[i].sum().item()
        sample_token_embeddings = token_embeddings[i : i + 1, :seq_len]  # [1, seq_len, hidden]
        sample_attention_mask = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
        sample_compression_token_embeddings = (
            compression_token_embeddings[i].unsqueeze(0).to(token_embeddings.dtype)
        )  # [1, num_compression_tokens, hidden]
        # Concatenate
        united_token_embeddings = torch.cat([sample_compression_token_embeddings, sample_token_embeddings], dim=1)  # [1, mem+seq, hidden]
        united_attention_mask = torch.cat(
            [torch.ones((1, num_compression_tokens), dtype=sample_attention_mask.dtype, device=device), sample_attention_mask], dim=1
        )  # [1, mem+seq]
        united_token_embeddings_list.append(united_token_embeddings)
        united_attention_mask_list.append(united_attention_mask)

    # Pad to maximum length and gather batches
    max_len = max(item.shape[1] for item in united_token_embeddings_list)
    batch_embeddings = []
    batch_attention = []
    for i in range(len(full_texts)):
        united_token_embeddings = united_token_embeddings_list[i]  # [1, mem+seq, hidden]
        united_attention_mask = united_attention_mask_list[i]  # [1, mem+seq]
        current_len = united_token_embeddings.shape[1]
        if current_len < max_len:
            pad_len = max_len - current_len
            hidden_size = united_token_embeddings.shape[2]
            united_token_embeddings = torch.cat(
                [
                    united_token_embeddings,
                    torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device)
                ],
                dim=1,
            )  # [1, max_len, hidden]
            united_attention_mask = torch.cat(
                [
                    united_attention_mask,
                    torch.zeros(1, pad_len, dtype=united_attention_mask.dtype, device=device),
                ],
                dim=1,
            )  # [1, max_len]
        batch_embeddings.append(united_token_embeddings)
        batch_attention.append(united_attention_mask)
    batch_embeddings = torch.cat(batch_embeddings, dim=0)  # [batch_size, max_len, hidden]
    batch_attention = torch.cat(batch_attention, dim=0)  # [batch_size, max_len]

    # Forward pass
    outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)
    # Compute PPL for each sample
    ppls = []
    for i in range(len(full_texts)):
        seq_len = attention_mask[i].sum().item()
        sample_logits = outputs.logits[i : i + 1, num_compression_tokens - 1 : num_compression_tokens - 1 + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]  # [1, seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)
    return ppls


def main():
    parser = argparse.ArgumentParser(description="Evaluate compression tokens on HellaSwag benchmark")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model checkpoint to use",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=100,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--num_compression_tokens",
        type=int,
        default=1,
        help="Number of compression tokens",
    )
    parser.add_argument(
        "--max_optimization_steps",
        type=int,
        default=1000,
        help="Maximum optimization steps for compression",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for compression optimization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for compression and evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "l2", "l1", "cosine"],
        help="Loss type for optimization. Use cross_entropy for CE-only; set hybrid_alpha to add activation alignment.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=0,
        help="Number of layers to align (0 = all layers). Used only when hybrid_alpha is set and loss_type != cross_entropy.",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="If set and loss_type != cross_entropy, adds hybrid_alpha * alignment_loss to CE loss.",
    )
    parser.add_argument(
        "--inverted_alignment",
        action="store_true",
        default=False,
        help="If set, aligns the last num_alignment_layers instead of the first.",
    )
    parser.add_argument(
        "--no_bos_token",
        action="store_true",
        default=False,
        help="Disable BOS token insertion during tokenization.",
    )
    parser.add_argument(
        "—-only_full_convergence",
        action="store_true",
        default=False,
        help="Only those examples for which full compression was achieved should be included in the results.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/hellaswag_evaluation",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Set random seed
    set_launch_seed(args.random_seed)
    # Resolve dtype
    torch_dtype = resolve_torch_dtype(args.dtype)
    device = get_device()
    # Load model and tokenizer
    print(f"Loading model from {args.model_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, dtype=torch_dtype)
    print("Loaded model dtype:", next(model.parameters()).dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    if args.no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = False
    # Add special tokens if no_bos_token is False or no_bos_token is True and add_bos_token is supported
    add_special_tokens = not (args.no_bos_token and not add_bos_supported)

    # Load HellaSwag dataset
    print("Loading HellaSwag dataset...")
    # ind - dataset ID
    # activity_label - specifies the subject areas for sentence completion evaluation
    # ctx - full context
    # endings - a list of 4 endings
    # label - correct label 0, 1, 2 or 3
    # split_type - indomain if the activity label is seen during training, else zeroshot
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    if args.limit_samples:
        dataset = dataset.select(range(args.limit_samples))
    print(f"Evaluating HellaSwag benchmark on {len(dataset)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluation results
    results = []
    correct_predictions_compressed = 0
    correct_predictions_baseline = 0
    total_predictions = 0
    total_tokens = 0
    total_characters = 0
    correct_tokens_baseline = 0
    correct_tokens_compressed = 0
    correct_characters_baseline = 0
    correct_characters_compressed = 0

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

        # Compute baseline PPL for all endings (batched)
        batch_baseline_ppls = []
        for sample_idx in range(actual_batch_size):
            sample_ppls = []
            contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
            try:
                # List of baseline (common model) sample perplexities for each pair context -> ending
                ppls = compute_ppl_baseline_batch(
                    model=model,
                    tokenizer=tokenizer,
                    contexts=contexts_for_batch,
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                sample_ppls = ppls
            except Exception as e:
                print(f"Error computing baseline PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_baseline_ppls.append(sample_ppls)

        # Compress contexts (batched)
        batch_compression_token_embeddings = None
        try:
            # List[num_compression_tokens, hidden]
            batch_compression_token_embeddings = compress_prefixes_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch_contexts,
                num_compression_tokens=args.num_compression_tokens,
                max_steps=args.max_optimization_steps,
                learning_rate=args.learning_rate,
                loss_type=args.loss_type,
                hybrid_alpha=args.hybrid_alpha,
                num_alignment_layers=args.num_alignment_layers,
                inverted_alignment=args.inverted_alignment,
                device=device,
                add_special_tokens=add_special_tokens,
            )
            batch_compression_token_embeddings = [
                item["compression_embedding"] if item["convergence"] >= 1 else "ignore" for item in batch_compression_token_embeddings
            ]
        except Exception as e:
            print(f"Error compressing batch {batch_idx}: {e}")
            batch_compression_token_embeddings = [None] * actual_batch_size

        # Compute compressed PPL for all endings (batched)
        batch_compressed_ppls = []
        for sample_idx in range(actual_batch_size):
            if batch_compression_token_embeddings[sample_idx] is None:
                batch_compressed_ppls.append([float("inf")] * len(batch_endings_list[sample_idx]))
                continue
            elif batch_compression_token_embeddings[sample_idx] == "ignore":
                batch_compressed_ppls.append("ignore")
                continue
            sample_ppls = []
            contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
            try:
                # Prepare contexts and endings for this sample
                batch_sample_compression_token_embeddings = [batch_compression_token_embeddings[sample_idx]] * len(batch_endings_list[sample_idx])
                # List of compression (compression prefix tuned) sample perplexities for each pair context -> ending
                ppls = compute_ppl_with_compression_batch(
                    model=model,
                    tokenizer=tokenizer,
                    compression_token_embeddings=batch_sample_compression_token_embeddings,
                    contexts=contexts_for_batch,
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                sample_ppls = ppls
            except Exception as e:
                print(f"Error computing compressed PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_compressed_ppls.append(sample_ppls)

        # Process results for this batch
        for sample_idx in range(actual_batch_size):
            idx = start_idx + sample_idx
            context = batch_contexts[sample_idx]
            endings = batch_endings_list[sample_idx]
            label = batch_labels[sample_idx]

            baseline_ppls = batch_baseline_ppls[sample_idx]
            baseline_predicted_label = int(torch.tensor(baseline_ppls).argmin().item())
            baseline_is_correct = baseline_predicted_label == label
            if baseline_is_correct:
                correct_predictions_baseline += 1

            compressed_ppls = batch_compressed_ppls[sample_idx]
            if compressed_ppls == "ignore":
                compressed_is_correct = False
                compressed_predicted_label = "ignored_due_convergence"
            else:
                compressed_predicted_label = int(torch.tensor(compressed_ppls).argmin().item())
                compressed_is_correct = compressed_predicted_label == label
                if compressed_is_correct:
                    correct_predictions_compressed += 1

            total_predictions += 1
            token_count = None
            char_count = None
            if 0 <= label < len(endings):
                correct_ending = endings[label]
                full_text = f"{context} {correct_ending}"
                token_count = count_text_tokens(tokenizer, full_text, add_special_tokens=add_special_tokens)
                char_count = count_text_characters(full_text)
                total_tokens += token_count
                total_characters += char_count
                if baseline_is_correct:
                    correct_tokens_baseline += token_count
                    correct_characters_baseline += char_count
                if compressed_is_correct:
                    correct_tokens_compressed += token_count
                    correct_characters_compressed += char_count

            # Store result
            result = {
                "sample_id": idx,
                "context": context,
                "endings": endings,
                "label": label,
                "lengths": {
                    "tokens": token_count,
                    "characters": char_count,
                },
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
            baseline_accuracy = correct_predictions_baseline / total_predictions if total_predictions > 0 else 0.0
            compressed_accuracy = correct_predictions_compressed / total_predictions if total_predictions > 0 else 0.0
            print(
                f"Progress: {total_predictions}/{len(dataset)}, Baseline Accuracy: {baseline_accuracy:.4f}, "
                f"Compressed Accuracy: {compressed_accuracy:.4f}"
            )

    # Compute final accuracies
    baseline_accuracy = correct_predictions_baseline / total_predictions if total_predictions > 0 else 0.0
    compressed_accuracy = correct_predictions_compressed / total_predictions if total_predictions > 0 else 0.0
    baseline_token_accuracy = correct_tokens_baseline / total_tokens if total_tokens > 0 else 0.0
    compressed_token_accuracy = correct_tokens_compressed / total_tokens if total_tokens > 0 else 0.0
    baseline_char_accuracy = correct_characters_baseline / total_characters if total_characters > 0 else 0.0
    compressed_char_accuracy = correct_characters_compressed / total_characters if total_characters > 0 else 0.0

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "args": vars(args),
                "baseline": {
                    "accuracy": baseline_accuracy,
                    "token_normalized_accuracy": baseline_token_accuracy,
                    "char_normalized_accuracy": baseline_char_accuracy,
                    "correct_predictions": correct_predictions_baseline,
                    "total_predictions": total_predictions,
                    "total_tokens": total_tokens,
                    "total_characters": total_characters,
                },
                "compressed": {
                    "accuracy": compressed_accuracy,
                    "token_normalized_accuracy": compressed_token_accuracy,
                    "char_normalized_accuracy": compressed_char_accuracy,
                    "correct_predictions": correct_predictions_compressed,
                    "total_predictions": total_predictions,
                    "total_tokens": total_tokens,
                    "total_characters": total_characters,
                },
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total samples: {total_predictions}")
    print("\nBaseline (without compression):")
    print(f"  Correct predictions: {correct_predictions_baseline}")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {baseline_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {baseline_char_accuracy:.4f}")
    print("\nCompressed (with compression tokens):")
    print(f"  Correct predictions: {correct_predictions_compressed}")
    print(f"  Accuracy: {compressed_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {compressed_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {compressed_char_accuracy:.4f}")
    print(f"\nDifference: {compressed_accuracy - baseline_accuracy:+.4f}")
    print(f"Results saved to: {results_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
