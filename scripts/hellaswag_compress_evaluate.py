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

from compression_horizon.metric import estimate_token_perplexity
from compression_horizon.train.loss import compute_hybrid_cross_entropy_and_alignment_loss
from compression_horizon.utils.launch import freeze_model_parameters, get_device, resolve_torch_dtype, set_launch_seed
from compression_horizon.utils.tokens import count_text_characters, count_text_tokens


def get_decoder_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
    """Return the list of decoder layers for supported model architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, SmolLM2, Gemma
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # Pythia / GPT-NeoX
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}. Cannot locate decoder layers.")


@torch.no_grad()
def compute_attention_mass_per_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_token_embeddings: torch.Tensor,
    context: str,
    num_compression_tokens: int = 1,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute per-layer attention mass on compression token positions.

    Does a single forward pass with output_attentions=True, then for each layer
    computes the mean attention weight directed at compression token positions
    (averaged over heads and query positions). Returns percentages (0-100).

    Args:
        model: The language model (must use eager attention).
        tokenizer: Tokenizer.
        compression_token_embeddings: [num_compression_tokens, hidden_size].
        context: The text context that was compressed.
        num_compression_tokens: Number of compression tokens.
        device: Device to use.
        add_special_tokens: Whether to add special tokens during tokenization.

    Returns:
        List of floats (one per layer), each in [0, 100] representing attention mass %.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    encoded = tokenizer(context, truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embed_fn = model.get_input_embeddings()
    token_embeddings = embed_fn(input_ids)  # [1, seq_len, hidden]

    comp_embeds = compression_token_embeddings.unsqueeze(0).to(token_embeddings.dtype).to(device)
    united_embeddings = torch.cat([comp_embeds, token_embeddings], dim=1)
    united_attention = torch.cat(
        [torch.ones((1, num_compression_tokens), dtype=attention_mask.dtype, device=device), attention_mask],
        dim=1,
    )

    with EagerAttentionContext(model):
        outputs = model(
            inputs_embeds=united_embeddings,
            attention_mask=united_attention,
            output_attentions=True,
        )

    # attentions: tuple of [1, num_heads, seq_len, seq_len] per layer
    attention_mass = []
    for attn_layer in outputs.attentions:
        # Average over heads: [1, seq_len, seq_len] -> mean over query positions
        # attn_layer[:, :, :, :num_ct] = attention to compression tokens
        avg_over_heads = attn_layer.mean(dim=1)  # [1, total_seq, total_seq]
        seq_len = int(united_attention.sum().item())
        # Only consider non-padded query positions
        comp_attn = avg_over_heads[0, :seq_len, :num_compression_tokens].sum(dim=-1)  # [seq_len]
        mass_pct = comp_attn.mean().item() * 100.0
        attention_mass.append(mass_pct)

    return attention_mass


class EagerAttentionContext:
    """Context manager that temporarily switches model to eager attention and restores original on exit."""

    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self._original_impl = None

    def __enter__(self):
        self._original_impl = getattr(self.model.config, "_attn_implementation", None)
        self.model.set_attn_implementation("eager")
        return self

    def __exit__(self, *args):
        if self._original_impl is not None:
            self.model.set_attn_implementation(self._original_impl)


class AttentionKnockoutContext:
    """Context manager that masks attention to compression token positions at specified layers.

    Switches to eager attention (required for 4D mask support), registers forward pre-hooks
    on target decoder layers that modify the 4D attention mask to set compression token
    columns to -inf, then restores original attention implementation on exit.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        knockout_layers: list[int],
        num_compression_tokens: int,
    ):
        self.model = model
        self.knockout_layers = knockout_layers
        self.num_compression_tokens = num_compression_tokens
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.layers = get_decoder_layers(model)
        self._eager_ctx = EagerAttentionContext(model)

    def _make_hook(self):
        num_ct = self.num_compression_tokens

        def hook_fn(module, args, kwargs):
            mask = kwargs.get("attention_mask", None)
            if mask is not None and mask.dim() == 4:
                mask = mask.clone()
                mask[:, :, :, :num_ct] = torch.finfo(mask.dtype).min
                kwargs["attention_mask"] = mask
            return args, kwargs

        return hook_fn

    def __enter__(self):
        self._eager_ctx.__enter__()
        hook_fn = self._make_hook()
        for layer_idx in self.knockout_layers:
            handle = self.layers[layer_idx].register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.hooks.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._eager_ctx.__exit__(*args)


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
            seq_len = int(attention_mask[i].sum().item())
            # Get embeddings for this sample (only non-padded tokens)
            sample_token_embeddings = token_embeddings[i : i + 1, :seq_len]  # [1, seq_len, hidden]
            sample_attention_mask = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
            sample_compression_token_embeddings = compression_token_embeddings[i : i + 1]  # [1, num_compression_tokens, hidden]
            # Concatenate
            united_token_embeddings = torch.cat(
                [sample_compression_token_embeddings, sample_token_embeddings], dim=1
            )  # [1, mem+seq, hidden]
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
                        torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device),
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
            seq_len = int(attention_mask[i].sum().item())
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
    model: AutoModelForCausalLM,
    batch_embeddings: torch.Tensor,
    batch_attention: torch.Tensor,
    batch_labels: torch.Tensor,
    num_compression_tokens: int,
) -> list[float]:
    """Calculate token-level accuracy between predicted and target tokens.

    Args:
        batch_embeddings: [batch_size, max_len, hidden] - input embeddings with compression tokens
        batch_attention: [batch_size, max_len] - attention mask
        batch_labels: [batch_size, max_seq_len] - target token IDs (without compression positions)
        num_compression_tokens: number of compression tokens prepended

    Returns:
        List of convergence scores (0.0 to 1.0) for each sample
    """
    outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)
    batch_size = batch_embeddings.shape[0]
    convergences = []
    for i in range(batch_size):
        # seq_len includes compression tokens
        seq_len = int(batch_attention[i].sum().item())
        orig_seq_len = seq_len - num_compression_tokens

        if orig_seq_len <= 0:
            convergences.append(0.0)
            continue

        # logits[num_compression_tokens - 1] predicts first original token
        # logits[num_compression_tokens - 1 + j] predicts original token j
        # We want predictions for tokens 0 to orig_seq_len-1
        sample_logits = outputs.logits[i, num_compression_tokens - 1 : seq_len - 1]  # [orig_seq_len, vocab]
        sample_predicted_tokens = sample_logits.argmax(dim=-1)  # [orig_seq_len]
        sample_labels = batch_labels[i, :orig_seq_len]  # [orig_seq_len]

        convergence = (sample_predicted_tokens == sample_labels).float().mean().item()
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
    encoded = tokenizer(
        full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens
    )
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Compute PPL for each sample
    ppls = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, :seq_len]  # [1, seq_len, vocab]
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
    encoded = tokenizer(
        full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens
    )
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Get token embeddings for all texts
    token_embeddings = model.model.embed_tokens(input_ids)  # [batch_size, seq_len, hidden]

    # Prepare batched inputs with compression tokens
    united_token_embeddings_list = []
    united_attention_mask_list = []
    num_compression_tokens = compression_token_embeddings[0].shape[0]
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_token_embeddings = token_embeddings[i : i + 1, :seq_len]  # [1, seq_len, hidden]
        sample_attention_mask = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
        sample_compression_token_embeddings = (
            compression_token_embeddings[i].unsqueeze(0).to(token_embeddings.dtype)
        )  # [1, num_compression_tokens, hidden]
        # Concatenate
        united_token_embeddings = torch.cat(
            [sample_compression_token_embeddings, sample_token_embeddings], dim=1
        )  # [1, mem+seq, hidden]
        united_attention_mask = torch.cat(
            [torch.ones((1, num_compression_tokens), dtype=sample_attention_mask.dtype, device=device), sample_attention_mask],
            dim=1,
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
                    torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device),
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
    # Compute PPL for each sample (same as baseline: logits[j] predicts labels[j+1])
    # logits[num_compression_tokens + j] predicts input_ids[j + 1]
    ppls = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, num_compression_tokens : num_compression_tokens + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]  # [1, seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)
    return ppls


@torch.no_grad()
def compute_ppl_with_compression_and_knockout_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_token_embeddings: list[torch.Tensor],
    contexts: list[str],
    endings: list[str],
    knockout_layers: list[int],
    num_compression_tokens: int = 1,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute PPL with compression tokens prepended and attention knockout at specified layers.

    Same as compute_ppl_with_compression_batch but masks attention to compression token
    positions at the specified layers (pre-softmax -inf masking).
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
    encoded = tokenizer(
        full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Get token embeddings
    embed_fn = model.get_input_embeddings()
    token_embeddings = embed_fn(input_ids)

    # Prepare batched inputs with compression tokens
    united_token_embeddings_list = []
    united_attention_mask_list = []
    num_ct = compression_token_embeddings[0].shape[0]
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_token_embeddings = token_embeddings[i : i + 1, :seq_len]
        sample_attention_mask = attention_mask[i : i + 1, :seq_len]
        sample_compression_token_embeddings = compression_token_embeddings[i].unsqueeze(0).to(token_embeddings.dtype)
        united_token_embeddings = torch.cat([sample_compression_token_embeddings, sample_token_embeddings], dim=1)
        united_attention_mask = torch.cat(
            [torch.ones((1, num_ct), dtype=sample_attention_mask.dtype, device=device), sample_attention_mask],
            dim=1,
        )
        united_token_embeddings_list.append(united_token_embeddings)
        united_attention_mask_list.append(united_attention_mask)

    # Pad to maximum length
    max_len = max(item.shape[1] for item in united_token_embeddings_list)
    batch_embeddings = []
    batch_attention = []
    for i in range(len(full_texts)):
        united_token_embeddings = united_token_embeddings_list[i]
        united_attention_mask = united_attention_mask_list[i]
        current_len = united_token_embeddings.shape[1]
        if current_len < max_len:
            pad_len = max_len - current_len
            hidden_size = united_token_embeddings.shape[2]
            united_token_embeddings = torch.cat(
                [
                    united_token_embeddings,
                    torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device),
                ],
                dim=1,
            )
            united_attention_mask = torch.cat(
                [
                    united_attention_mask,
                    torch.zeros(1, pad_len, dtype=united_attention_mask.dtype, device=device),
                ],
                dim=1,
            )
        batch_embeddings.append(united_token_embeddings)
        batch_attention.append(united_attention_mask)
    batch_embeddings = torch.cat(batch_embeddings, dim=0)
    batch_attention = torch.cat(batch_attention, dim=0)

    # Forward pass with attention knockout
    with AttentionKnockoutContext(model, knockout_layers, num_ct):
        outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)

    # Compute PPL for each sample
    ppls = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, num_ct : num_ct + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]
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
        "--only_full_convergence",
        action="store_true",
        default=False,
        help="Only count examples with perfect convergence (accuracy=1.0) in compressed metrics.",
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
    parser.add_argument(
        "--no-intervention",
        "--no_intervention",
        dest="intervention",
        action="store_false",
        default=True,
        help="Disable attention knockout intervention mode (enabled by default).",
    )
    parser.add_argument(
        "--skip_per_layer",
        action="store_true",
        default=False,
        help="Skip per-layer knockout sweep (only used with --intervention).",
    )
    parser.add_argument(
        "--skip_cumulative",
        action="store_true",
        default=False,
        help="Skip cumulative knockout sweep (only used with --intervention).",
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

    # Get number of layers for intervention mode
    num_model_layers = None
    if args.intervention:
        num_model_layers = len(get_decoder_layers(model))
        print(f"Intervention mode enabled. Model has {num_model_layers} layers.")
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
    # Baseline counters (always count all samples)
    total_predictions_baseline = 0
    correct_predictions_baseline = 0
    total_tokens_baseline = 0
    total_characters_baseline = 0
    correct_tokens_baseline = 0
    correct_characters_baseline = 0
    # Compressed counters (may exclude non-converged samples when only_full_convergence=True)
    total_predictions_compressed = 0
    correct_predictions_compressed = 0
    total_tokens_compressed = 0
    total_characters_compressed = 0
    correct_tokens_compressed = 0
    correct_characters_compressed = 0

    # Intervention counters (per-layer and cumulative knockout)
    if args.intervention:
        intervention_per_layer_correct = {li: 0 for li in range(num_model_layers)}
        intervention_per_layer_total = {li: 0 for li in range(num_model_layers)}
        intervention_cumulative_correct = {li: 0 for li in range(num_model_layers)}
        intervention_cumulative_total = {li: 0 for li in range(num_model_layers)}

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
            contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
            try:
                sample_ppls = compute_ppl_baseline_batch(
                    model=model,
                    tokenizer=tokenizer,
                    contexts=contexts_for_batch,
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
            except Exception as e:
                print(f"Error computing baseline PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_baseline_ppls.append(sample_ppls)

        # Compress contexts (batched)
        batch_compression_results = None
        try:
            # List[dict] with keys: compression_embedding, convergence
            batch_compression_results = compress_prefixes_batch(
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
        except Exception as e:
            print(f"Error compressing batch {batch_idx}: {e}")
            batch_compression_results = [{"compression_embedding": None, "convergence": 0.0}] * actual_batch_size

        # Compute compressed PPL for all endings (batched)
        batch_compressed_ppls = []
        for sample_idx in range(actual_batch_size):
            compression_result = batch_compression_results[sample_idx]
            compression_embedding = compression_result["compression_embedding"]
            convergence = compression_result["convergence"]

            # Check if compression failed
            if compression_embedding is None:
                batch_compressed_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                continue

            # Compute PPL with compression tokens
            contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
            try:
                batch_sample_compression_token_embeddings = [compression_embedding] * len(batch_endings_list[sample_idx])
                ppls = compute_ppl_with_compression_batch(
                    model=model,
                    tokenizer=tokenizer,
                    compression_token_embeddings=batch_sample_compression_token_embeddings,
                    contexts=contexts_for_batch,
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
            except Exception as e:
                print(f"Error computing compressed PPL for sample {start_idx + sample_idx}: {e}")
                ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_compressed_ppls.append({"ppls": ppls, "convergence": convergence})

        # Compute knockout PPLs and attention mass for intervention mode
        batch_per_layer_knockout_ppls = []
        batch_cumulative_knockout_ppls = []
        batch_attention_mass = []
        if args.intervention:
            for sample_idx in range(actual_batch_size):
                compression_result = batch_compression_results[sample_idx]
                compression_embedding = compression_result["compression_embedding"]

                if compression_embedding is None:
                    num_endings = len(batch_endings_list[sample_idx])
                    batch_per_layer_knockout_ppls.append({li: [float("inf")] * num_endings for li in range(num_model_layers)})
                    batch_cumulative_knockout_ppls.append({li: [float("inf")] * num_endings for li in range(num_model_layers)})
                    batch_attention_mass.append([0.0] * num_model_layers)
                    continue

                # Compute per-layer attention mass
                try:
                    attn_mass = compute_attention_mass_per_layer(
                        model=model,
                        tokenizer=tokenizer,
                        compression_token_embeddings=compression_embedding,
                        context=batch_contexts[sample_idx],
                        num_compression_tokens=args.num_compression_tokens,
                        device=device,
                        add_special_tokens=add_special_tokens,
                    )
                except Exception as e:
                    print(f"Error computing attention mass for sample {start_idx + sample_idx}: {e}")
                    attn_mass = [0.0] * num_model_layers
                batch_attention_mass.append(attn_mass)

                contexts_for_sample = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
                comp_embeds_for_sample = [compression_embedding] * len(batch_endings_list[sample_idx])

                # Per-layer knockout
                sample_per_layer = {}
                if not args.skip_per_layer:
                    for li in range(num_model_layers):
                        try:
                            ko_ppls = compute_ppl_with_compression_and_knockout_batch(
                                model=model,
                                tokenizer=tokenizer,
                                compression_token_embeddings=comp_embeds_for_sample,
                                contexts=contexts_for_sample,
                                endings=batch_endings_list[sample_idx],
                                knockout_layers=[li],
                                num_compression_tokens=args.num_compression_tokens,
                                device=device,
                                add_special_tokens=add_special_tokens,
                            )
                        except Exception as e:
                            print(f"Error in per-layer KO (layer {li}) for sample {start_idx + sample_idx}: {e}")
                            ko_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                        sample_per_layer[li] = ko_ppls
                batch_per_layer_knockout_ppls.append(sample_per_layer)

                # Cumulative knockout
                sample_cumulative = {}
                if not args.skip_cumulative:
                    for li in range(num_model_layers):
                        try:
                            ko_ppls = compute_ppl_with_compression_and_knockout_batch(
                                model=model,
                                tokenizer=tokenizer,
                                compression_token_embeddings=comp_embeds_for_sample,
                                contexts=contexts_for_sample,
                                endings=batch_endings_list[sample_idx],
                                knockout_layers=list(range(li + 1)),
                                num_compression_tokens=args.num_compression_tokens,
                                device=device,
                                add_special_tokens=add_special_tokens,
                            )
                        except Exception as e:
                            print(f"Error in cumulative KO (layers 0..{li}) for sample {start_idx + sample_idx}: {e}")
                            ko_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                        sample_cumulative[li] = ko_ppls
                batch_cumulative_knockout_ppls.append(sample_cumulative)

        # Process results for this batch
        for sample_idx in range(actual_batch_size):
            idx = start_idx + sample_idx
            context = batch_contexts[sample_idx]
            endings = batch_endings_list[sample_idx]
            label = batch_labels[sample_idx]

            # Baseline evaluation
            baseline_ppls = batch_baseline_ppls[sample_idx]
            baseline_predicted_label = int(torch.tensor(baseline_ppls).argmin().item())
            baseline_is_correct = baseline_predicted_label == label

            # Compressed evaluation
            compressed_result = batch_compressed_ppls[sample_idx]
            compressed_ppls = compressed_result["ppls"]
            convergence = compressed_result["convergence"]
            is_fully_converged = convergence >= 1.0

            # Determine if this sample should be counted in compressed metrics
            should_count_compressed = not args.only_full_convergence or is_fully_converged

            compressed_predicted_label = int(torch.tensor(compressed_ppls).argmin().item())
            compressed_is_correct = compressed_predicted_label == label

            # Update baseline counters (always count all samples)
            total_predictions_baseline += 1
            if baseline_is_correct:
                correct_predictions_baseline += 1

            # Update compressed counters (respects only_full_convergence flag)
            if should_count_compressed:
                total_predictions_compressed += 1
                if compressed_is_correct:
                    correct_predictions_compressed += 1

            # Compute token/char counts for correct ending
            token_count = None
            char_count = None
            if 0 <= label < len(endings):
                correct_ending = endings[label]
                full_text = f"{context} {correct_ending}"
                token_count = count_text_tokens(tokenizer, full_text, add_special_tokens=add_special_tokens)
                char_count = count_text_characters(full_text)

                # Baseline token/char counts
                total_tokens_baseline += token_count
                total_characters_baseline += char_count
                if baseline_is_correct:
                    correct_tokens_baseline += token_count
                    correct_characters_baseline += char_count

                # Compressed token/char counts (respects only_full_convergence flag)
                if should_count_compressed:
                    total_tokens_compressed += token_count
                    total_characters_compressed += char_count
                    if compressed_is_correct:
                        correct_tokens_compressed += token_count
                        correct_characters_compressed += char_count

            # Store result (always save, regardless of convergence)
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
                    "convergence": convergence,
                    "is_fully_converged": is_fully_converged,
                    "counted_in_metrics": should_count_compressed,
                },
            }

            # Add intervention results
            if args.intervention:
                # Attention mass per layer
                if batch_attention_mass:
                    result["attention_mass_per_layer"] = batch_attention_mass[sample_idx]

                # Per-layer knockout
                if not args.skip_per_layer and batch_per_layer_knockout_ppls:
                    per_layer_results = {}
                    per_layer_data = batch_per_layer_knockout_ppls[sample_idx]
                    for li, ko_ppls in per_layer_data.items():
                        pred = int(torch.tensor(ko_ppls).argmin().item())
                        is_correct = pred == label
                        per_layer_results[str(li)] = {
                            "ppls": ko_ppls,
                            "predicted_label": pred,
                            "is_correct": is_correct,
                        }
                        intervention_per_layer_total[li] += 1
                        if is_correct:
                            intervention_per_layer_correct[li] += 1
                    result["per_layer_knockout"] = per_layer_results

                # Cumulative knockout
                if not args.skip_cumulative and batch_cumulative_knockout_ppls:
                    cumulative_results = {}
                    cumulative_data = batch_cumulative_knockout_ppls[sample_idx]
                    for li, ko_ppls in cumulative_data.items():
                        pred = int(torch.tensor(ko_ppls).argmin().item())
                        is_correct = pred == label
                        cumulative_results[str(li)] = {
                            "ppls": ko_ppls,
                            "predicted_label": pred,
                            "is_correct": is_correct,
                        }
                        intervention_cumulative_total[li] += 1
                        if is_correct:
                            intervention_cumulative_correct[li] += 1
                    result["cumulative_knockout"] = cumulative_results

            results.append(result)

        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
            baseline_accuracy = (
                correct_predictions_baseline / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
            )
            compressed_accuracy = (
                correct_predictions_compressed / total_predictions_compressed if total_predictions_compressed > 0 else 0.0
            )
            print(
                f"Progress: {total_predictions_baseline}/{len(dataset)}, Baseline Accuracy: {baseline_accuracy:.4f}, "
                f"Compressed Accuracy: {compressed_accuracy:.4f} ({total_predictions_compressed} samples)"
            )

    # Compute final accuracies
    baseline_accuracy = correct_predictions_baseline / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
    compressed_accuracy = (
        correct_predictions_compressed / total_predictions_compressed if total_predictions_compressed > 0 else 0.0
    )
    baseline_token_accuracy = correct_tokens_baseline / total_tokens_baseline if total_tokens_baseline > 0 else 0.0
    compressed_token_accuracy = correct_tokens_compressed / total_tokens_compressed if total_tokens_compressed > 0 else 0.0
    baseline_char_accuracy = correct_characters_baseline / total_characters_baseline if total_characters_baseline > 0 else 0.0
    compressed_char_accuracy = (
        correct_characters_compressed / total_characters_compressed if total_characters_compressed > 0 else 0.0
    )

    # Build intervention summary
    intervention_summary = None
    if args.intervention:
        intervention_summary = {}
        if not args.skip_per_layer:
            per_layer_summary = {}
            for li in range(num_model_layers):
                total = intervention_per_layer_total[li]
                correct = intervention_per_layer_correct[li]
                per_layer_summary[str(li)] = {
                    "accuracy": correct / total if total > 0 else 0.0,
                    "correct": correct,
                    "total": total,
                }
            intervention_summary["per_layer_knockout"] = per_layer_summary
        if not args.skip_cumulative:
            cumulative_summary = {}
            for li in range(num_model_layers):
                total = intervention_cumulative_total[li]
                correct = intervention_cumulative_correct[li]
                cumulative_summary[str(li)] = {
                    "accuracy": correct / total if total > 0 else 0.0,
                    "correct": correct,
                    "total": total,
                }
            intervention_summary["cumulative_knockout"] = cumulative_summary

        # Average attention mass across samples
        all_attn_mass = [r["attention_mass_per_layer"] for r in results if "attention_mass_per_layer" in r]
        if all_attn_mass:
            avg_attn_mass = [sum(s[li] for s in all_attn_mass) / len(all_attn_mass) for li in range(num_model_layers)]
            intervention_summary["avg_attention_mass_per_layer"] = avg_attn_mass

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    output_data = {
        "args": vars(args),
        "baseline": {
            "accuracy": baseline_accuracy,
            "token_normalized_accuracy": baseline_token_accuracy,
            "char_normalized_accuracy": baseline_char_accuracy,
            "correct_predictions": correct_predictions_baseline,
            "total_predictions": total_predictions_baseline,
            "total_tokens": total_tokens_baseline,
            "total_characters": total_characters_baseline,
        },
        "compressed": {
            "accuracy": compressed_accuracy,
            "token_normalized_accuracy": compressed_token_accuracy,
            "char_normalized_accuracy": compressed_char_accuracy,
            "correct_predictions": correct_predictions_compressed,
            "total_predictions": total_predictions_compressed,
            "total_predictions_all": total_predictions_baseline,
            "total_tokens": total_tokens_compressed,
            "total_characters": total_characters_compressed,
            "only_full_convergence": args.only_full_convergence,
        },
        "results": results,
    }
    if intervention_summary is not None:
        output_data["intervention_summary"] = intervention_summary
        output_data["num_model_layers"] = num_model_layers
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total samples: {total_predictions_baseline}")
    print(f"Only full convergence: {args.only_full_convergence}")
    print("\nBaseline (without compression):")
    print(f"  Correct predictions: {correct_predictions_baseline}/{total_predictions_baseline}")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {baseline_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {baseline_char_accuracy:.4f}")
    print("\nCompressed (with compression tokens):")
    print(f"  Samples counted: {total_predictions_compressed}/{total_predictions_baseline}")
    print(f"  Correct predictions: {correct_predictions_compressed}/{total_predictions_compressed}")
    print(f"  Accuracy: {compressed_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {compressed_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {compressed_char_accuracy:.4f}")
    print(f"\nDifference: {compressed_accuracy - baseline_accuracy:+.4f}")

    if args.intervention and intervention_summary:
        if "per_layer_knockout" in intervention_summary:
            print("\nPer-layer Knockout:")
            best_layer = max(
                intervention_summary["per_layer_knockout"],
                key=lambda li: intervention_summary["per_layer_knockout"][li]["accuracy"],
            )
            worst_layer = min(
                intervention_summary["per_layer_knockout"],
                key=lambda li: intervention_summary["per_layer_knockout"][li]["accuracy"],
            )
            best_acc = intervention_summary["per_layer_knockout"][best_layer]["accuracy"]
            worst_acc = intervention_summary["per_layer_knockout"][worst_layer]["accuracy"]
            print(f"  Best single-layer KO: layer {best_layer} (accuracy={best_acc:.4f})")
            print(f"  Worst single-layer KO: layer {worst_layer} (accuracy={worst_acc:.4f})")
        if "cumulative_knockout" in intervention_summary:
            print("\nCumulative Knockout:")
            last_layer = str(num_model_layers - 1)
            full_ko_acc = intervention_summary["cumulative_knockout"][last_layer]["accuracy"]
            print(f"  Full knockout (all layers) accuracy: {full_ko_acc:.4f}")
            print(f"  Base accuracy: {baseline_accuracy:.4f}")
            print(f"  Sanity check delta (should be ~0): {full_ko_acc - baseline_accuracy:+.4f}")

    print(f"\nResults saved to: {results_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
