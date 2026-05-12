"""Downstream multiple-choice evaluation under compression (paper Section 5.6, Tables 5 & 10).

This module produces the 8 PPL variants reported in Table 10 of the paper.
For each benchmark instance (prefix + 4 candidate endings) and an optional
trained compression embedding, we compute per-variant negative log-likelihood
of each ending, then pick the argmin-PPL ending as the prediction.

The 8 variants (from Table 10, plus the "edge" variants which differ in whether
the compression→first-prefix-token logit is included in the scored window):

    Baselines (no compression embedding):
        1. baseline               — full PPL over [prefix + ending]
        2. baseline_endings       — PPL over ending tokens only

    Compression with prefix in context:
        3. compression            — full PPL over [prefix + ending] (compression positions excluded)
        4. compression_edge       — same + include the comp→first-prefix logit
        5. compression_endings    — PPL over ending tokens only

    Compression without prefix in context (replaces prefix):
        6. compression_only           — full PPL over [ending] after [comp]
        7. compression_only_edge      — same + include comp→first-ending logit
        8. compression_only_endings   — PPL over ending tokens only

For each ending we always evaluate teacher-forced (no generation). The model
output picks the continuation with the lowest PPL; accuracy is fraction of
correct predictions over the chosen subset (all samples, or only those with
``convergence == 1.0`` if ``--only_full_convergence`` is set).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.analysis.perplexity import (
    estimate_token_perplexity,
    estimate_token_perplexity_full_labels,
)

PPL_VARIANT_KEYS: tuple[str, ...] = (
    "baseline",
    "baseline_endings",
    "compression",
    "compression_edge",
    "compression_endings",
    "compression_only",
    "compression_only_edge",
    "compression_only_endings",
)

# Variants that are computed with compression embedding prepended. The other
# two (baseline*) use the original prefix only and live on the full-sample
# denominator (they are always reported on every sample).
_COMPRESSION_VARIANTS: tuple[str, ...] = (
    "compression",
    "compression_edge",
    "compression_endings",
    "compression_only",
    "compression_only_edge",
    "compression_only_endings",
)


@torch.no_grad()
def compute_ppl_baseline_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    endings: list[str],
    device: torch.device,
    add_special_tokens: bool = True,
) -> tuple[list[float], list[float]]:
    """Return ``(full_ppls, endings_only_ppls)`` for the four candidate endings.

    No compression embedding involved. Mirrors the legacy
    ``compute_ppl_baseline_batch`` (variants 1 and 2 of Table 10).
    """
    model = model.to(device)
    model.eval()

    if not context:
        return [], []

    full_texts = [f"{context} {ending}" for ending in endings]
    encoded = tokenizer(
        full_texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    context_ids = tokenizer(f"{context} ", add_special_tokens=add_special_tokens)["input_ids"]
    context_len = len(context_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    full_ppls: list[float] = []
    endings_ppls: list[float] = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, :seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]

        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        full_ppls.append(ppl if not math.isnan(ppl) else float("inf"))

        ending_logits = sample_logits[:, context_len - 1 :, :]
        ending_ids = sample_input_ids[:, context_len - 1 :]
        ending_ppl = estimate_token_perplexity(ending_logits, ending_ids)
        endings_ppls.append(ending_ppl if not math.isnan(ending_ppl) else float("inf"))
    return full_ppls, endings_ppls


@torch.no_grad()
def compute_ppl_compression_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_token_embeddings: torch.Tensor,
    context: str,
    endings: list[str],
    device: torch.device,
    add_special_tokens: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(full_ppls, edge_ppls, endings_ppls)`` under compression.

    Tokenizes ``f"{context}{ending}"`` (the context can be empty for the
    "compression only" variants), prepends the compression embedding, runs a
    single batched forward, and slices logits per variant:

    - ``full_ppls``    : PPL over [prefix + ending] excluding the comp→first-token logit
    - ``edge_ppls``    : same but including the comp→first-token logit
                          (off-by-one fix from paper erratum)
    - ``endings_ppls`` : PPL over ending tokens only
    """
    model = model.to(device)
    model.eval()

    full_texts = [f"{context}{ending}" for ending in endings]
    encoded = tokenizer(
        full_texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if context:
        context_ids = tokenizer(context, add_special_tokens=add_special_tokens)["input_ids"]
        context_len = len(context_ids)
    else:
        context_len = 0

    token_embeddings = model.get_input_embeddings()(input_ids)
    num_compression_tokens = compression_token_embeddings.shape[0]

    # Per-sample concat with compression embedding, then right-pad to batch max length.
    united_emb_list: list[torch.Tensor] = []
    united_mask_list: list[torch.Tensor] = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_emb = token_embeddings[i : i + 1, :seq_len]
        sample_mask = attention_mask[i : i + 1, :seq_len]
        comp_emb = compression_token_embeddings.unsqueeze(0).to(token_embeddings.dtype)
        united_emb = torch.cat([comp_emb, sample_emb], dim=1)
        united_mask = torch.cat(
            [
                torch.ones((1, num_compression_tokens), dtype=sample_mask.dtype, device=device),
                sample_mask,
            ],
            dim=1,
        )
        united_emb_list.append(united_emb)
        united_mask_list.append(united_mask)

    max_len = max(item.shape[1] for item in united_emb_list)
    hidden_size = united_emb_list[0].shape[2]
    batch_emb_list: list[torch.Tensor] = []
    batch_mask_list: list[torch.Tensor] = []
    for emb, mask in zip(united_emb_list, united_mask_list):
        pad_len = max_len - emb.shape[1]
        if pad_len > 0:
            emb = torch.cat(
                [
                    emb,
                    torch.zeros(1, pad_len, hidden_size, dtype=emb.dtype, device=device),
                ],
                dim=1,
            )
            mask = torch.cat([mask, torch.zeros(1, pad_len, dtype=mask.dtype, device=device)], dim=1)
        batch_emb_list.append(emb)
        batch_mask_list.append(mask)
    batch_emb = torch.cat(batch_emb_list, dim=0)
    batch_mask = torch.cat(batch_mask_list, dim=0)

    outputs = model(inputs_embeds=batch_emb, attention_mask=batch_mask)

    full_ppls: list[float] = []
    edge_ppls: list[float] = []
    endings_ppls: list[float] = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())

        # variant 3 / 6: full, exclude comp→first-prefix-token logit
        sample_logits = outputs.logits[i : i + 1, num_compression_tokens : num_compression_tokens + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        full_ppls.append(ppl if not math.isnan(ppl) else float("inf"))

        # variant 4 / 7: full + edge (use the comp→first-prefix-token logit too)
        edge_logits = outputs.logits[i : i + 1, num_compression_tokens - 1 : num_compression_tokens + seq_len]
        edge_ppl = estimate_token_perplexity_full_labels(edge_logits, sample_input_ids)
        edge_ppls.append(edge_ppl if not math.isnan(edge_ppl) else float("inf"))

        # variant 5 / 8: endings only
        ending_start = num_compression_tokens + max(0, context_len - 1)
        ending_logits = outputs.logits[i : i + 1, ending_start : num_compression_tokens + seq_len, :]
        ending_ids = sample_input_ids[:, max(0, context_len - 1) : seq_len]
        ending_ppl = estimate_token_perplexity(ending_logits, ending_ids)
        endings_ppls.append(ending_ppl if not math.isnan(ending_ppl) else float("inf"))

    return full_ppls, edge_ppls, endings_ppls


def predict_best_continuation(ppls: list[float]) -> int:
    """Argmin of per-ending PPL → predicted label."""
    return int(torch.tensor(ppls).argmin().item())


def aggregate_variant_accuracy(
    records: list[dict],
    variant: str,
    *,
    only_full_convergence: bool,
) -> dict:
    """Compute counts and accuracies for one PPL variant across saved records.

    Each record must contain ``label``, ``convergence``, ``lengths`` (with
    ``tokens`` and ``characters``), and ``{variant}`` dict with
    ``is_correct``. Baselines are always counted on all samples; compression
    variants respect ``only_full_convergence``.
    """
    is_compression = variant in _COMPRESSION_VARIANTS

    correct_predictions = 0
    total_predictions = 0
    correct_tokens = 0
    total_tokens = 0
    correct_chars = 0
    total_chars = 0

    for r in records:
        is_converged = float(r.get("convergence", 0.0)) >= 1.0
        include = not (is_compression and only_full_convergence) or is_converged
        if not include:
            continue
        entry = r.get(variant)
        if entry is None:
            continue
        total_predictions += 1
        token_count = (r.get("lengths") or {}).get("tokens") or 0
        char_count = (r.get("lengths") or {}).get("characters") or 0
        total_tokens += token_count
        total_chars += char_count
        if entry["is_correct"]:
            correct_predictions += 1
            correct_tokens += token_count
            correct_chars += char_count

    return {
        "accuracy": (correct_predictions / total_predictions if total_predictions else 0.0),
        "token_normalized_accuracy": (correct_tokens / total_tokens if total_tokens else 0.0),
        "char_normalized_accuracy": correct_chars / total_chars if total_chars else 0.0,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "correct_characters": correct_chars,
        "total_characters": total_chars,
    }


def summarize_downstream(
    records: list[dict],
    *,
    only_full_convergence: bool = False,
) -> dict:
    """Aggregate all 8 variants of the per-instance MC outcomes.

    Returns a dict keyed by variant name (Table-10 ordering) with accuracy
    statistics; plus ``num_samples_total`` and ``num_full_convergence`` summary
    counters.
    """
    summary: dict = {
        "num_samples_total": len(records),
        "num_full_convergence": sum(1 for r in records if float(r.get("convergence", 0.0)) >= 1.0),
        "only_full_convergence": only_full_convergence,
    }
    for variant in PPL_VARIANT_KEYS:
        summary[variant] = aggregate_variant_accuracy(records, variant, only_full_convergence=only_full_convergence)
    return summary


@torch.no_grad()
def compute_continuation_nll(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    continuation: str,
    compression_embedding: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Per-token-averaged NLL of ``continuation`` given ``prefix`` (legacy single-pair API).

    Kept for unit-test compatibility; the 8-variant pipeline uses the
    ``compute_ppl_*_batch`` helpers above instead.
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    prefix_ids = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if cont_ids.shape[1] == 0:
        return float("nan")

    full_ids = torch.cat([prefix_ids, cont_ids], dim=1)
    inputs_embeds = model.get_input_embeddings()(full_ids)
    if compression_embedding is not None:
        comp = compression_embedding.unsqueeze(0).to(inputs_embeds.dtype).to(device)
        inputs_embeds = torch.cat([comp, inputs_embeds], dim=1)
        num_comp = compression_embedding.shape[0]
    else:
        num_comp = 0

    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    prefix_len = prefix_ids.shape[1]
    cont_len = cont_ids.shape[1]
    start = num_comp + prefix_len - 1
    end = num_comp + prefix_len + cont_len - 1
    cont_logits = outputs.logits[0, start:end]
    log_probs = torch.log_softmax(cont_logits.float(), dim=-1)
    nll_per_token = -log_probs.gather(1, cont_ids[0].unsqueeze(1)).squeeze(1)
    return float(nll_per_token.mean().item())
