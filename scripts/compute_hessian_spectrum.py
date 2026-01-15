#!/usr/bin/env python3
"""
Compute and visualize Hessian eigenspectrum along progressive training trajectory.

This script:
1. Loads progressive training artifacts
2. Reconstructs the model and compression tokens at each stage
3. Computes Hessian eigenvalues of the loss w.r.t. compression tokens
4. Visualizes how the spectrum evolves during optimization

Hypothesis: "Walls" in optimization show up as eigenvalue spikes, and compression
works by finding "flat" regions in specific directions.
"""
import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.utils.launch import freeze_model_parameters, get_device


def load_progressive_dataset(dataset_path: str) -> Dataset:
    """Load progressive training artifacts dataset."""
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


def compute_loss_for_hessian(
    model,
    compression_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    num_compression_tokens: int,
    loss_type: str = "cross_entropy",
    num_alignment_layers: int = 0,
    hybrid_alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute loss given compression tokens (for Hessian computation).

    This replicates the loss computation from MyTrainer.compute_loss but
    takes compression_tokens as input (not as a parameter).
    """
    # Get target hidden states (no grad needed for target)
    model.eval()
    with torch.no_grad():
        outputs = model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Prepare inputs with compression tokens
    compression_tokens_attention_mask = torch.ones(
        compression_tokens.shape[0],
        compression_tokens.shape[1],
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    united_token_embeddings = torch.cat([compression_tokens, token_embeddings.detach()], dim=1)
    united_attention_mask = torch.cat([compression_tokens_attention_mask, attention_mask], dim=1)

    # Get compression outputs (need gradients w.r.t. compression_tokens)
    # Model can stay in eval mode since it's frozen; gradients flow through compression_tokens
    compression_outputs = model(
        inputs_embeds=united_token_embeddings,
        attention_mask=united_attention_mask,
        output_hidden_states=True,
    )

    # Cross entropy loss
    # Logits shape: [batch, num_compression_tokens + seq_len, vocab_size]
    # We want to predict input_ids, so we take logits starting from position num_compression_tokens-1
    # (the last compression token predicts the first input token)
    # The slice [num_compression_tokens - 1 : -1] gives us logits for positions that predict input_ids
    logits_full = compression_outputs.logits  # [batch, total_len, vocab_size]
    seq_len = input_ids.shape[1]

    # Use the same indexing as the original trainer: [num_compression_tokens - 1 : -1]
    # This gives us logits for predicting the input sequence
    logits_slice = logits_full[:, num_compression_tokens - 1 : -1]  # [batch, seq_len, vocab_size]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Ensure shapes match - they should be the same, but handle edge cases
    logits_seq_len = logits_slice.shape[1]
    labels_seq_len = labels.shape[1]

    if logits_seq_len != labels_seq_len:
        # Take the minimum to ensure they match
        min_len = min(logits_seq_len, labels_seq_len)
        if min_len == 0:
            # Fallback: return a zero loss that still depends on compression_tokens,
            # so autograd can form a valid graph (Hessian will be zero).
            return compression_tokens.sum() * 0.0
        logits_slice = logits_slice[:, :min_len, :]
        labels = labels[:, :min_len]
        attention_mask = attention_mask[:, :min_len]
        labels[attention_mask == 0] = -100

    # Flatten for cross_entropy: [batch * seq_len, vocab_size] and [batch * seq_len]
    logits_flat = logits_slice.flatten(0, 1)  # [batch * seq_len, vocab_size]
    labels_flat = labels.flatten()  # [batch * seq_len]

    # Final shape check before cross_entropy
    if logits_flat.shape[0] != labels_flat.shape[0]:
        # This shouldn't happen, but handle it gracefully
        min_size = min(logits_flat.shape[0], labels_flat.shape[0])
        if min_size == 0:
            # Return a zero loss that still depends on compression_tokens.
            return compression_tokens.sum() * 0.0
        logits_flat = logits_flat[:min_size, :]
        labels_flat = labels_flat[:min_size]

    try:
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction="mean",
        )
    except RuntimeError as e:
        # Provide more informative error message
        raise RuntimeError(
            f"Cross entropy shape mismatch: logits_flat.shape={logits_flat.shape}, "
            f"labels_flat.shape={labels_flat.shape}, "
            f"logits_slice.shape={logits_slice.shape}, labels.shape={labels.shape}, "
            f"num_compression_tokens={num_compression_tokens}, seq_len={seq_len}, "
            f"logits_full.shape={logits_full.shape}"
        ) from e

    # Activation alignment loss (if hybrid_alpha is set)
    if hybrid_alpha is not None and loss_type != "cross_entropy":
        total_layers = len(outputs.hidden_states)
        if num_alignment_layers > 0:
            num_layers = max(0, min(num_alignment_layers, total_layers))
            alignment_layer_indices = range(num_layers)
        else:
            alignment_layer_indices = range(total_layers)

        alignment_loss = 0
        for i in alignment_layer_indices:
            compression_hidden_states = compression_outputs.hidden_states[i][:, num_compression_tokens:]
            target_hidden_states = outputs.hidden_states[i]
            if loss_type == "l2":
                layer_alignment_loss = (
                    F.mse_loss(
                        compression_hidden_states,
                        target_hidden_states,
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .sqrt()
                    .mean()
                )
            elif loss_type == "l1":
                layer_alignment_loss = (
                    F.l1_loss(
                        compression_hidden_states,
                        target_hidden_states,
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean()
                )
            elif loss_type == "cosine":
                cosine = F.cosine_similarity(compression_hidden_states, target_hidden_states, dim=-1)
                layer_alignment_loss = (1.0 - cosine).mean()
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")
            alignment_loss = alignment_loss + layer_alignment_loss
        loss = loss + hybrid_alpha * alignment_loss

    return loss


def compute_hessian_eigenvalues(
    model,
    compression_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    num_compression_tokens: int,
    loss_type: str = "cross_entropy",
    num_alignment_layers: int = 0,
    hybrid_alpha: Optional[float] = None,
    num_eigenvalues: int = 20,
    use_lanczos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hessian eigenvalues w.r.t. compression tokens.

    Args:
        model: The language model (frozen)
        compression_tokens: Current compression token embeddings [batch, num_tokens, hidden]
        input_ids: Input token IDs [batch, seq_len]
        token_embeddings: Input token embeddings [batch, seq_len, hidden]
        attention_mask: Attention mask [batch, seq_len]
        num_compression_tokens: Number of compression tokens
        loss_type: Type of alignment loss
        num_alignment_layers: Number of layers to align (0 = all)
        hybrid_alpha: Weight for alignment loss
        num_eigenvalues: Number of eigenvalues to compute
        use_lanczos: Whether to use Lanczos method (more efficient for large Hessians)

    Returns:
        eigenvalues: Tensor of eigenvalues (sorted descending)
        eigenvectors: Tensor of corresponding eigenvectors [num_eigenvalues, flattened_dim]
    """
    # device = compression_tokens.device
    # dtype = compression_tokens.dtype

    # Flatten compression tokens for Hessian computation
    flat_shape = compression_tokens.shape
    # Create a tensor that requires grad - this will be the parameter we optimize
    # We need to clone to avoid modifying the original, then flatten and set requires_grad
    # Use reshape instead of view for better compatibility
    flattened = compression_tokens.detach().clone().reshape(-1)
    flattened = flattened.requires_grad_(True)  # This ensures it's a leaf tensor with requires_grad

    def loss_fn(flat_params: torch.Tensor) -> torch.Tensor:
        """Loss function that takes flattened parameters."""
        # flat_params should already require grad from outer scope
        # Reshape to original shape - reshape maintains gradient connection
        # Even if it creates a copy, gradients will still flow back to flat_params
        comp_tokens = flat_params.reshape(flat_shape)
        # comp_tokens should now be connected to flat_params in the computation graph
        # Pass comp_tokens directly to compute_loss_for_hessian
        return compute_loss_for_hessian(
            model,
            comp_tokens,
            input_ids,
            token_embeddings,
            attention_mask,
            num_compression_tokens,
            loss_type,
            num_alignment_layers,
            hybrid_alpha,
        )

    # Compute Hessian-vector products using autograd
    # For efficiency, we'll use a Lanczos-based approach for large Hessians
    # Always use Lanczos for now as it's more memory efficient
    if use_lanczos or flattened.numel() > 500:
        # Use iterative method to compute top eigenvalues
        eigenvalues, eigenvectors = compute_hessian_eigenvalues_lanczos(loss_fn, flattened, num_eigenvalues=num_eigenvalues)
    else:
        # Compute full Hessian for small parameter spaces
        try:
            from torch.func import hessian

            H = hessian(loss_fn)(flattened)
            # Compute eigenvalues
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            # Sort descending
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # Take top num_eigenvalues
            eigenvalues = eigenvalues[:num_eigenvalues]
            eigenvectors = eigenvectors[:, :num_eigenvalues].T
        except (ImportError, RuntimeError) as e:
            # Fallback: use Lanczos method
            print(f"Warning: Full Hessian computation failed ({e}), using Lanczos method")
            eigenvalues, eigenvectors = compute_hessian_eigenvalues_lanczos(loss_fn, flattened, num_eigenvalues=num_eigenvalues)

    return eigenvalues.detach().cpu(), eigenvectors.detach().cpu()


def compute_hessian_eigenvalues_lanczos(
    loss_fn,
    params: torch.Tensor,
    num_eigenvalues: int = 20,
    num_lanczos_iterations: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top eigenvalues using Lanczos algorithm with Hessian-vector products.

    This is more memory-efficient for large parameter spaces.
    """
    if num_lanczos_iterations is None:
        num_lanczos_iterations = min(2 * num_eigenvalues + 10, params.numel())

    device = params.device
    dtype = params.dtype

    def hvp(v: torch.Tensor) -> torch.Tensor:
        """Hessian-vector product: H @ v"""
        # Params should already require grad from outer scope
        # We need to ensure params is used directly in the computation graph
        # Compute loss - this must use params in a way that creates gradients
        loss = loss_fn(params)
        # Verify that loss depends on params by checking if it has grad_fn
        if loss.grad_fn is None:
            raise RuntimeError(
                f"Loss does not depend on params. "
                f"params.requires_grad={params.requires_grad}, "
                f"loss.requires_grad={loss.requires_grad}, "
                f"loss.grad_fn={loss.grad_fn}"
            )
        # Compute gradient w.r.t. params
        # Use allow_unused=True temporarily to see which params are unused
        try:
            grad = torch.autograd.grad(
                loss, params, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=False
            )[0]
        except RuntimeError as e:
            if "not have been used" in str(e):
                raise RuntimeError(
                    f"Params not used in computation graph. "
                    f"params.requires_grad={params.requires_grad}, "
                    f"params.shape={params.shape}, "
                    f"loss.grad_fn={loss.grad_fn}. "
                    f"Check that compression_tokens is properly used in compute_loss_for_hessian."
                ) from e
            raise
        if grad is None:
            # If grad is None, params weren't actually connected; treat as zero curvature.
            return torch.zeros_like(params)
        if not grad.requires_grad:
            # If the gradient is a constant w.r.t. params (e.g., linear loss), Hessian is zero.
            return torch.zeros_like(params)
        # Compute HVP: derivative of (grad @ v) w.r.t. params
        # This is equivalent to H @ v where H is the Hessian
        try:
            hvp_val = torch.autograd.grad(
                grad, params, grad_outputs=v, retain_graph=False, only_inputs=True, allow_unused=False
            )[0]
        except RuntimeError as e:
            # Happens when grad is non-differentiable (no second derivatives available).
            # In that case the Hessian is effectively zero.
            msg = str(e)
            if "does not require grad" in msg or "does not have a grad_fn" in msg:
                return torch.zeros_like(params)
            raise
        if hvp_val is None:
            return torch.zeros_like(params)
        return hvp_val.detach()

    # Ensure params requires grad before starting
    if not params.requires_grad:
        params = params.requires_grad_(True)

    # Initialize Lanczos vectors
    q = torch.randn_like(params)
    q = q / (torch.norm(q) + 1e-10)
    Q = [q.clone()]
    alpha = []
    beta = [torch.tensor(0.0, device=device, dtype=dtype)]

    # Lanczos iteration
    for i in range(num_lanczos_iterations):
        if i == 0:
            r = hvp(q)
        else:
            r = hvp(q) - beta[-1] * Q[-2]
        alpha_i = torch.dot(r, q).item()
        alpha.append(alpha_i)
        r = r - alpha_i * q
        if i < num_lanczos_iterations - 1:
            beta_i = torch.norm(r).item()
            if beta_i < 1e-10:
                break
            beta.append(beta_i)
            q = r / beta_i
            Q.append(q.clone())
        else:
            beta.append(torch.tensor(0.0, device=device, dtype=dtype))

    # Build tridiagonal matrix
    n = len(alpha)
    if n < 2:
        # Fallback: return dummy values
        eigenvals = torch.zeros(num_eigenvalues, device=device, dtype=dtype)
        eigenvecs = torch.eye(params.numel(), num_eigenvalues, device=device, dtype=dtype)
        return eigenvals.detach().cpu(), eigenvecs.detach().cpu().T

    T = torch.zeros(n, n, device=device, dtype=dtype)
    for i in range(n):
        T[i, i] = alpha[i]
        if i < n - 1:
            T[i, i + 1] = beta[i + 1]
            T[i + 1, i] = beta[i + 1]

    # Compute eigenvalues of tridiagonal matrix
    eigenvals, eigenvecs_tridiag = torch.linalg.eigh(T)
    idx = torch.argsort(eigenvals, descending=True)
    eigenvals = eigenvals[idx][:num_eigenvalues]
    eigenvecs_tridiag = eigenvecs_tridiag[:, idx][:, :num_eigenvalues]

    # Convert back to original space
    Q_matrix = torch.stack(Q, dim=1)  # [param_dim, num_iterations]
    eigenvecs = Q_matrix @ eigenvecs_tridiag  # [param_dim, num_eigenvalues]

    return eigenvals.detach().cpu(), eigenvecs.detach().cpu().T


def compute_hessian_for_stage(
    model,
    row: Dict[str, Any],
    tokenizer: AutoTokenizer,
    device: torch.device,
    num_eigenvalues: int = 20,
    loss_type: str = "cross_entropy",
    num_alignment_layers: int = 0,
    hybrid_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute Hessian eigenvalues for a single stage."""
    # Extract data from row
    embedding = torch.tensor(row["embedding"], dtype=torch.float32).to(device)
    text = row.get("text", "")
    stage_seq_len = int(row.get("stage_seq_len", 0))

    # Tokenize text - use truncation but don't force padding to exact length
    # The actual sequence length might be shorter than stage_seq_len
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=stage_seq_len,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # If sequence is shorter than stage_seq_len, pad it
    if input_ids.shape[1] < stage_seq_len:
        pad_length = stage_seq_len - input_ids.shape[1]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padding = torch.full((input_ids.shape[0], pad_length), pad_token_id, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, padding], dim=1)
        padding_mask = torch.zeros((attention_mask.shape[0], pad_length), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, padding_mask], dim=1)

    # Get token embeddings
    with torch.no_grad():
        token_embeddings = model.model.embed_tokens(input_ids)

    # Get number of compression tokens
    num_compression_tokens = embedding.shape[1]

    # Compute Hessian eigenvalues
    eigenvalues, eigenvectors = compute_hessian_eigenvalues(
        model,
        embedding.unsqueeze(0),  # Add batch dimension
        input_ids,
        token_embeddings,
        attention_mask,
        num_compression_tokens,
        loss_type=loss_type,
        num_alignment_layers=num_alignment_layers,
        hybrid_alpha=hybrid_alpha,
        num_eigenvalues=num_eigenvalues,
    )

    return {
        "eigenvalues": eigenvalues.numpy(),
        "eigenvectors": eigenvectors.numpy(),
        "stage_index": int(row.get("stage_index", -1)),
        "stage_seq_len": stage_seq_len,
        "sample_id": int(row.get("sample_id", -1)),
    }


def plot_spectrum_evolution(
    all_spectra: List[Dict[str, Any]],
    output_path: str,
    sample_id: Optional[int] = None,
):
    """Plot how the Hessian spectrum evolves across stages."""
    if not all_spectra:
        print("No spectra to plot")
        return

    # Sort by stage_index
    all_spectra = sorted(all_spectra, key=lambda x: x["stage_index"])

    # Extract data
    stage_indices = [s["stage_index"] for s in all_spectra]
    stage_seq_lens = [s["stage_seq_len"] for s in all_spectra]
    eigenvalues_list = [s["eigenvalues"] for s in all_spectra]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Eigenvalue evolution (heatmap)
    max_eigenvalues = max(len(eig) for eig in eigenvalues_list)
    spectrum_matrix = np.zeros((len(all_spectra), max_eigenvalues))
    for i, eig in enumerate(eigenvalues_list):
        spectrum_matrix[i, : len(eig)] = eig

    im = axes[0].imshow(
        spectrum_matrix.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
    )
    axes[0].set_xlabel("Stage Index", fontsize=12)
    axes[0].set_ylabel("Eigenvalue Index (sorted descending)", fontsize=12)
    axes[0].set_title("Hessian Eigenvalue Spectrum Evolution", fontsize=14)
    axes[0].set_xticks(range(len(stage_indices)))
    axes[0].set_xticklabels(
        [f"S{s}\nL={stage_length}" for s, stage_length in zip(stage_indices, stage_seq_lens)], rotation=45, ha="right"
    )
    plt.colorbar(im, ax=axes[0], label="Eigenvalue")

    # Plot 2: Top eigenvalues over stages
    num_top = min(10, max_eigenvalues)
    for idx in range(num_top):
        values = [eig[idx] if idx < len(eig) else 0 for eig in eigenvalues_list]
        axes[1].plot(stage_indices, values, marker="o", label=f"λ{idx+1}", alpha=0.7)

    axes[1].set_xlabel("Stage Index", fontsize=12)
    axes[1].set_ylabel("Eigenvalue", fontsize=12)
    axes[1].set_title(f"Top {num_top} Eigenvalues Across Stages", fontsize=14)
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(stage_indices)
    axes[1].set_xticklabels(
        [f"S{s}\nL={stage_length}" for s, stage_length in zip(stage_indices, stage_seq_lens)], rotation=45, ha="right"
    )

    sample_str = f" (sample_id={sample_id})" if sample_id is not None else ""
    fig.suptitle(f"Hessian Spectrum Evolution{sample_str}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved spectrum evolution plot to: {output_path}")
    plt.close()


def plot_eigenvalue_statistics(
    all_spectra: List[Dict[str, Any]],
    output_path: str,
    sample_id: Optional[int] = None,
):
    """Plot statistics of eigenvalues (max, min, mean, condition number)."""
    if not all_spectra:
        return

    all_spectra = sorted(all_spectra, key=lambda x: x["stage_index"])
    stage_indices = [s["stage_index"] for s in all_spectra]
    eigenvalues_list = [s["eigenvalues"] for s in all_spectra]

    max_eig = [eig[0] if len(eig) > 0 else 0 for eig in eigenvalues_list]
    min_eig = [eig[-1] if len(eig) > 0 else 0 for eig in eigenvalues_list]
    mean_eig = [eig.mean() if len(eig) > 0 else 0 for eig in eigenvalues_list]
    condition_numbers = [eig[0] / eig[-1] if len(eig) > 1 and eig[-1] > 1e-10 else 0 for eig in eigenvalues_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(stage_indices, max_eig, marker="o", label="Max (λ₁)")
    axes[0, 0].set_xlabel("Stage Index")
    axes[0, 0].set_ylabel("Eigenvalue")
    axes[0, 0].set_title("Maximum Eigenvalue")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(stage_indices, min_eig, marker="o", color="orange", label="Min")
    axes[0, 1].set_xlabel("Stage Index")
    axes[0, 1].set_ylabel("Eigenvalue")
    axes[0, 1].set_title("Minimum Eigenvalue")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(stage_indices, mean_eig, marker="o", color="green", label="Mean")
    axes[1, 0].set_xlabel("Stage Index")
    axes[1, 0].set_ylabel("Eigenvalue")
    axes[1, 0].set_title("Mean Eigenvalue")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(stage_indices, condition_numbers, marker="o", color="red", label="κ = λ₁/λₙ")
    axes[1, 1].set_xlabel("Stage Index")
    axes[1, 1].set_ylabel("Condition Number")
    axes[1, 1].set_title("Hessian Condition Number")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale("log")

    sample_str = f" (sample_id={sample_id})" if sample_id is not None else ""
    fig.suptitle(f"Hessian Eigenvalue Statistics{sample_str}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved eigenvalue statistics to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute and visualize Hessian eigenspectrum for progressive training")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to progressive_prefixes dataset",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Optional sample_id filter (if None, process all samples)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Model checkpoint (if None, inferred from artifacts)",
    )
    parser.add_argument(
        "--num_eigenvalues",
        type=int,
        default=20,
        help="Number of eigenvalues to compute",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        help="Loss type used in training (cross_entropy, l2, l1, cosine)",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=0,
        help="Number of alignment layers (0 = all)",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="Weight for alignment loss (if hybrid loss was used)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (if None, inferred from dataset_path)",
    )

    args = parser.parse_args()

    # Determine output directory
    out_dir = args.output_dir
    if out_dir is None:
        dataset_path = args.dataset_path
        if "artifacts/experiments" in dataset_path or "artifacts/experiments_progressive" in dataset_path:
            exp_dir = os.path.dirname(dataset_path)
            out_dir = os.path.join(exp_dir, "hessian_analysis")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join("artifacts/experiments", f"hessian_analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    ds = load_progressive_dataset(args.dataset_path)
    rows = filter_records(ds, sample_id=args.sample_id, stage_index=None)

    if not rows:
        raise ValueError("No records found with given filters.")

    # Determine model checkpoint
    model_checkpoint = args.model_checkpoint
    if model_checkpoint is None:
        # Try to infer from artifacts
        model_checkpoints = [str(r.get("model_checkpoint", "")).strip() for r in rows]
        model_checkpoints = [m for m in model_checkpoints if m]
        if model_checkpoints:
            # Use most common
            from collections import Counter

            model_checkpoint = Counter(model_checkpoints).most_common(1)[0][0]
            print(f"Inferred model_checkpoint: {model_checkpoint}")
        else:
            raise ValueError("Could not infer model_checkpoint. Please provide --model_checkpoint")

    # Load model and tokenizer
    device = get_device()
    print(f"Loading model: {model_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
    model.eval()
    freeze_model_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Group by sample
    by_sid = collate_stages_by_sample(rows)

    # Process each sample
    for sample_id, stage_rows in tqdm(by_sid.items(), desc="Processing samples"):
        print(f"\nProcessing sample_id={sample_id} with {len(stage_rows)} stages")

        all_spectra = []
        for row in tqdm(stage_rows, desc=f"Computing Hessian for sample {sample_id}", leave=False):
            spectrum_data = compute_hessian_for_stage(
                model,
                row,
                tokenizer,
                device,
                num_eigenvalues=args.num_eigenvalues,
                loss_type=args.loss_type,
                num_alignment_layers=args.num_alignment_layers,
                hybrid_alpha=args.hybrid_alpha,
            )
            all_spectra.append(spectrum_data)

        if not all_spectra:
            print(f"No spectra computed for sample {sample_id}")
            continue

        # Save raw data
        np.savez(
            os.path.join(out_dir, f"hessian_spectrum_sample_{sample_id}.npz"),
            **{f"stage_{s['stage_index']}": s["eigenvalues"] for s in all_spectra},
        )

        # Create visualizations
        plot_spectrum_evolution(
            all_spectra,
            os.path.join(out_dir, f"spectrum_evolution_sample_{sample_id}.png"),
            sample_id=sample_id,
        )
        plot_eigenvalue_statistics(
            all_spectra,
            os.path.join(out_dir, f"eigenvalue_statistics_sample_{sample_id}.png"),
            sample_id=sample_id,
        )

    print(f"\nAnalysis complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
