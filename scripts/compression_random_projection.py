import argparse
import os
from typing import Any, Dict, Tuple
from tqdm.auto import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.compute_embeddings_stat import compute_stats


def _prepare_model(model_name: str, device: torch.device):
    print("Preparing model", model_name, device)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def _tokenize_and_embed(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    inputs_embeds = model.model.embed_tokens(input_ids)
    return input_ids, attention_mask, inputs_embeds


@torch.inference_mode()
def _compute_convergence(
    model: AutoModelForCausalLM,
    compression_tokens: torch.Tensor,  # [B, C, D]
    inputs_embeds: torch.Tensor,  # [1, T, D]
    attention_mask: torch.Tensor,  # [1, T]
    input_ids: torch.Tensor,  # [1, T]
) -> np.ndarray:
    # Support batched compression tokens and expand text inputs to match batch
    B = compression_tokens.shape[0]
    C = compression_tokens.shape[1]
    comp_mask = torch.ones((B, C), dtype=attention_mask.dtype, device=attention_mask.device)
    inputs_b = inputs_embeds.expand(B, -1, -1)
    attn_b = attention_mask.expand(B, -1)
    ids_b = input_ids.expand(B, -1)
    x = torch.cat([compression_tokens, inputs_b], dim=1)
    m = torch.cat([comp_mask, attn_b], dim=1)
    out = model(inputs_embeds=x, attention_mask=m)
    preds = out.logits[:, 0:-1].argmax(dim=-1)
    denom = attn_b.sum(dim=-1).clamp(min=1)
    conv = (preds == ids_b[:, :]).sum(dim=-1) / denom
    return conv.detach().cpu().numpy()


def _select_anchor(params: Dict[str, Any], which: str) -> torch.Tensor:
    endpoints = params.get("endpoints", {})
    e0 = endpoints.get("e0")
    e1 = endpoints.get("e1")
    cps = params.get("control_points", None)
    if isinstance(e0, np.ndarray):
        e0 = torch.tensor(e0)
    if isinstance(e1, np.ndarray):
        e1 = torch.tensor(e1)
    if cps is not None and isinstance(cps, np.ndarray):
        cps = torch.tensor(cps)

    if which == "e0":
        if e0 is None:
            raise ValueError("e0 not found in params")
        return e0
    if which == "e1":
        if e1 is None:
            raise ValueError("e1 not found in params")
        return e1
    if which.startswith("cp"):
        if cps is None or cps.numel() == 0:
            raise ValueError("No control_points in params to select from")
        idx = int(which[2:]) if len(which) > 2 else 0
        if idx < 0 or idx >= cps.shape[0]:
            raise ValueError(f"Control point index out of range: {idx}")
        return cps[idx]
    raise ValueError(f"Unknown anchor specifier: {which}")


def _sample_unit_directions(num: int, dim: int, device: torch.device) -> torch.Tensor:
    v = torch.randn(num, dim, device=device)
    v = v / (v.norm(dim=1, keepdim=True).clamp(min=1e-12))
    return v


@torch.inference_mode()
def evaluate_random_projection(
    model: AutoModelForCausalLM,
    anchor: torch.Tensor,  # [C, D]
    radii: np.ndarray,  # [R]
    num_directions: int,
    inputs_embeds: torch.Tensor,  # [1, T, D]
    attention_mask: torch.Tensor,  # [1, T]
    input_ids: torch.Tensor,  # [1, T]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = inputs_embeds.device
    C, D = anchor.shape
    dim = C * D
    dirs = _sample_unit_directions(num_directions, dim, device).view(num_directions, C, D)
    R = len(radii)
    accs = np.zeros((num_directions, R), dtype=np.float32)
    base = anchor.to(device=device, dtype=torch.float32)
    for j, r in enumerate(tqdm(radii)):
        ct_batch = base.unsqueeze(0) + float(r) * dirs  # [num_directions, C, D]
        convs = _compute_convergence(model, ct_batch, inputs_embeds, attention_mask, input_ids)  # [num_directions]
        accs[:, j] = convs
    mean_acc = accs.mean(axis=0)
    std_acc = accs.std(axis=0)
    return accs, mean_acc, std_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Random projection accuracy around a compression embedding point")
    parser.add_argument("--params_path", type=str, required=True, help="Path to saved params .pt file (from interpolation)")
    parser.add_argument("--anchor", type=str, default="e1", help="Which anchor to use: e0 | e1 | cp{k}")
    parser.add_argument("--num_directions", type=int, default=16, help="Number of random directions")
    parser.add_argument("--num_radii", type=int, default=25, help="Number of radii to sample from 0..max_radius")
    parser.add_argument("--max_radius", type=float, default=None, help="Maximum radius; default=0.5*||e1-e0||")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Override model; otherwise read from params")
    parser.add_argument("--output_dir", type=str, default="/tmp", help="Directory to save plot/CSV; defaults next to params")
    parser.add_argument("--save_csv", action="store_true", help="If set, also save CSV of per-direction accuracies")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    params_path = os.path.abspath(args.params_path)
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")

    params = torch.load(params_path, map_location="cpu")
    stats = compute_stats(params)

    model_name = args.model_checkpoint or stats.get("model_checkpoint") or params.get("model_checkpoint")
    if not model_name:
        raise ValueError("Model checkpoint not found; pass --model_checkpoint or include in params")

    text_eval = params.get("text_eval")
    if text_eval is None or str(text_eval).strip() == "":
        raise ValueError("params must contain 'text_eval' used for evaluation")

    # Select anchor embedding
    anchor = _select_anchor(params, args.anchor)
    if not isinstance(anchor, torch.Tensor):
        anchor = torch.tensor(anchor)
    anchor = anchor.to(dtype=torch.float32)
    if anchor.dim() != 2:
        raise ValueError(f"Anchor must be [C, D], got {tuple(anchor.shape)}")

    # Default max radius = ||e1 - e0||
    if args.max_radius is not None:
        max_r = float(args.max_radius)
    else:
        e0 = params.get("endpoints", {}).get("e0")
        e1 = params.get("endpoints", {}).get("e1")
        if not isinstance(e0, torch.Tensor):
            e0 = torch.tensor(e0)
        if not isinstance(e1, torch.Tensor):
            e1 = torch.tensor(e1)
        max_r = float(torch.linalg.norm((e1 - e0).reshape(-1)).item())

    max_r = max(1e-6, max_r)

    radii = (np.linspace(0.0, max_r, int(max(2, args.num_radii))),)
    # Radii comes as a 1-tuple due to trailing comma above; unwrap
    radii = radii[0]

    # Prepare model and inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _prepare_model(model_name, device)
    input_ids, attention_mask, inputs_embeds = _tokenize_and_embed(model, tokenizer, str(text_eval), device)

    # Evaluate
    accs, mean_acc, std_acc = evaluate_random_projection(
        model=model,
        anchor=anchor.to(device),
        radii=radii,
        num_directions=int(args.num_directions),
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        input_ids=input_ids,
    )

    # Outputs
    out_dir = args.output_dir or os.path.dirname(params_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(params_path))[0]

    # Plot mean ± std and faint per-direction curves
    plt.figure(figsize=(7, 4))
    for i in range(accs.shape[0]):
        plt.plot(radii, accs[i], color="tab:blue", alpha=0.2, linewidth=1)
    plt.plot(radii, mean_acc, color="tab:red", linewidth=2, label="mean")
    plt.fill_between(radii, mean_acc - std_acc, mean_acc + std_acc, color="tab:red", alpha=0.2, label="std")
    plt.xlabel("radius")
    plt.ylabel("convergence accuracy")
    plt.title(f"Random projection around {args.anchor}")
    plt.xlim(0, 3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{base}_randproj_{args.anchor}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    if args.save_csv:
        import csv

        csv_path = os.path.join(out_dir, f"{base}_randproj_{args.anchor}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["radius", "mean_acc", "std_acc"] + [f"dir_{i}" for i in range(accs.shape[0])]
            writer.writerow(header)
            for j in range(len(radii)):
                row = [float(radii[j]), float(mean_acc[j]), float(std_acc[j])] + [
                    float(accs[i, j]) for i in range(accs.shape[0])
                ]
                writer.writerow(row)
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
