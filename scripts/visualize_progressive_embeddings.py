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
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_progressive_dataset(dataset_path: str) -> Dataset:
    return Dataset.load_from_disk(dataset_path)


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
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
        for r in rows[:max_eval_samples]:
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

    for sid, stages in by_sid.items():
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
        plot_cumulative_explained_variance(
            X,
            max_components=16,
            title=f"Sample {sid}: Cumulative Explained Variance",
            outfile=os.path.join(out_dir, f"sid{sid}_cumulative_variance.png"),
        )

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

    if len(ppl_all) > 1:
        plot_correlation(
            np.array(ppl_all),
            np.array(summary_steps[1:]),
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
