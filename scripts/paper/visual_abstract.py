import argparse
import os
from typing import Any, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import seaborn as sns


def _load_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _find_pair_index(pair_indices: np.ndarray, target_pair: Tuple[int, int]) -> int:
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError(f"pair_indices must have shape [num_pairs,2], got {pair_indices.shape}")
    for idx, (i, j) in enumerate(pair_indices.tolist()):
        if (int(i), int(j)) == target_pair:
            return int(idx)
    raise ValueError(f"Pair {target_pair} not found in pair_indices={pair_indices.tolist()}")


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def _scale_from_accuracy(acc: np.ndarray, threshold: float) -> np.ndarray:
    """Map accuracy to [0,1] scale starting at threshold."""
    thr = float(threshold)
    denom = max(1.0 - thr, 1e-12)
    return np.clip((acc - thr) / denom, 0.0, 1.0)


def _estimate_cell_area(grid_x: np.ndarray, grid_y: np.ndarray) -> float:
    """Estimate cell area for a uniform meshgrid."""
    if grid_x.ndim != 2 or grid_y.ndim != 2:
        raise ValueError(f"grid_x/grid_y must be 2D, got {grid_x.ndim}D/{grid_y.ndim}D")
    # Use unique coordinates (robust to float noise) to estimate spacing.
    xs = np.unique(grid_x.reshape(-1))
    ys = np.unique(grid_y.reshape(-1))
    if xs.size < 2 or ys.size < 2:
        return 0.0
    dx = float(np.median(np.diff(np.sort(xs))))
    dy = float(np.median(np.diff(np.sort(ys))))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual-abstract static image from landscape_pca_pairs.npz")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to .npz produced by visualize_landscale_2pca.py")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: alongside npz)")
    parser.add_argument(
        "--skip_first_anchor",
        default=False,
        action="store_true",
        help="If set, skip drawing the first anchor point (and its region/label).",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=None,
        help="Number of anchor points to plot (uniformly sampled). Default: all anchors available in the npz.",
    )
    parser.add_argument(
        "--scatter_alpha",
        type=float,
        default=0.1,
        help="Alpha for background trajectory scatter points",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help='Accuracy threshold for "near-ideal" region (default: 0.9)',
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI")
    args = parser.parse_args()

    # Plot styling
    if sns is not None:
        sns.set_theme(style="whitegrid")

    npz = _load_npz(args.npz_path)

    required = ["pair_indices", "grid_x", "grid_y", "accuracy", "coords", "explained_variance_ratio"]
    missing = [k for k in required if k not in npz]
    if missing:
        raise ValueError(f"Missing keys in npz: {missing}. Available keys: {sorted(npz.keys())}")

    pair_indices = npz["pair_indices"].astype(np.int64)
    pair_idx = _find_pair_index(pair_indices, (0, 1))  # PC1 vs PC2

    coords = npz["coords"].astype(np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must have shape [N,>=2], got {coords.shape}")
    coords_xy = coords[:, :2]

    explained = npz["explained_variance_ratio"].astype(np.float64)
    if explained.ndim != 1 or explained.shape[0] < 2:
        raise ValueError(f"explained_variance_ratio must have shape [>=2], got {explained.shape}")
    ev1 = float(explained[0])
    ev2 = float(explained[1])
    ev_cum_2 = float(ev1 + ev2)
    ev_cum_all = float(np.sum(explained))
    print(f"Explained variance: PC1={ev1:.6f}, PC2={ev2:.6f}, cumulative(PC1+PC2)={ev_cum_2:.6f}")
    print(f"Cumulative explained variance (all fitted PCs) = {ev_cum_all:.6f}")

    # Determine anchors: prefer sampled_indices + sampled_seq_len (multi-frame),
    # otherwise fall back to current_idx/current_seq_len (single frame).
    if "sampled_indices" in npz and "sampled_seq_len" in npz and "anchor_coords" in npz:
        sampled_indices = npz["sampled_indices"].astype(np.int64).reshape(-1)
        # sampled_seq_len = npz["sampled_seq_len"].astype(np.int64).reshape(-1)
        anchor_coords = _ensure_2d(npz["anchor_coords"].astype(np.float32))
        if anchor_coords.shape[0] != sampled_indices.shape[0]:
            raise ValueError(f"anchor_coords rows ({anchor_coords.shape[0]}) != sampled_indices ({sampled_indices.shape[0]})")
        anchor_xy_all = anchor_coords[:, :2]
        # anchor_labels_all = sampled_seq_len  # token prefix length
    elif "current_idx" in npz and "current_seq_len" in npz and "anchor_coords" in npz:
        sampled_indices = np.array([int(npz["current_idx"].reshape(-1)[0])], dtype=np.int64)
        # anchor_labels_all = np.array([int(npz["current_seq_len"].reshape(-1)[0])], dtype=np.int64)
        anchor_coords = _ensure_2d(npz["anchor_coords"].astype(np.float32))
        anchor_xy_all = anchor_coords[:, :2]
    else:
        raise ValueError(
            "NPZ does not contain anchor metadata. Expected either "
            "(sampled_indices+sampled_seq_len+anchor_coords) or (current_idx+current_seq_len+anchor_coords)."
        )

    n_anchors_total = int(anchor_xy_all.shape[0])
    if args.num_anchors is None or args.num_anchors <= 0 or args.num_anchors >= n_anchors_total:
        anchor_sel = np.arange(n_anchors_total, dtype=np.int64)
    else:
        anchor_sel = np.unique(np.linspace(0, n_anchors_total - 1, num=int(args.num_anchors), dtype=np.int64))

    if bool(args.skip_first_anchor) and anchor_sel.size > 0:
        anchor_sel = anchor_sel[1:]

    anchor_xy = anchor_xy_all[anchor_sel]
    # anchor_labels = anchor_labels_all[anchor_sel]
    anchor_indices = sampled_indices[anchor_sel] if sampled_indices.shape[0] >= n_anchors_total else anchor_sel

    # Accuracy surface extraction for PC1-PC2:
    # - single-frame: accuracy has shape [num_pairs,H,W]
    # - multi-frame: accuracy has shape [num_frames,num_pairs,H,W]
    acc = npz["accuracy"]
    grid_x = npz["grid_x"][pair_idx]
    grid_y = npz["grid_y"][pair_idx]
    if grid_x.shape != grid_y.shape:
        raise ValueError(f"grid_x and grid_y shapes differ: {grid_x.shape} vs {grid_y.shape}")

    if acc.ndim == 3:
        # [num_pairs,H,W] -> choose pair -> same surface for all anchors
        acc_per_anchor = np.repeat(acc[pair_idx][None, ...], repeats=n_anchors_total, axis=0)
    elif acc.ndim == 4:
        # [num_frames,num_pairs,H,W] -> align to sampled frames
        if acc.shape[0] != n_anchors_total:
            raise ValueError(f"accuracy frames ({acc.shape[0]}) != anchors ({n_anchors_total})")
        acc_per_anchor = acc[:, pair_idx]
    else:
        raise ValueError(f"Unsupported accuracy ndim={acc.ndim}, shape={acc.shape}")

    # Plot
    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.npz_path), "visual_abstract_pc1_pc2.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 8.5))

    # Anchor colors (gradient)
    if sns is not None:
        # Use a pleasant scientific-looking palette.
        colors = np.array(sns.color_palette("rocket_r", n_colors=max(int(len(anchor_sel)), 1)))
        # Ensure RGBA for later use
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=colors.dtype)], axis=1)
    else:
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.05, 0.95, num=len(anchor_sel)))

    # Near-ideal regions for each anchor (accuracy > threshold) as continuous overlays.
    # Within the region, apply a color/opacity gradient based on accuracy.
    # Draw first (beneath trajectory + anchors).
    thr = float(args.threshold)
    max_region_alpha = 0.8
    cell_area = _estimate_cell_area(grid_x, grid_y)
    near_perfect_area_by_anchor: Dict[int, float] = {}
    for k, (aidx, color) in enumerate(zip(anchor_sel.tolist(), colors)):
        acc_map = acc_per_anchor[int(aidx)]
        mask = acc_map > thr
        near_perfect_area_by_anchor[int(aidx)] = float(mask.sum()) * cell_area if cell_area > 0 else float(mask.sum())
        if not np.any(mask):
            continue
        scale = _scale_from_accuracy(acc_map, threshold=thr)
        alpha_grid = (max_region_alpha * scale * mask.astype(np.float32)).astype(np.float32)

        # Color gradient: keep anchor hue, increase brightness with accuracy.
        rgb = np.clip(color[:3] * (0.35 + 0.65 * scale[..., None]), 0.0, 1.0).astype(np.float32)
        rgba_img = np.zeros((acc_map.shape[0], acc_map.shape[1], 4), dtype=np.float32)
        rgba_img[:, :, :3] = rgb
        rgba_img[:, :, 3] = alpha_grid

        ax.imshow(
            rgba_img,
            origin="lower",
            extent=[
                float(grid_x.min()),
                float(grid_x.max()),
                float(grid_y.min()),
                float(grid_y.max()),
            ],
            interpolation="bilinear",
            zorder=1 + 0.001 * float(k),
        )

    # Background trajectory scatter (draw above regions)
    ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        s=40,
        c="black",
        alpha=float(args.scatter_alpha),
        linewidths=0,
        zorder=2,
    )

    # Anchor markers + labels (on top). Size is proportional to near-ideal area.
    areas = np.array([float(near_perfect_area_by_anchor.get(int(anchor_sel[k]), 0.0)) for k in range(len(anchor_sel))])
    if areas.size > 0 and np.isfinite(areas).any():
        a_min = float(np.nanmin(areas))
        a_max = float(np.nanmax(areas))
    else:
        a_min = 0.0
        a_max = 0.0

    def _size_from_area(area: float) -> float:
        # Map area to marker size range for readability.
        # Use sqrt scaling to compress dynamic range.
        if not np.isfinite(area) or area <= 0 or a_max <= a_min:
            return 110.0
        t = (float(area) - a_min) / (a_max - a_min + 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
        return 90.0 + 260.0 * (t**0.5)

    for k, ((x, y), idx, color) in enumerate(zip(anchor_xy, anchor_indices.tolist(), colors)):
        aidx = int(anchor_sel[k])
        area = float(near_perfect_area_by_anchor.get(aidx, 0.0))
        size = _size_from_area(area)
        ax.scatter(
            [x],
            [y],
            s=size,
            c=[color],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            float(x),
            float(y),
            f"{idx}",
            fontsize=20,
            ha="left",
            va="bottom",
            color="black",
            zorder=6,
        )

    fig_title = f"Visual abstract: PC1-PC2 accuracy regions (>{thr:.2f}), cumulative={ev_cum_2:.3f}"
    print(fig_title)

    ax.set_xlabel(f"PC1 {{{ev1:.3f}}}", fontsize=25)
    ax.set_ylabel(f"PC2 {{{ev2:.3f}}}", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.axis("equal")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
