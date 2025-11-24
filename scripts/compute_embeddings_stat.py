import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def frobenius_l2_norm(t: torch.Tensor) -> float:
    v = t.reshape(-1)
    return float(torch.linalg.norm(v, ord=2).item())


def l1_norm(t: torch.Tensor) -> float:
    v = t.reshape(-1)
    return float(torch.linalg.norm(v, ord=1).item())


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    v = (a - b).reshape(-1)
    return float(torch.linalg.norm(v, ord=2).item())


def _integral_sqrt_quadratic(a: float, b: float, c: float) -> float:
    """Compute definite integral from 0 to 1 of sqrt(a t^2 + b t + c) dt.

    Uses closed-form where a>0, with stable fallbacks when a≈0.
    """
    import math

    eps = 1e-12

    # Guard small negatives due to numerical roundoff
    def _safe_sqrt(x: float) -> float:
        return math.sqrt(max(x, 0.0))

    if abs(a) <= eps:
        # Reduce to integral of sqrt(b t + c)
        if abs(b) <= eps:
            # Integral of sqrt(c) dt from 0..1
            return _safe_sqrt(c)
        else:
            # (2 / (3 b)) * [ (b t + c)^(3/2) ]_0^1
            term1 = b + c
            term0 = c
            return (2.0 / (3.0 * b)) * (max(term1, 0.0) ** 1.5 - max(term0, 0.0) ** 1.5)

    # a > 0 case: use closed form
    sqrt_a = math.sqrt(a)

    def F(t: float) -> float:
        at2btc = a * t * t + b * t + c
        s = _safe_sqrt(at2btc)
        term1 = (2.0 * a * t + b) * s / (4.0 * a)
        k = (4.0 * a * c - b * b) / (8.0 * a**1.5)
        # log argument: 2 sqrt(a) s + 2 a t + b
        log_arg = 2.0 * sqrt_a * s + 2.0 * a * t + b
        # Ensure strictly positive argument for log
        log_arg = max(log_arg, eps)
        term2 = k * math.log(log_arg)
        return term1 + term2

    return F(1.0) - F(0.0)


def quadratic_bezier_arc_length(e0: torch.Tensor, c1: torch.Tensor, e1: torch.Tensor) -> float:
    """Arc length of quadratic Bezier from t in [0,1] in flattened space.

    B(t) = (1-t)^2 e0 + 2(1-t)t c1 + t^2 e1
    L = ∫_0^1 ||B'(t)|| dt
    """
    # Work in double precision on CPU for numerical stability
    x0 = e0.reshape(-1).to(dtype=torch.float64, device="cpu")
    x1 = c1.reshape(-1).to(dtype=torch.float64, device="cpu")
    x2 = e1.reshape(-1).to(dtype=torch.float64, device="cpu")

    A = x1 - x0  # c1 - e0
    C = x2 - 2.0 * x1 + x0  # e1 - 2*c1 + e0

    a = float(torch.dot(C, C).item())
    b = float((2.0 * torch.dot(A, C)).item())
    c = float(torch.dot(A, A).item())

    # dB/dt = 2 * (A + t C); |dB/dt| = 2 * sqrt(a t^2 + b t + c)
    base = _integral_sqrt_quadratic(a, b, c)
    return 2.0 * base


def compute_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    endpoints = params.get("endpoints", {})
    e0 = endpoints.get("e0")
    e1 = endpoints.get("e1")

    if e0 is None or e1 is None:
        raise ValueError("Params file does not contain endpoints 'e0' and 'e1'.")

    # Ensure tensors on CPU and float32
    if not isinstance(e0, torch.Tensor):
        e0 = torch.tensor(e0)
    if not isinstance(e1, torch.Tensor):
        e1 = torch.tensor(e1)
    e0 = e0.to(dtype=torch.float32, device="cpu")
    e1 = e1.to(dtype=torch.float32, device="cpu")

    control_points = params.get("control_points")
    if control_points is None:
        control_points = torch.empty(0)
    elif not isinstance(control_points, torch.Tensor):
        control_points = torch.tensor(control_points)
    control_points = control_points.to(dtype=torch.float32, device="cpu")

    stats: Dict[str, Any] = {
        "model_checkpoint": params.get("model_checkpoint"),
        "bezier_order": int(params.get("bezier_order", 0)),
        "num_compression_tokens": int(params.get("num_compression_tokens", e0.shape[0])),
        "hidden_size": int(params.get("hidden_size", e0.shape[1] if e0.dim() >= 2 else 0)),
        "e0": {
            "l2_norm": frobenius_l2_norm(e0),
            "l1_norm": l1_norm(e0),
        },
        "e1": {
            "l2_norm": frobenius_l2_norm(e1),
            "l1_norm": l1_norm(e1),
        },
        "e0_e1": {
            "l2_distance": l2_distance(e0, e1),
        },
        "control_points": [],
        "quadratic_bezier_arc_length": None,
    }

    if control_points.numel() > 0:
        # control_points shape: [K, C, D]
        K = int(control_points.shape[0])
        cps: List[Dict[str, Any]] = []
        for k in range(K):
            ck = control_points[k]
            cps.append(
                {
                    "index": k,
                    "l2_norm": frobenius_l2_norm(ck),
                    "l1_norm": l1_norm(ck),
                    "l2_distance_to_e0": l2_distance(ck, e0),
                    "l2_distance_to_e1": l2_distance(ck, e1),
                }
            )
        stats["control_points"] = cps

        # If quadratic (one control point), compute exact arc length
        if K == 1:
            try:
                stats["quadratic_bezier_arc_length"] = quadratic_bezier_arc_length(e0, control_points[0], e1)
            except Exception:
                stats["quadratic_bezier_arc_length"] = None

    return stats


def bernstein_bezier_points(e0: torch.Tensor, control_points: torch.Tensor, e1: torch.Tensor, ts: np.ndarray) -> torch.Tensor:
    """Evaluate general Bezier curve of order n with endpoints e0,e1 and internal control points.

    e0: [C, D]
    control_points: [K, C, D] where K = n-1 (can be 0)
    e1: [C, D]
    ts: numpy array of shape [T] in [0,1]
    returns: [T, C, D] tensor on CPU, float32
    """
    device = torch.device("cpu")
    e0 = e0.to(dtype=torch.float32, device=device)
    e1 = e1.to(dtype=torch.float32, device=device)
    if control_points is None:
        control_points = torch.empty(0, *e0.shape, dtype=torch.float32, device=device)
    else:
        control_points = control_points.to(dtype=torch.float32, device=device)

    points: List[torch.Tensor] = [e0]
    if control_points.numel() > 0:
        points.extend([control_points[i] for i in range(control_points.shape[0])])
    points.append(e1)
    P = torch.stack(points, dim=0)  # [n+1, C, D]
    n = P.shape[0] - 1

    t = torch.from_numpy(ts.astype(np.float32))
    T = t.shape[0]
    one_minus_t = 1.0 - t

    # Compute Bernstein coefficients for each i=0..n
    coeffs = []
    for i in range(n + 1):
        binom = float(np.math.comb(n, i))
        coeffs.append(binom * (one_minus_t ** (n - i)) * (t**i))
    coeffs_t = torch.stack(coeffs, dim=1)  # [T, n+1]
    ct = (coeffs_t.view(T, n + 1, 1, 1) * P.view(1, n + 1, *P.shape[1:])).sum(dim=1)  # [T, C, D]
    return ct.to(dtype=torch.float32, device=torch.device("cpu"))


def plot_bezier_pca_projection(
    e0: torch.Tensor,
    control_points: torch.Tensor,
    e1: torch.Tensor,
    out_path: str,
    num_points: int = 200,
    other_endpoints: Optional[List[Tuple[str, torch.Tensor]]] = None,
):
    # Sample curve points
    ts = np.linspace(0.0, 1.0, int(max(2, num_points)), dtype=np.float32)
    curve = bernstein_bezier_points(e0, control_points, e1, ts)  # [T, C, D]

    # Collect anchor points for labeling
    anchors: List[torch.Tensor] = [e0]
    if control_points is not None and control_points.numel() > 0:
        for i in range(control_points.shape[0]):
            anchors.append(control_points[i])
    anchors.append(e1)

    # Additional endpoints from other params files
    other_endpoints = other_endpoints or []
    other_anchors: List[torch.Tensor] = [p for (_, p) in other_endpoints]
    other_labels: List[str] = [name for (name, _) in other_endpoints]

    # Build matrix of points to fit PCA: curve samples + anchors
    curve_flat = curve.reshape(curve.shape[0], -1).numpy()
    anchors_flat = torch.stack([a.reshape(-1) for a in anchors], dim=0).numpy()
    if len(other_anchors) > 0:
        print("pca all with anchor")
        other_flat = torch.stack([a.reshape(-1) for a in other_anchors], dim=0).numpy()
        X = np.concatenate([curve_flat, anchors_flat, other_flat], axis=0)
    else:
        X = np.concatenate([curve_flat, anchors_flat], axis=0)
    if X.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    K = curve.shape[0]
    XY_curve = XY[:K]
    XY_anchors = XY[K : K + len(anchors)]
    XY_other = XY[K + len(anchors) :] if len(other_anchors) > 0 else np.zeros((0, 2), dtype=np.float32)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(XY_curve[:, 0], XY_curve[:, 1], color="tab:blue", linewidth=2, label="Bezier curve")
    # Anchors: e0, control points, e1
    labels = (
        ["e0"] + [f"cp{i}" for i in range(control_points.shape[0])] + ["e1"] if control_points is not None else ["e0", "e1"]
    )
    colors = ["tab:green"] + ["tab:orange"] * (len(labels) - 2) + ["tab:red"]
    for i, (x, y) in enumerate(XY_anchors):
        plt.scatter([x], [y], s=80, color=colors[i % len(colors)], zorder=3, alpha=0.5)
        plt.text(x, y, labels[i], fontsize=8, ha="left", va="bottom")

    # Plot other endpoints (from other params files)
    if XY_other.shape[0] > 0:
        plt.scatter(XY_other[:, 0], XY_other[:, 1], s=40, color="tab:purple", marker="x", label="other endpoints", zorder=2)
        # annotate lightly
        for (x, y), name in zip(XY_other, other_labels):
            plt.text(float(x), float(y), name, fontsize=7, ha="left", va="bottom", color="tab:purple")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Bezier curve projection (PCA)")
    plt.xlim(-28.5, -27.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_and_save_pairwise_l2(endpoints: List[Tuple[str, torch.Tensor]], csv_path: str) -> None:
    labels = [name for (name, _) in endpoints]
    vecs = [t.reshape(-1).to(dtype=torch.float32, device="cpu").numpy() for (_, t) in endpoints]
    X = np.stack(vecs, axis=0) if len(vecs) > 0 else np.zeros((0, 0), dtype=np.float32)
    with open(csv_path, "w") as f:
        f.write("i,j,label_i,label_j,l2\n")
        n = X.shape[0]
        for i in range(n):
            for j in range(n):
                d = float(np.linalg.norm(X[i] - X[j])) if n > 0 else 0.0
                f.write(f"{i},{j},{labels[i]},{labels[j]},{d:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embedding stats from saved interpolation params (.pt)")
    parser.add_argument("--params_path", type=str, required=True, help="Path to saved params .pt file")
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, save stats as JSON next to the params file",
    )
    parser.add_argument(
        "--plot_curve",
        action="store_true",
        help="If set, plot PCA projection of Bezier curve using endpoints and control points",
    )
    parser.add_argument("--num_points", type=int, default=200, help="Samples along the curve for plotting")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to save plots")
    parser.add_argument(
        "--other_params_paths",
        type=str,
        nargs="*",
        default=None,
        help="Additional .pt files; their e0/e1 endpoints will be overlaid on the projection",
    )
    args = parser.parse_args()

    params_path = os.path.abspath(args.params_path)
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")

    params = torch.load(params_path, map_location="cpu")
    stats = compute_stats(params)

    # Print concise human-readable report
    print(f"File: {params_path}")
    if stats.get("model_checkpoint"):
        print(f"Model: {stats['model_checkpoint']}")
    print(f"Order: {stats['bezier_order']}")
    print(f"C (tokens): {stats['num_compression_tokens']}, D (hidden): {stats['hidden_size']}")
    print("--- Endpoints ---")
    print(f"e0: L2={stats['e0']['l2_norm']:.6f}, L1={stats['e0']['l1_norm']:.6f}")
    print(f"e1: L2={stats['e1']['l2_norm']:.6f}, L1={stats['e1']['l1_norm']:.6f}")
    print(f"L2 distance (e0,e1): {stats['e0_e1']['l2_distance']:.6f}")

    cps: List[Dict[str, Any]] = stats.get("control_points", [])
    if len(cps) > 0:
        print("--- Control points ---")
        for cp in cps:
            print(
                f"cp[{cp['index']}]: L2={cp['l2_norm']:.6f}, L1={cp['l1_norm']:.6f}, "
                f"d2e0={cp['l2_distance_to_e0']:.6f}, d2e1={cp['l2_distance_to_e1']:.6f}"
            )
        if stats.get("quadratic_bezier_arc_length") is not None:
            print(f"Quadratic Bezier arc length (integral): {stats['quadratic_bezier_arc_length']:.6f}")
    else:
        print("No control points present.")

    if args.save_json:
        out_path = os.path.splitext(params_path)[0] + "_embedding_stats.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved JSON: {out_path}")

    # Optional plot
    if args.plot_curve:
        endpoints = params.get("endpoints", {})
        e0 = endpoints.get("e0")
        e1 = endpoints.get("e1")
        if e0 is None or e1 is None:
            raise ValueError("Endpoints e0/e1 not found in params; cannot plot curve")
        if not isinstance(e0, torch.Tensor):
            e0 = torch.tensor(e0)
        if not isinstance(e1, torch.Tensor):
            e1 = torch.tensor(e1)
        cps = params.get("control_points", None)
        if cps is not None and not isinstance(cps, torch.Tensor):
            cps = torch.tensor(cps)
        # Load endpoints from other params files, if provided
        other_eps: List[Tuple[str, torch.Tensor]] = []
        if args.other_params_paths:
            for opath in args.other_params_paths:
                other = torch.load(os.path.abspath(os.path.join(opath, "bezier_params_sid0.pt")), map_location="cpu")
                o_end = other.get("endpoints", {})
                oe0 = o_end.get("e0")
                oe1 = o_end.get("e1")
                if oe0 is None or oe1 is None:
                    continue
                if not isinstance(oe0, torch.Tensor):
                    oe0 = torch.tensor(oe0)
                if not isinstance(oe1, torch.Tensor):
                    oe1 = torch.tensor(oe1)
                base = os.path.splitext(os.path.basename(opath))[0]
                other_eps.append((f"e0:{base}", oe0.to(dtype=torch.float32)))
                other_eps.append((f"e1:{base}", oe1.to(dtype=torch.float32)))

        # Determine output path
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(params_path))[0]
            proj_path = os.path.join(args.output_dir, f"{base}_projection.png")
        else:
            proj_path = os.path.splitext(params_path)[0] + "_projection.png"
        plot_bezier_pca_projection(
            e0,
            cps,
            e1,
            proj_path,
            num_points=int(args.num_points),
            other_endpoints=other_eps,
        )
        print(f"Saved projection: {proj_path}")

        # Pairwise L2 distances among endpoints (main + others)
        endpoint_list: List[Tuple[str, torch.Tensor]] = [
            ("e0:main", e0.to(dtype=torch.float32)),
            ("e1:main", e1.to(dtype=torch.float32)),
        ]
        endpoint_list.extend(other_eps)
        pairwise_csv = (
            os.path.join(args.output_dir, os.path.splitext(os.path.basename(params_path))[0] + "_pairwise_endpoints_l2.csv")
            if args.output_dir is not None
            else os.path.splitext(params_path)[0] + "_pairwise_endpoints_l2.csv"
        )
        compute_and_save_pairwise_l2(endpoint_list, pairwise_csv)
        print(f"Saved pairwise L2 distances: {pairwise_csv}")


if __name__ == "__main__":
    main()
