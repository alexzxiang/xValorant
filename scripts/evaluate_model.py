"""
Evaluation utilities for round win probability models.

Computes Brier score, log loss, AUC, and plots a reliability diagram.
Designed to be importable by both train_baseline.py and train_model.py.

Usage (standalone):
    python scripts/evaluate_model.py \
        --model-path models/baseline/xgb_baseline.json \
        --dataset-dir data/baseline_dataset \
        --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_prob))
    except ImportError:
        # Manual trapezoidal AUC
        sorted_idx = np.argsort(-y_prob)
        y_sorted = y_true[sorted_idx]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / (tp[-1] + 1e-9)
        fpr = fp / (fp[-1] + 1e-9)
        return float(np.trapz(tpr, fpr))


def reliability_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration data for a reliability diagram.

    Returns (bin_midpoints, mean_predicted, fraction_positive) arrays of length n_bins.
    Bins with no samples have NaN values.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    mean_predicted = np.full(n_bins, np.nan)
    fraction_positive = np.full(n_bins, np.nan)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            mean_predicted[i] = y_prob[mask].mean()
            fraction_positive[i] = y_true[mask].mean()

    return bin_midpoints, mean_predicted, fraction_positive


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Reliability Diagram",
    save_path: Path | None = None,
    n_bins: int = 10,
) -> None:
    """
    Plot a reliability diagram (calibration curve).
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    _, mean_pred, frac_pos = reliability_data(y_true, y_prob, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax = axes[0]
    valid = ~np.isnan(mean_pred)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred[valid], frac_pos[valid], "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Right: prediction histogram
    ax2 = axes[1]
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Attack won", color="green")
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Defense won", color="red")
    ax2.set_xlabel("Predicted P(attack wins)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Reliability diagram saved to {save_path}")
    else:
        plt.show()

    plt.close()


def evaluate_and_report(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "eval",
    plot_dir: Path | None = None,
) -> dict[str, float]:
    """
    Run inference, compute metrics, print report, optionally save diagram.

    Works with any model that has a predict_proba(X) method returning (n, 2) array,
    or a predict(X) method returning probabilities directly.
    """
    # Get probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2:
            y_prob = probs[:, 1].astype(np.float64)
        else:
            y_prob = probs.astype(np.float64)
    else:
        y_prob = model.predict(X).astype(np.float64)

    y_true = y.astype(np.float64)

    metrics = {
        "brier": brier_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob),
        "auc": roc_auc(y_true, y_prob),
        "n_samples": int(len(y)),
        "atk_win_rate": float(y_true.mean()),
    }

    print(
        f"[{split_name}] n={metrics['n_samples']:,}  "
        f"Brier: {metrics['brier']:.4f}  "
        f"LogLoss: {metrics['logloss']:.4f}  "
        f"AUC: {metrics['auc']:.4f}  "
        f"(atk_win_rate={metrics['atk_win_rate']:.1%})"
    )

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagram(
            y_true,
            y_prob,
            title=f"Reliability Diagram — {split_name}",
            save_path=plot_dir / f"reliability_{split_name}.png",
        )

    return metrics


# ─── Standalone CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=["xgb", "torch"], default="xgb")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/baseline_dataset"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--plot-dir", type=Path, default=None)
    args = parser.parse_args()

    X = np.load(args.dataset_dir / f"X_{args.split}.npy")
    y = np.load(args.dataset_dir / f"y_{args.split}.npy")

    if args.model_type == "xgb":
        try:
            import xgboost as xgb
        except ImportError:
            print("xgboost not installed", file=sys.stderr)
            sys.exit(1)
        model = xgb.XGBClassifier()
        model.load_model(str(args.model_path))
    else:
        print("torch model evaluation not yet implemented in standalone mode")
        sys.exit(1)

    evaluate_and_report(
        model, X, y,
        split_name=args.split,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
