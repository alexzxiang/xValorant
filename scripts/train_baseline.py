"""
Train XGBoost baseline round win predictor.

Uses features from build_dataset.py (no CV — economy + alive counts only).
This is the floor that every subsequent model must beat.

Usage:
    python scripts/train_baseline.py \
        --dataset-dir data/baseline_dataset \
        --output-dir models/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("xgboost not installed. Run: pip install xgboost", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline model")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/baseline_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/baseline"),
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()

    d = args.dataset_dir
    if not d.exists():
        print(f"Dataset directory not found: {d}", file=sys.stderr)
        print("Run build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    # Load data
    X_train = np.load(d / "X_train.npy")
    y_train = np.load(d / "y_train.npy")
    X_val = np.load(d / "X_val.npy")
    y_val = np.load(d / "y_val.npy")
    X_test = np.load(d / "X_test.npy")
    y_test = np.load(d / "y_test.npy")

    with open(d / "feature_names.json") as f:
        feature_names = json.load(f)

    print(f"Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    print(
        f"Attack win rate — train: {y_train.mean():.1%}, "
        f"val: {y_val.mean():.1%}, test: {y_test.mean():.1%}"
    )

    # Train
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    # Evaluate — import relative to scripts/ directory
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_model import evaluate_and_report

    print("\n=== Validation set ===")
    val_metrics = evaluate_and_report(model, X_val, y_val, split_name="val")

    print("\n=== Test set ===")
    test_metrics = evaluate_and_report(model, X_test, y_test, split_name="test")

    # Feature importances
    print("\nFeature importances (gain):")
    importances = model.get_booster().get_score(importance_type="gain")
    feat_importance = sorted(importances.items(), key=lambda x: -x[1])
    for fname, score in feat_importance:
        # XGBoost names features f0, f1, ... unless we set feature names
        idx = int(fname[1:]) if fname.startswith("f") else -1
        display_name = feature_names[idx] if 0 <= idx < len(feature_names) else fname
        print(f"  {display_name:30s} {score:.1f}")

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "xgb_baseline.json"
    model.save_model(str(model_path))

    metrics = {"val": val_metrics, "test": test_metrics}
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(
        f"Val  Brier: {val_metrics['brier']:.4f}  LogLoss: {val_metrics['logloss']:.4f}  AUC: {val_metrics['auc']:.4f}"
    )
    print(
        f"Test Brier: {test_metrics['brier']:.4f}  LogLoss: {test_metrics['logloss']:.4f}  AUC: {test_metrics['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
