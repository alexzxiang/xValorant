"""
Train the DeepSets RoundPredictor on per-player dataset from build_dataset_deepsets.py.

Usage:
    python scripts/train_deepsets.py \
        --dataset-dir data/deepsets_dataset \
        --output-dir models/deepsets_v1

    # Enable Transformer upgrade:
    python scripts/train_deepsets.py --use-transformer

    # Compare against baseline:
    python scripts/train_deepsets.py --compare-baseline models/xgb_baseline.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("PyTorch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from valoscribe.models.round_predictor import RoundPredictor, ModelConfig
from evaluate_model import brier_score, log_loss, roc_auc, plot_reliability_diagram


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeepSetsDataset(Dataset):
    """
    Loads the per-player tensor arrays produced by build_dataset_deepsets.py.
    """

    def __init__(self, data_dir: Path, split: str):
        d = data_dir
        self.continuous_feats = torch.tensor(np.load(d / f"{split}_continuous_feats.npy"), dtype=torch.float32)
        self.agent_ids         = torch.tensor(np.load(d / f"{split}_agent_ids.npy"),         dtype=torch.long)
        self.weapon_tier_ids   = torch.tensor(np.load(d / f"{split}_weapon_tier_ids.npy"),   dtype=torch.long)
        self.global_feats      = torch.tensor(np.load(d / f"{split}_global_feats.npy"),      dtype=torch.float32)
        self.map_ids           = torch.tensor(np.load(d / f"{split}_map_ids.npy"),           dtype=torch.long)
        self.atk_mask          = torch.tensor(np.load(d / f"{split}_atk_mask.npy"),          dtype=torch.float32)
        self.alive_mask        = torch.tensor(np.load(d / f"{split}_alive_mask.npy"),        dtype=torch.float32)
        self.y                 = torch.tensor(np.load(d / f"{split}_y.npy"),                 dtype=torch.float32)
        self.n = len(self.y)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return (
            self.continuous_feats[idx],
            self.agent_ids[idx],
            self.weapon_tier_ids[idx],
            self.global_feats[idx],
            self.map_ids[idx],
            self.atk_mask[idx],
            self.alive_mask[idx],
            self.y[idx],
        )


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        cont, aids, tids, gfeat, mids, atk, alive, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits = model(cont, aids, tids, gfeat, mids, atk, alive)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in loader:
        cont, aids, tids, gfeat, mids, atk, alive, labels = [b.to(device) for b in batch]
        logits = model(cont, aids, tids, gfeat, mids, atk, alive)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), y_prob, y_true


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DeepSets RoundPredictor")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/deepsets_dataset"))
    parser.add_argument("--output-dir",  type=Path, default=Path("models/deepsets_v1"))
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=2048)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--player-hidden",  type=int, default=128)
    parser.add_argument("--player-layers",  type=int, default=3)
    parser.add_argument("--global-hidden",  type=int, default=256)
    parser.add_argument("--global-layers",  type=int, default=3)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--device",      default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--use-transformer", action="store_true",
                        help="Replace mean-pool with Transformer self-attention")
    parser.add_argument("--n-heads",          type=int, default=4)
    parser.add_argument("--n-transformer-layers", type=int, default=2)
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    print(f"Loading dataset from {args.dataset_dir}...")
    train_ds = DeepSetsDataset(args.dataset_dir, "train")
    val_ds   = DeepSetsDataset(args.dataset_dir, "val")
    test_ds  = DeepSetsDataset(args.dataset_dir, "test")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 4,            num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 4,            num_workers=0)

    # Model
    cfg = ModelConfig(
        player_hidden=args.player_hidden,
        player_layers=args.player_layers,
        global_hidden=args.global_hidden,
        global_layers=args.global_layers,
        dropout=args.dropout,
        use_transformer=args.use_transformer,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_transformer_layers,
    )
    model = RoundPredictor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    arch = "Transformer" if args.use_transformer else "DeepSets"
    print(f"Model: {arch}, {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_brier = float("inf")
    best_epoch = 0
    patience_count = 0

    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience})...")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValBrier':>10}  {'ValAUC':>8}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_brier = brier_score(val_labels, val_probs)
        val_auc   = roc_auc(val_labels, val_probs)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_brier:>10.4f}  {val_auc:>8.4f}")

        if val_brier < best_brier:
            best_brier = val_brier
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
                break

    print(f"\nBest val Brier: {best_brier:.4f} @ epoch {best_epoch}")

    # Save config alongside checkpoint for reproducibility
    with open(args.output_dir / "model_config.json", "w") as f:
        json.dump({
            "arch": arch,
            "player_hidden": args.player_hidden,
            "player_layers": args.player_layers,
            "global_hidden": args.global_hidden,
            "global_layers": args.global_layers,
            "dropout": args.dropout,
            "use_transformer": args.use_transformer,
            "n_params": n_params,
        }, f, indent=2)

    # Reload best and evaluate
    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device))

    print("\n=== Final evaluation ===")
    _, val_probs, val_labels   = eval_epoch(model, val_loader,  criterion, device)
    _, test_probs, test_labels = eval_epoch(model, test_loader, criterion, device)

    val_metrics  = {"brier": brier_score(val_labels, val_probs),
                    "logloss": log_loss(val_labels, val_probs),
                    "auc": roc_auc(val_labels, val_probs)}
    test_metrics = {"brier": brier_score(test_labels, test_probs),
                    "logloss": log_loss(test_labels, test_probs),
                    "auc": roc_auc(test_labels, test_probs)}

    print(f"Val   Brier: {val_metrics['brier']:.4f}  LogLoss: {val_metrics['logloss']:.4f}  AUC: {val_metrics['auc']:.4f}")
    print(f"Test  Brier: {test_metrics['brier']:.4f}  LogLoss: {test_metrics['logloss']:.4f}  AUC: {test_metrics['auc']:.4f}")
    print(f"\nXGBoost baseline (Phase 0): Brier ~0.180, AUC ~0.797")

    plot_reliability_diagram(
        val_labels, val_probs,
        title=f"Reliability — val ({arch})",
        save_path=args.output_dir / "reliability_val.png",
    )
    plot_reliability_diagram(
        test_labels, test_probs,
        title=f"Reliability — test ({arch})",
        save_path=args.output_dir / "reliability_test.png",
    )

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics,
                   "best_epoch": best_epoch, "arch": arch}, f, indent=2)

    print(f"\nCheckpoint: {args.output_dir}/best_model.pt")
    print(f"Config:     {args.output_dir}/model_config.json")
    print(f"Metrics:    {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
