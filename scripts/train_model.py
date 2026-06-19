"""
Train per-map round win predictors.

One model is trained per map. If --dataset-dir contains map subdirectories
(each with train.npz), all maps are trained automatically. To train a single
map, point --dataset-dir at that map's subdirectory directly.

Two model types:
  deepsetsv1 (default) — full per-player DeepSets model (RoundPredictor).
  baseline             — simple MLP on aggregate features only.

Usage:
    # Train all maps found under the dataset dir
    python scripts/train_model.py \\
        --dataset-dir data/dataset_masters_london \\
        --model deepsetsv1 \\
        --output-dir models/masters_london

    # Train a single map
    python scripts/train_model.py \\
        --dataset-dir data/dataset_masters_london/ascent \\
        --model deepsetsv1 \\
        --output-dir models/masters_london/ascent
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


# ── Datasets ───────────────────────────────────────────────────────────────────

class DeepSetsDataset(Dataset):
    """Loads per-player tensors from a .npz file for the RoundPredictor."""

    def __init__(self, npz_path: Path):
        data = dict(np.load(npz_path))
        self.player_feats = torch.tensor(data["player_feats"], dtype=torch.float32)
        self.agent_ids = torch.tensor(data["agent_ids"], dtype=torch.long)
        self.weapon_ids = torch.tensor(data["weapon_ids"], dtype=torch.long)
        self.role_ids = torch.tensor(data["role_ids"], dtype=torch.long)
        self.global_feats = torch.tensor(data["global_feats"], dtype=torch.float32)
        self.atk_mask = torch.tensor(data["atk_mask"], dtype=torch.float32)
        self.alive_mask = torch.tensor(data["alive_mask"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.player_feats[idx],
            self.agent_ids[idx],
            self.weapon_ids[idx],
            self.role_ids[idx],
            self.global_feats[idx],
            self.atk_mask[idx],
            self.alive_mask[idx],
            self.y[idx],
        )


class AggregateDataset(Dataset):
    """Loads aggregate features only — used by the baseline MLP."""

    def __init__(self, npz_path: Path):
        data = dict(np.load(npz_path))
        self.X = torch.tensor(data["agg_feats"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Baseline MLP ───────────────────────────────────────────────────────────────

class BaselineMLPModel(nn.Module):
    """Simple MLP on aggregate features — the neural-network baseline floor."""

    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
            layers += [
                nn.Linear(d_in, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Attack/defense flip augmentation ──────────────────────────────────────────
#
# Flipping which team is "attack" gives a free second labeled sample from every
# frame: swap all atk/def global features, negate score_diff and atk_is_team1,
# flip atk_mask, and invert the label.
#
# Permutation index i says "new feature i comes from old feature perm[i]".
# Order matches GLOBAL_FEAT_NAMES in build_dataset.py (31 dims):
#   0=atk_is_team1, 1=round_norm, 2=score_diff, 3=time_norm, 4=spike_planted,
#   5=spike_time_norm, 6=atk_econ, 7=def_econ, 8=atk_alive, 9=def_alive,
#   10=atk_ult, 11=def_ult, 12=atk_rifle, 13=def_rifle, 14=atk_health, 15=def_health,
#   16=atk_loadout, 17=def_loadout, 18=ctrl_atk, 19=ctrl_def,
#   20=init_atk, 21=init_def, 22=atk_pos_cov, 23=def_pos_cov,
#   24=atk_cx, 25=atk_cy, 26=def_cx, 27=def_cy, 28=atk_spread, 29=def_spread,
#   30=min_cross_dist
_GLOBAL_FLIP_PERM = [
    0, 1, 2, 3, 4, 5,          # keep indices (signs handled separately)
    7, 6,                       # atk_econ <-> def_econ
    9, 8,                       # atk_alive <-> def_alive
    11, 10,                     # atk_ult <-> def_ult
    13, 12,                     # atk_rifle <-> def_rifle
    15, 14,                     # atk_health <-> def_health
    17, 16,                     # atk_loadout <-> def_loadout
    19, 18,                     # ctrl_atk <-> ctrl_def
    21, 20,                     # init_atk <-> init_def
    23, 22,                     # atk_pos_cov <-> def_pos_cov
    26, 27, 24, 25,             # centroid x/y: atk <-> def
    29, 28,                     # atk_spread <-> def_spread
    30,                         # min_cross_dist (symmetric)
]
_FLIP_PERM_T: torch.Tensor | None = None


def _get_flip_perm(device: torch.device) -> torch.Tensor:
    global _FLIP_PERM_T
    if _FLIP_PERM_T is None or _FLIP_PERM_T.device != device:
        _FLIP_PERM_T = torch.tensor(_GLOBAL_FLIP_PERM, dtype=torch.long, device=device)
    return _FLIP_PERM_T


def flip_atk_def(
    gf: torch.Tensor,      # (B, 31)
    atk_mask: torch.Tensor, # (B, 10)
    y: torch.Tensor,        # (B,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (gf_flipped, atk_mask_flipped, y_flipped) with atk/def swapped."""
    perm = _get_flip_perm(gf.device)
    gf_f = gf[:, perm].clone()
    gf_f[:, 0] = 1.0 - gf_f[:, 0]   # atk_is_team1
    gf_f[:, 2] = -gf_f[:, 2]         # score_diff (atk-def -> def-atk)
    return gf_f, 1.0 - atk_mask, 1.0 - y


# ── Training helpers ───────────────────────────────────────────────────────────

def train_epoch_deepsets(model, loader, optimizer, criterion, device, augment_flip: bool = False):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        pf, aid, wtid, rid, gf, am, alm, y = [t.to(device) for t in batch]

        if augment_flip:
            gf_f, am_f, y_f = flip_atk_def(gf, am, y)
            pf  = torch.cat([pf,  pf],  dim=0)
            aid = torch.cat([aid, aid], dim=0)
            wtid = torch.cat([wtid, wtid], dim=0)
            rid = torch.cat([rid, rid], dim=0)
            gf  = torch.cat([gf,  gf_f], dim=0)
            am  = torch.cat([am,  am_f], dim=0)
            alm = torch.cat([alm, alm], dim=0)
            y   = torch.cat([y,   y_f],  dim=0)

        optimizer.zero_grad()
        logits = model(pf, aid, wtid, rid, gf, am, alm)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n_samples += len(y)
    return total_loss / n_samples


def train_epoch_baseline(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_deepsets(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in loader:
        pf, aid, wtid, rid, gf, am, alm, y = [t.to(device) for t in batch]
        logits = model(pf, aid, wtid, rid, gf, am, alm)
        total_loss += criterion(logits, y).item() * len(y)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), y_prob, y_true


@torch.no_grad()
def eval_baseline(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), y_prob, y_true


# ── Main ───────────────────────────────────────────────────────────────────────

def discover_map_dirs(dataset_dir: Path) -> list[tuple[str, Path]]:
    """
    Return (map_name, map_dir) pairs to train.

    If dataset_dir itself contains train.npz it's a single-map dir.
    Otherwise scan subdirectories for per-map dirs produced by build_dataset.py.
    """
    if (dataset_dir / "train.npz").exists():
        return [(dataset_dir.name, dataset_dir)]
    found = []
    for subdir in sorted(dataset_dir.iterdir()):
        if subdir.is_dir() and (subdir / "train.npz").exists():
            found.append((subdir.name, subdir))
    return found


def train_one_map(
    map_name: str,
    dataset_dir: Path,
    output_dir: Path,
    args,
    device: torch.device,
) -> dict:
    """Train (or skip) one map. Returns metrics dict."""
    use_deepsets = args.model == "deepsetsv1"

    train_path = dataset_dir / "train.npz"
    val_path = dataset_dir / "val.npz"
    test_path = dataset_dir / "test.npz"

    if use_deepsets:
        train_ds = DeepSetsDataset(train_path)
        val_ds = DeepSetsDataset(val_path) if val_path.exists() and np.load(val_path)["y"].size > 0 else None
        test_ds = DeepSetsDataset(test_path) if test_path.exists() and np.load(test_path)["y"].size > 0 else None
    else:
        train_ds = AggregateDataset(train_path)
        val_ds = AggregateDataset(val_path) if val_path.exists() and np.load(val_path)["y"].size > 0 else None
        test_ds = AggregateDataset(test_path) if test_path.exists() and np.load(test_path)["y"].size > 0 else None

    val_str = f"{len(val_ds):,}" if val_ds else "0 (no val split)"
    test_str = f"{len(test_ds):,}" if test_ds else "0 (no test split)"
    print(f"  Train: {len(train_ds):,}  Val: {val_str}  Test: {test_str}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, num_workers=0) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, num_workers=0) if test_ds else None

    if use_deepsets:
        global_feat_dim = int(train_ds.global_feats.shape[-1])
        config = ModelConfig(
            player_hidden=args.player_hidden,
            player_layers=args.player_layers,
            global_hidden=args.global_hidden,
            global_layers=args.global_layers,
            dropout=args.dropout,
            use_transformer=args.use_transformer,
            n_heads=2,
            n_transformer_layers=1,
            use_spatial_bias=args.use_spatial_bias,
            global_feat_dim=global_feat_dim,
        )
        model = RoundPredictor(config).to(device)
        train_fn = train_epoch_deepsets
        eval_fn = eval_deepsets
    else:
        in_dim = train_ds[0][0].shape[0]
        model = BaselineMLPModel(in_dim, hidden=args.global_hidden,
                                 n_layers=args.global_layers, dropout=args.dropout).to(device)
        train_fn = train_epoch_baseline
        eval_fn = eval_baseline

    n_params = sum(p.numel() for p in model.parameters())
    if use_deepsets:
        arch = "Transformer+SpatialBias" if (args.use_transformer and args.use_spatial_bias) \
               else ("Transformer" if args.use_transformer else "DeepSets")
        print(f"  Arch: {arch} | global_feat_dim={global_feat_dim} | params={n_params:,}")
    else:
        print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    output_dir.mkdir(parents=True, exist_ok=True)

    best_brier = float("inf")
    best_epoch = 0
    patience_count = 0

    print(
        f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValBrier':>10}  {'ValAUC':>8}"
    )

    for epoch in range(1, args.epochs + 1):
        if use_deepsets:
            train_loss = train_fn(model, train_loader, optimizer, criterion, device,
                                  augment_flip=args.augment_flip)
        else:
            train_loss = train_fn(model, train_loader, optimizer, criterion, device)

        if val_loader is not None:
            val_loss, val_probs, val_labels = eval_fn(model, val_loader, criterion, device)
            val_brier = brier_score(val_labels, val_probs)
            val_auc = roc_auc(val_labels, val_probs)
            print(f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_brier:>10.4f}  {val_auc:>8.4f}")

            if val_brier < best_brier:
                best_brier = val_brier
                best_epoch = epoch
                patience_count = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} (best={best_epoch})")
                    break
        else:
            # No val set — train for all epochs, save final
            print(f"  {epoch:>6}  {train_loss:>10.4f}  {'—':>10}  {'—':>10}  {'—':>8}")
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            best_epoch = epoch

        scheduler.step()

    # Reload best weights
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    val_metrics: dict = {}
    test_metrics: dict = {}

    if val_loader is not None:
        _, val_probs, val_labels = eval_fn(model, val_loader, criterion, device)
        val_metrics = {
            "brier": brier_score(val_labels, val_probs),
            "logloss": log_loss(val_labels, val_probs),
            "auc": roc_auc(val_labels, val_probs),
        }
        plot_reliability_diagram(
            val_labels, val_probs,
            title=f"Reliability — {map_name} val ({args.model})",
            save_path=output_dir / "reliability_val.png",
        )
        print(f"\n  Val   Brier: {val_metrics['brier']:.4f}  LogLoss: {val_metrics['logloss']:.4f}  AUC: {val_metrics['auc']:.4f}")

    if test_loader is not None:
        _, test_probs, test_labels = eval_fn(model, test_loader, criterion, device)
        test_metrics = {
            "brier": brier_score(test_labels, test_probs),
            "logloss": log_loss(test_labels, test_probs),
            "auc": roc_auc(test_labels, test_probs),
        }
        plot_reliability_diagram(
            test_labels, test_probs,
            title=f"Reliability — {map_name} test ({args.model})",
            save_path=output_dir / "reliability_test.png",
        )
        print(f"  Test  Brier: {test_metrics['brier']:.4f}  LogLoss: {test_metrics['logloss']:.4f}  AUC: {test_metrics['auc']:.4f}")

    run_config = {
        "map": map_name,
        "model": args.model,
        "use_transformer": args.use_transformer if use_deepsets else False,
        "player_hidden": args.player_hidden if use_deepsets else None,
        "global_hidden": args.global_hidden,
        "player_layers": args.player_layers if use_deepsets else None,
        "global_layers": args.global_layers,
        "dropout": args.dropout,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"  Checkpoint: {output_dir}/best_model.pt")
    return run_config


def main():
    parser = argparse.ArgumentParser(description="Train per-map round win predictors")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/dataset_masters_london"),
        help="Dataset root. If it contains per-map subdirs, all are trained. "
             "If it directly contains train.npz, treated as a single-map dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/masters_london"))
    parser.add_argument(
        "--model", choices=["deepsetsv1", "baseline"], default="deepsetsv1",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--player-hidden", type=int, default=64)
    parser.add_argument("--global-hidden", type=int, default=64)
    parser.add_argument("--player-layers", type=int, default=2)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--augment-flip", action="store_true", default=True,
                        help="Double training data by flipping attack/defense teams (default: on).")
    parser.add_argument("--no-augment-flip", dest="augment_flip", action="store_false")
    parser.add_argument("--use-transformer", action="store_true", default=True,
                        help="Cross-player Transformer (default: on). Disable with --no-transformer.")
    parser.add_argument("--no-transformer", dest="use_transformer", action="store_false")
    parser.add_argument("--use-spatial-bias", action="store_true", default=True,
                        help="Spatial proximity attention bias on Transformer (default: on).")
    parser.add_argument("--no-spatial-bias", dest="use_spatial_bias", action="store_false")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

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

    map_dirs = discover_map_dirs(args.dataset_dir)
    if not map_dirs:
        print(f"Error: no map datasets found under {args.dataset_dir}", file=sys.stderr)
        print("Run: python scripts/build_dataset.py --output-dir data/dataset_masters_london", file=sys.stderr)
        sys.exit(1)

    print(f"Maps to train: {[m for m, _ in map_dirs]}\n")

    all_metrics = {}
    for map_name, map_dataset_dir in map_dirs:
        out_dir = args.output_dir / map_name if len(map_dirs) > 1 else args.output_dir
        print(f"{'='*60}")
        print(f"Training map: {map_name.upper()}")
        metrics = train_one_map(map_name, map_dataset_dir, out_dir, args, device)
        all_metrics[map_name] = metrics

    if len(map_dirs) > 1:
        print(f"\n{'='*60}")
        print("Summary:")
        for map_name, m in all_metrics.items():
            val_b = m["val"].get("brier", "—")
            test_b = m["test"].get("brier", "—")
            val_str = f"{val_b:.4f}" if isinstance(val_b, float) else val_b
            test_str = f"{test_b:.4f}" if isinstance(test_b, float) else test_b
            print(f"  {map_name:<12} val_brier={val_str}  test_brier={test_str}")


if __name__ == "__main__":
    main()
