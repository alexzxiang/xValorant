"""
predict_frames.py — run the round win predictor on frames from a processed map.

Loads frame_states.csv from an already-processed map directory, picks frames at
interesting game states, runs them through the trained model, and prints a
human-readable breakdown showing what the model saw and what it predicted.

Usage:
    python scripts/predict_frames.py \
        --series masters_london_2026/nrg_vs_lev/map1_ascent \
        --model-dir models/masters_london \
        [--n-samples 20] [--round N] [--seed 42]

Examples:
    # Sample 20 random frames from ascent
    python scripts/predict_frames.py --series masters_london_2026/nrg_vs_lev/map1_ascent --model-dir models/masters_london

    # Show every frame in round 7 of fracture
    python scripts/predict_frames.py --series masters_london_2026/leviatn_vs_g2_esports/map1_fracture --model-dir models/masters_london --round 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# -- import feature logic from build_dataset ----------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import (
    compute_features,
    build_round_outcomes,
    build_round_labels,
    get_team1_side,
    AGENT_ROLES,
    WEAPON_TIER_BY_NAME,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.models.round_predictor import RoundPredictor, ModelConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    try:
        if val == "" or val is None:
            return default
        f = float(val)
        return default if f != f else f
    except (ValueError, TypeError):
        return default


def _safe_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return False


def load_model(model_dir: Path, map_name: str) -> RoundPredictor:
    map_dir = model_dir / map_name
    cfg_path = map_dir / "config.json"
    ckpt_path = map_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    cfg_data = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg = ModelConfig(**{k: v for k, v in cfg_data.items() if hasattr(ModelConfig, k)})
    model = RoundPredictor(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model(model: RoundPredictor, feats: dict) -> float:
    """Run one frame through the model; return P(attack wins) as float."""
    def t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype).unsqueeze(0)

    with torch.no_grad():
        prob = model.predict_proba(
            continuous_feats=t(feats["player_feats"]),
            agent_ids=t(feats["agent_ids"], dtype=torch.long),
            weapon_ids=t(feats["weapon_ids"], dtype=torch.long),
            role_ids=t(feats["role_ids"], dtype=torch.long),
            global_feats=t(feats["global_feats"]),
            atk_mask=t(feats["atk_mask"]),
            alive_mask=t(feats["alive_mask"]),
        )
    return float(prob.squeeze())


def describe_player(row: pd.Series, pidx: int, player_teams: dict, team1_name: str,
                    team1_side: str, agents_config: dict) -> str:
    """One-line summary for a single player slot."""
    prefix = f"player_{pidx}_"
    alive = _safe_bool(row.get(f"{prefix}alive", False))
    if not alive:
        return None
    agent = str(row.get(f"{prefix}agent", "") or "").lower().strip() or "?"
    weapon = str(row.get(f"{prefix}weapon", "") or "").lower().strip() or "?"
    tier = WEAPON_TIER_BY_NAME.get(weapon, "?")
    health = _safe_float(row.get(f"{prefix}health", 0))
    ult_ready = _safe_bool(row.get(f"{prefix}ultimate_full", False))
    pteam = player_teams.get(pidx, "")
    is_atk = (pteam == team1_name) == (team1_side == "attack")
    side_tag = "ATK" if is_atk else "DEF"
    role = AGENT_ROLES.get(agent, "?")
    ult_tag = " [ULT]" if ult_ready else ""
    pos_x = _safe_float(row.get(f"{prefix}pos_x", ""))
    pos_y = _safe_float(row.get(f"{prefix}pos_y", ""))
    has_pos = pos_x != 0 or pos_y != 0
    pos_tag = f" ({pos_x:.2f},{pos_y:.2f})" if has_pos else ""
    return f"  {side_tag} {agent:<10} {role:<11} hp={int(health):3d}  {weapon:<10} [{tier}]{ult_tag}{pos_tag}"


def print_frame(row: pd.Series, feats: dict, prob_atk: float, true_label: int | None,
                team1_name: str, team2_name: str, team1_starting_side: str,
                player_teams: dict, agents_config: dict):
    round_num = int(float(row.get("round_number", 0) or 0))
    game_timer = _safe_float(row.get("game_timer", 0))
    spike_timer = _safe_float(row.get("spike_timer", 0))
    score_t1 = int(_safe_float(row.get("score_team1", 0)))
    score_t2 = int(_safe_float(row.get("score_team2", 0)))
    ts = _safe_float(row.get("timestamp", 0))

    team1_side = get_team1_side(round_num, team1_starting_side)
    team2_side = "defense" if team1_side == "attack" else "attack"

    spike_str = f"  SPIKE PLANTED ({spike_timer:.0f}s)" if spike_timer > 0 else ""
    label_str = ""
    if true_label is not None:
        outcome = "ATK wins" if true_label == 1 else "DEF wins"
        correct = (true_label == 1) == (prob_atk > 0.5)
        label_str = f"  | actual={outcome} ({'correct' if correct else 'WRONG'})"

    print(f"\n{'='*70}")
    print(f"Round {round_num}  |  timer={game_timer:.0f}s{spike_str}  |  ts={ts:.0f}s")
    print(f"Score: {team1_name} ({team1_side}) {score_t1} - {score_t2} {team2_name} ({team2_side})")
    print(f"P(attack wins) = {prob_atk:.1%}   P(defense wins) = {1-prob_atk:.1%}{label_str}")

    atk_alive = int(feats["global_feats"][8] * 5)
    def_alive = int(feats["global_feats"][9] * 5)
    atk_ults = int(round(feats["global_feats"][10] * 5))
    def_ults = int(round(feats["global_feats"][11] * 5))
    print(f"Alive: ATK {atk_alive}/5 (ults={atk_ults})  DEF {def_alive}/5 (ults={def_ults})")

    print("Players:")
    for pidx in range(10):
        line = describe_player(row, pidx, player_teams, team1_name, team1_side, agents_config)
        if line:
            print(line)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series", required=True,
        help="Path to processed map dir, e.g. masters_london_2026/nrg_vs_lev/map1_ascent",
    )
    parser.add_argument(
        "--model-dir", default="models/masters_london",
        help="Root model directory containing <map_name>/best_model.pt",
    )
    parser.add_argument(
        "--n-samples", type=int, default=15,
        help="Number of frames to sample (ignored if --round is set)",
    )
    parser.add_argument(
        "--round", type=int, default=None,
        help="If set, show all frames in this round number",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    series_dir = Path(args.series)
    output_dir = series_dir / "output"
    metadata_path = series_dir / "metadata.json"

    if not metadata_path.exists():
        sys.exit(f"metadata.json not found at {metadata_path}")
    if not (output_dir / "frame_states.csv").exists():
        sys.exit(f"frame_states.csv not found in {output_dir}")

    with open(metadata_path, encoding="cp1252") as f:
        metadata = json.load(f)

    teams = {t["name"]: t for t in metadata["teams"]}
    team_names = list(teams.keys())
    team1_name, team2_name = team_names[0], team_names[1]
    team1_starting_side = teams[team1_name]["starting_side"]
    map_name = metadata.get("map", "unknown").lower()

    agents_config_path = Path("src/valoscribe/config/agents_champs2025.json")
    agents_config = json.loads(agents_config_path.read_text()) if agents_config_path.exists() else {}

    round_outcomes = build_round_outcomes(output_dir / "event_log.jsonl")
    round_labels = build_round_labels(round_outcomes, team1_name, team2_name, team1_starting_side)

    df = pd.read_csv(output_dir / "frame_states.csv", dtype=str, encoding="cp1252")
    df = df[df["phase"] == "ACTIVE_ROUND"].copy()
    df = df[df["game_timer"].apply(lambda v: _safe_float(v, default=101.0)) <= 97.0]

    # Build player->team map
    player_teams: dict[int, str] = {}
    for pidx in range(10):
        col = f"player_{pidx}_team"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                player_teams[pidx] = vals.value_counts().idxmax()

    if args.round is not None:
        df = df[df["round_number"].apply(lambda v: int(float(v)) if v else -1) == args.round]
        if df.empty:
            sys.exit(f"No ACTIVE_ROUND frames found for round {args.round}")
        sample = df
    else:
        n = min(args.n_samples, len(df))
        sample = df.sample(n=n, random_state=args.seed)
        sample = sample.sort_values("timestamp")

    print(f"\nMap: {map_name.upper()}  |  {team1_name} (starts {team1_starting_side}) vs {team2_name}")
    print(f"Model: {args.model_dir}/{map_name}/best_model.pt")

    model = load_model(Path(args.model_dir), map_name)
    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params)")

    correct = 0
    total = 0

    for _, row in sample.iterrows():
        rn_raw = row.get("round_number", "")
        try:
            rn = int(float(rn_raw))
        except (ValueError, TypeError):
            continue

        feats = compute_features(row, rn, team1_name, team1_starting_side, player_teams, agents_config)
        prob_atk = run_model(model, feats)
        true_label = round_labels.get(rn)

        print_frame(row, feats, prob_atk, true_label,
                    team1_name, team2_name, team1_starting_side,
                    player_teams, agents_config)

        if true_label is not None:
            correct += int((prob_atk > 0.5) == (true_label == 1))
            total += 1

    if total > 0:
        print(f"\n{'='*70}")
        print(f"Frame-level accuracy on this sample: {correct}/{total} = {correct/total:.1%}")
        print("(Frame accuracy inflates vs round accuracy — frames within a round are correlated)")


if __name__ == "__main__":
    main()
