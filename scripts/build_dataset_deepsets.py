"""
Build per-player DeepSets training dataset from Phase 1 extended valoscribe output.

Reads frame_states.csv + event_log.jsonl from all processed maps (the Phase 1
extended format that includes weapon, weapon_tier, credits, pos_x, pos_y columns)
and produces tensor arrays matching the RoundPredictor.forward() signature.

Usage:
    python scripts/build_dataset_deepsets.py \
        --data-dir champs2025_processed_vods \
        --output-dir data/deepsets_dataset

Output files (per split):
    {split}_continuous_feats.npy   (N, 10, 12) float32
    {split}_agent_ids.npy          (N, 10)     int32
    {split}_weapon_tier_ids.npy    (N, 10)     int32
    {split}_global_feats.npy       (N, 6)      float32
    {split}_map_ids.npy            (N,)        int32
    {split}_atk_mask.npy           (N, 10)     float32
    {split}_alive_mask.npy         (N, 10)     float32
    {split}_y.npy                  (N,)        float32

    feature_schema.json  (documents what each column/dim means)
    splits.json          (which match_ids in each split)

Per-player continuous features (12 dims):
    0:  alive            (0/1)
    1:  health_norm      (health / 150)
    2:  armor_norm       (armor / 50)
    3:  ability_1_norm   (ability_1 / 3, clamped [0,1])
    4:  ability_2_norm   (ability_2 / 3, clamped [0,1])
    5:  ability_3_norm   (ability_3 / 3, clamped [0,1])
    6:  ult_charge_norm  (ult_charges / 8, clamped [0,1])
    7:  ult_ready        (1 if ultimate_full, else 0)
    8:  credits_norm     (credits / 9000, clamped [0,1])
    9:  pos_x            ([0,1] normalized map x, 0 if not detected)
    10: pos_y            ([0,1] normalized map y, 0 if not detected)
    11: has_position     (0/1)

Global features (6 dims):
    0:  round_number_norm   (round / 30)
    1:  score_diff_norm     ((atk_score - def_score) / 13)
    2:  time_remaining_norm (game_timer / 100)
    3:  spike_planted       (0/1)
    4:  spike_time_norm     (spike_timer / 45, 0 if not planted)
    5:  is_first_half       (1 if round_num <= 12, else 0)

Categorical features (encoded as integer indices):
    agent_ids:       1-indexed agent name (0 = unknown/padding)
    weapon_tier_ids: 0=unknown, 1=sidearm, 2=smg, 3=shotgun,
                     4=rifle, 5=sniper, 6=heavy, 7=melee
    map_ids:         1-indexed map name (0 = unknown/padding)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Categorical vocabularies ──────────────────────────────────────────────────

# All agents that appear in Champions 2025 broadcasts (from templates/active_round_agents)
AGENTS = [
    "astra", "breach", "brimstone", "chamber", "cypher",
    "deadlock", "fade", "gekko", "harbor", "jett",
    "kayo", "killjoy", "neon", "omen", "raze",
    "sage", "skye", "sova", "tejo", "viper",
    "vyse", "waylay", "yoru",
]
AGENT_TO_IDX: dict[str, int] = {a: i + 1 for i, a in enumerate(AGENTS)}  # 0 = unknown

WEAPON_TIER_TO_IDX: dict[str, int] = {
    "unknown":  0,
    "sidearm":  1,
    "smg":      2,
    "shotgun":  3,
    "rifle":    4,
    "sniper":   5,
    "heavy":    6,
    "melee":    7,
}

MAPS = [
    "ascent", "bind", "breeze", "corrode", "abyss",
    "fracture", "haven", "icebox", "lotus", "pearl",
    "split", "sunset",
]
MAP_TO_IDX: dict[str, int] = {m: i + 1 for i, m in enumerate(MAPS)}  # 0 = unknown


# ── Round side helper (shared with build_dataset.py) ─────────────────────────

def get_team1_side(round_num: int, starting_side: str) -> str:
    if round_num <= 12:
        block = 0
    elif round_num <= 24:
        block = 1
    else:
        block = 2 + (round_num - 25) % 2
    if block % 2 == 0:
        return starting_side
    return "defense" if starting_side == "attack" else "attack"


# ── Map discovery ─────────────────────────────────────────────────────────────

def find_map_dirs(data_dir: Path) -> list[tuple[str, Path]]:
    results = []
    for series_dir in sorted(data_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        match_id = series_dir.name
        for map_dir in sorted(series_dir.iterdir()):
            if not map_dir.is_dir() or map_dir.name == "metadata":
                continue
            output_dir = map_dir / "output"
            if (
                (output_dir / "frame_states.csv").exists()
                and (output_dir / "event_log.jsonl").exists()
                and (map_dir / "metadata.json").exists()
            ):
                results.append((match_id, map_dir))
    return results


# ── Label extraction ──────────────────────────────────────────────────────────

def build_round_outcomes(event_log_path: Path) -> dict[int, str]:
    outcomes: dict[int, str] = {}
    with open(event_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "round_end":
                outcomes[event["round_number"]] = event["winner"]
    return outcomes


def build_round_labels(
    round_outcomes: dict[int, str],
    team1_name: str,
    team2_name: str,
    team1_starting_side: str,
) -> dict[int, int]:
    labels: dict[int, int] = {}
    for rn, winner_team in round_outcomes.items():
        team1_side = get_team1_side(rn, team1_starting_side)
        team2_side = "defense" if team1_side == "attack" else "attack"
        if winner_team == team1_name:
            winning_side = team1_side
        elif winner_team == team2_name:
            winning_side = team2_side
        else:
            continue
        labels[rn] = 1 if winning_side == "attack" else 0
    return labels


# ── Feature helpers ───────────────────────────────────────────────────────────

def _float(val, default: float = 0.0) -> float:
    try:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def build_player_features(row: pd.Series, player_idx: int) -> tuple[np.ndarray, int, int]:
    """
    Returns (continuous_12, agent_id, weapon_tier_id) for one player.
    """
    p = f"player_{player_idx}_"

    alive = 1.0 if _bool(row.get(f"{p}alive")) else 0.0
    health = _clamp(_float(row.get(f"{p}health")) / 150.0)
    armor = _clamp(_float(row.get(f"{p}armor")) / 50.0)

    ab1 = _clamp(_float(row.get(f"{p}ability_1")) / 3.0)
    ab2 = _clamp(_float(row.get(f"{p}ability_2")) / 3.0)
    ab3 = _clamp(_float(row.get(f"{p}ability_3")) / 3.0)

    ult_charge_norm = _clamp(_float(row.get(f"{p}ultimate_charges")) / 8.0)
    ult_ready = 1.0 if _bool(row.get(f"{p}ultimate_full")) else 0.0

    credits_norm = _clamp(_float(row.get(f"{p}credits")) / 9000.0)

    pos_x_raw = row.get(f"{p}pos_x", "")
    pos_y_raw = row.get(f"{p}pos_y", "")
    has_pos = pos_x_raw != "" and pos_y_raw != ""
    pos_x = _clamp(_float(pos_x_raw)) if has_pos else 0.0
    pos_y = _clamp(_float(pos_y_raw)) if has_pos else 0.0

    continuous = np.array([
        alive, health, armor,
        ab1, ab2, ab3,
        ult_charge_norm, ult_ready,
        credits_norm,
        pos_x, pos_y, float(has_pos),
    ], dtype=np.float32)

    agent_name = str(row.get(f"{p}agent", "") or "").lower().strip()
    agent_id = AGENT_TO_IDX.get(agent_name, 0)

    tier_str = str(row.get(f"{p}weapon_tier", "") or "").lower().strip()
    weapon_tier_id = WEAPON_TIER_TO_IDX.get(tier_str, 0)

    return continuous, agent_id, weapon_tier_id


def build_global_features(
    row: pd.Series,
    round_num: int,
    atk_score: float,
    def_score: float,
) -> np.ndarray:
    round_norm = _clamp(round_num / 30.0)
    score_diff = _clamp((atk_score - def_score) / 13.0, -1.0, 1.0)
    time_norm = _clamp(_float(row.get("game_timer", 0)) / 100.0)

    spike_timer_raw = _float(row.get("spike_timer", 0))
    spike_planted = 1.0 if spike_timer_raw > 0 else 0.0
    spike_time_norm = _clamp(spike_timer_raw / 45.0)

    is_first_half = 1.0 if round_num <= 12 else 0.0

    return np.array([
        round_norm, score_diff, time_norm,
        spike_planted, spike_time_norm, is_first_half,
    ], dtype=np.float32)


# ── Per-map processing ────────────────────────────────────────────────────────

def process_map(map_dir: Path, verbose: bool = True) -> tuple[dict[str, np.ndarray], int] | None:
    """
    Returns (arrays_dict, n_rounds) where arrays_dict has all tensor components
    for this map's ACTIVE_ROUND frames. Returns None on failure.
    """
    output_dir = map_dir / "output"
    metadata_path = map_dir / "metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    teams = metadata.get("teams", [])
    if len(teams) < 2:
        print(f"  Skip {map_dir.name}: < 2 teams")
        return None

    team1_name = teams[0]["name"]
    team2_name = teams[1]["name"]
    team1_starting_side = teams[0]["starting_side"]

    map_name = metadata.get("map", "").lower().strip()
    map_id = MAP_TO_IDX.get(map_name, 0)
    if map_id == 0 and verbose:
        print(f"  Warning: unknown map name '{map_name}' — will use map_id=0")

    round_outcomes = build_round_outcomes(output_dir / "event_log.jsonl")
    if not round_outcomes:
        print(f"  Skip {map_dir.name}: no round_end events")
        return None

    round_labels = build_round_labels(round_outcomes, team1_name, team2_name, team1_starting_side)

    df = pd.read_csv(output_dir / "frame_states.csv", dtype=str)
    df = df[df["phase"] == "ACTIVE_ROUND"].copy()
    if df.empty:
        print(f"  Skip {map_dir.name}: no ACTIVE_ROUND frames")
        return None

    # Determine each player's team from the most common value in the CSV
    player_team: dict[int, str] = {}
    for pidx in range(10):
        col = f"player_{pidx}_team"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                player_team[pidx] = vals.value_counts().idxmax()

    continuous_list, agent_list, tier_list = [], [], []
    global_list, map_list, atk_mask_list, alive_mask_list, y_list = [], [], [], [], []
    skipped = 0

    for _, row in df.iterrows():
        try:
            rn = int(float(row.get("round_number", "")))
        except (ValueError, TypeError):
            skipped += 1
            continue

        if rn not in round_labels:
            skipped += 1
            continue

        game_timer = _float(row.get("game_timer", -1), default=-1.0)
        if game_timer > 97.0:  # skip first ~2s of round
            skipped += 1
            continue

        team1_side_this_round = get_team1_side(rn, team1_starting_side)

        # Which players are on which side this round
        atk_mask = np.zeros(10, dtype=np.float32)
        for pidx in range(10):
            pteam = player_team.get(pidx, "")
            if pteam == team1_name:
                pside = team1_side_this_round
            elif pteam == team2_name:
                pside = "defense" if team1_side_this_round == "attack" else "attack"
            else:
                pside = "defense"  # unknown → default to def
            if pside == "attack":
                atk_mask[pidx] = 1.0

        # Per-player features
        cont_frame = np.zeros((10, 12), dtype=np.float32)
        agent_frame = np.zeros(10, dtype=np.int32)
        tier_frame = np.zeros(10, dtype=np.int32)
        alive_mask = np.zeros(10, dtype=np.float32)

        for pidx in range(10):
            cont, aid, tid = build_player_features(row, pidx)
            cont_frame[pidx] = cont
            agent_frame[pidx] = aid
            tier_frame[pidx] = tid
            alive_mask[pidx] = cont[0]  # cont[0] == alive

        # Scores — from attack team's perspective
        score_t1 = _float(row.get("score_team1", 0))
        score_t2 = _float(row.get("score_team2", 0))
        if team1_side_this_round == "attack":
            atk_score, def_score = score_t1, score_t2
        else:
            atk_score, def_score = score_t2, score_t1

        global_feat = build_global_features(row, rn, atk_score, def_score)

        continuous_list.append(cont_frame)
        agent_list.append(agent_frame)
        tier_list.append(tier_frame)
        global_list.append(global_feat)
        map_list.append(map_id)
        atk_mask_list.append(atk_mask)
        alive_mask_list.append(alive_mask)
        y_list.append(float(round_labels[rn]))

    if not y_list:
        print(f"  Skip {map_dir.name}: no labeled frames after filtering")
        return None

    n_frames = len(y_list)
    n_atk_wins = sum(y_list)
    if verbose:
        print(
            f"  {map_dir.parent.name}/{map_dir.name}: "
            f"{n_frames} frames, {int(n_atk_wins)} atk wins, "
            f"{len(round_labels)} rounds, map={map_name}"
        )

    arrays = {
        "continuous_feats": np.stack(continuous_list),    # (N, 10, 12)
        "agent_ids":         np.stack(agent_list),         # (N, 10)
        "weapon_tier_ids":   np.stack(tier_list),          # (N, 10)
        "global_feats":      np.stack(global_list),        # (N, 6)
        "map_ids":           np.array(map_list, dtype=np.int32),
        "atk_mask":          np.stack(atk_mask_list),      # (N, 10)
        "alive_mask":        np.stack(alive_mask_list),    # (N, 10)
        "y":                 np.array(y_list, dtype=np.float32),
    }
    return arrays, len(round_labels)


# ── Concatenate arrays across maps ───────────────────────────────────────────

def concat_arrays(list_of_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list_of_dicts[0].keys()
    return {k: np.concatenate([d[k] for d in list_of_dicts], axis=0) for k in keys}


# ── Dataset split ─────────────────────────────────────────────────────────────

def split_by_match(
    match_data: dict[str, dict[str, np.ndarray]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict, dict, dict]:
    rng = np.random.default_rng(seed)
    match_ids = sorted(match_data.keys())
    idx = list(range(len(match_ids)))
    rng.shuffle(idx)
    shuffled = [match_ids[i] for i in idx]

    n = len(shuffled)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_ids = shuffled[-n_test:]
    val_ids = shuffled[-(n_test + n_val) : -n_test]
    train_ids = shuffled[: -(n_test + n_val)]

    def collect(ids):
        dicts = [match_data[mid] for mid in ids if mid in match_data]
        if not dicts:
            return None
        return concat_arrays(dicts)

    splits_meta = {"train": train_ids, "val": val_ids, "test": test_ids}
    return collect(train_ids), collect(val_ids), collect(test_ids), splits_meta


# ── Main ──────────────────────────────────────────────────────────────────────

FEATURE_SCHEMA = {
    "continuous_feats": {
        "shape": "(N, 10, 12)",
        "dims": [
            "alive", "health_norm", "armor_norm",
            "ability_1_norm", "ability_2_norm", "ability_3_norm",
            "ult_charge_norm", "ult_ready", "credits_norm",
            "pos_x", "pos_y", "has_position",
        ],
    },
    "agent_ids": {
        "shape": "(N, 10)",
        "vocab": AGENT_TO_IDX,
        "note": "0 = unknown/padding",
    },
    "weapon_tier_ids": {
        "shape": "(N, 10)",
        "vocab": WEAPON_TIER_TO_IDX,
        "note": "0 = unknown",
    },
    "global_feats": {
        "shape": "(N, 6)",
        "dims": [
            "round_number_norm", "score_diff_norm", "time_remaining_norm",
            "spike_planted", "spike_time_norm", "is_first_half",
        ],
    },
    "map_ids": {
        "shape": "(N,)",
        "vocab": MAP_TO_IDX,
        "note": "0 = unknown/padding",
    },
    "atk_mask":   {"shape": "(N, 10)", "note": "1 = player is on attack team this round"},
    "alive_mask": {"shape": "(N, 10)", "note": "1 = player alive"},
    "y":          {"shape": "(N,)", "note": "1 = attack wins round, 0 = defense wins"},
}


def main():
    parser = argparse.ArgumentParser(description="Build DeepSets per-player dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("champs2025_processed_vods"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/deepsets_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.data_dir} for processed maps...")
    map_dirs = find_map_dirs(args.data_dir)
    print(f"Found {len(map_dirs)} maps across {len(set(m for m, _ in map_dirs))} series\n")

    match_data: dict[str, dict[str, np.ndarray]] = {}
    total_rounds = 0

    for match_id, map_dir in map_dirs:
        result = process_map(map_dir, verbose=not args.quiet)
        if result is None:
            continue
        arrays, n_rounds = result
        total_rounds += n_rounds

        if match_id in match_data:
            existing = match_data[match_id]
            match_data[match_id] = concat_arrays([existing, arrays])
        else:
            match_data[match_id] = arrays

    if not match_data:
        print("No usable data found.", file=sys.stderr)
        sys.exit(1)

    total_frames = sum(len(d["y"]) for d in match_data.values())
    total_atk = sum(d["y"].sum() for d in match_data.values())
    print(
        f"\nTotal: {total_frames:,} frames, {total_rounds} rounds, "
        f"{len(match_data)} series, attack win rate: {total_atk/total_frames:.1%}"
    )

    # Check Phase 1 coverage (how many frames have weapon/position data)
    all_feats = np.concatenate([d["continuous_feats"] for d in match_data.values()])
    has_pos_rate = all_feats[:, :, 11].mean()  # dim 11 = has_position
    credits_filled = (all_feats[:, :, 8] > 0).mean()  # credits_norm > 0
    print(f"Position fill rate: {has_pos_rate:.1%}  Credits fill rate: {credits_filled:.1%}")

    # Check weapon tier coverage
    all_tiers = np.concatenate([d["weapon_tier_ids"] for d in match_data.values()])
    weapon_filled = (all_tiers > 0).mean()
    print(f"Weapon tier fill rate: {weapon_filled:.1%}")

    (train_arr, val_arr, test_arr, splits_meta) = split_by_match(
        match_data,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    for split_name, arrays in [("train", train_arr), ("val", val_arr), ("test", test_arr)]:
        if arrays is None:
            print(f"Warning: {split_name} split is empty")
            continue
        n = len(arrays["y"])
        atk_rate = arrays["y"].mean()
        print(f"  {split_name}: {n:,} frames, atk win rate: {atk_rate:.1%}")
        for key, arr in arrays.items():
            np.save(args.output_dir / f"{split_name}_{key}.npy", arr)

    with open(args.output_dir / "feature_schema.json", "w") as f:
        json.dump(FEATURE_SCHEMA, f, indent=2)

    with open(args.output_dir / "splits.json", "w") as f:
        json.dump(splits_meta, f, indent=2)

    print(f"\nDataset saved to {args.output_dir}/")
    print("Files per split: continuous_feats, agent_ids, weapon_tier_ids,")
    print("                 global_feats, map_ids, atk_mask, alive_mask, y")
    print("\nTo train the DeepSets model:")
    print("  python scripts/train_deepsets.py --dataset-dir data/deepsets_dataset")


if __name__ == "__main__":
    main()
