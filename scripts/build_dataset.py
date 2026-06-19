"""
Build labeled training dataset from valoscribe processed map outputs.

One model is trained per map, so datasets are split by map name:

    <output_dir>/<map_name>/train.npz
    <output_dir>/<map_name>/val.npz
    <output_dir>/<map_name>/test.npz
    <output_dir>/<map_name>/metadata.json
    <output_dir>/<map_name>/splits.json

Split is always by series (match) to prevent data leakage. Maps with fewer
than 2 series get all data in train; fewer than 3 get no test split.

Usage:
    python scripts/build_dataset.py \
        --data-dirs masters_london_2026 \
        --output-dir data/dataset_masters_london
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Agent / role constants ───────────────────────────────────────────────────

# Role IDs: 0=unknown, 1=duelist, 2=initiator, 3=controller, 4=sentinel
ROLE_IDS: dict[str, int] = {
    "duelist": 1, "initiator": 2, "controller": 3, "sentinel": 4
}

AGENT_ROLES: dict[str, str] = {
    # Duelists — entry fraggers, high individual carry, more replaceable
    "jett": "duelist", "reyna": "duelist", "raze": "duelist",
    "phoenix": "duelist", "neon": "duelist", "yoru": "duelist",
    "iso": "duelist", "waylay": "duelist",
    # Initiators — info gathering, enable site entry; losing one hurts coordination
    "sova": "initiator", "breach": "initiator", "skye": "initiator",
    "kayo": "initiator", "fade": "initiator", "gekko": "initiator",
    "tejo": "initiator",
    # Controllers — smokes, area denial; losing one breaks site execution/defense
    "brimstone": "controller", "viper": "controller", "omen": "controller",
    "astra": "controller", "harbor": "controller", "clove": "controller",
    # Sentinels — site anchors, information tools; critical for defense economy
    "sage": "sentinel", "cypher": "sentinel", "killjoy": "sentinel",
    "deadlock": "sentinel", "chamber": "sentinel", "vyse": "sentinel",
}

# Alphabetically ordered for determinism across runs
_ALL_AGENTS = sorted(AGENT_ROLES.keys())
AGENT_IDS: dict[str, int] = {a: i + 1 for i, a in enumerate(_ALL_AGENTS)}  # 1-indexed, 0=unknown

NUM_ROLES = 4       # duelist, initiator, controller, sentinel
NUM_AGENTS = 30     # pad to 30 for future agents
NUM_WEAPONS = 30    # specific weapon names; 0=unknown/padding


# ─── Weapon constants ─────────────────────────────────────────────────────────

# Specific weapon names — ordered alphabetically for determinism; 0=unknown/padding
_ALL_WEAPONS = sorted([
    # Sidearms
    "classic", "shorty", "frenzy", "ghost", "sheriff", "bandit",
    # SMGs
    "stinger", "spectre",
    # Shotguns
    "bucky", "judge",
    # Rifles (most important distinction in competitive play)
    "bulldog", "guardian", "phantom", "vandal",
    # Snipers
    "marshal", "outlaw", "operator",
    # Heavy
    "ares", "odin",
    # Melee
    "melee", "knife",
])
WEAPON_IDS: dict[str, int] = {w: i + 1 for i, w in enumerate(_ALL_WEAPONS)}  # 1-indexed

# Weapon tier for aggregate economy features (not used in per-player embedding)
WEAPON_TIER_BY_NAME: dict[str, str] = {
    "classic": "sidearm", "shorty": "sidearm", "frenzy": "sidearm",
    "ghost": "sidearm", "sheriff": "sidearm", "bandit": "sidearm",
    "stinger": "smg", "spectre": "smg",
    "bucky": "shotgun", "judge": "shotgun",
    "bulldog": "rifle", "guardian": "rifle", "phantom": "rifle", "vandal": "rifle",
    "marshal": "sniper", "outlaw": "sniper", "operator": "sniper",
    "ares": "heavy", "odin": "heavy",
    "melee": "melee", "knife": "melee",
}

# Economy quality score — used only in aggregate features
WEAPON_TIER_QUALITY: dict[str, float] = {
    "unknown": 1.0,   # assume pistol-level on missing data
    "sidearm": 1.0, "smg": 2.0, "shotgun": 2.0,
    "rifle": 4.0, "sniper": 4.0, "heavy": 3.0, "melee": 0.0,
}
RIFLE_PLUS_TIERS: set[str] = {"rifle", "sniper"}


# ─── Feature schema ───────────────────────────────────────────────────────────

PLAYER_FEAT_DIM = 11   # continuous dims per player, fed to DeepSets encoder

PLAYER_FEAT_NAMES = [
    "alive",            # 0/1
    "health_norm",      # health / 150
    "armor_norm",       # armor / 50
    "ability_1_norm",   # charges / max_charges
    "ability_2_norm",
    "ability_3_norm",
    "ult_charge_norm",  # ult charges / max_charges
    "ult_ready",        # 0/1 (ultimate full)
    "pos_x",            # normalized map x (0 if not detected)
    "pos_y",            # normalized map y (0 if not detected)
    "has_position",     # 0/1 flag — lets model know if pos_x/y are valid
]

GLOBAL_FEAT_NAMES = [
    # ── Tactical / temporal ──────────────────────────────────────────────────────
    "atk_is_team1",        # 1 if team1 is attacking this round
    "round_number_norm",   # round_num / 30
    "score_diff",          # (atk_score - def_score) / 13
    "time_remaining_norm", # game_timer / 100
    "spike_planted",       # 0/1
    "spike_time_norm",     # spike_timer / 45 (0 if not planted)
    # ── Economy context (pistol / eco / force / full-buy) ────────────────────────
    # Derived from avg loadout quality of alive players; 0=pistol, 1=full-buy
    "atk_economy",
    "def_economy",
    # ── Team aggregates (explicitly available to the global MLP) ────────────────
    # The per-player transformer pools implicitly capture these, but having them
    # directly here gives the global head a short-circuit path to critical scalars.
    "atk_alive_norm",      # alive / 5
    "def_alive_norm",
    "atk_ult_frac",        # ults ready / 5
    "def_ult_frac",
    "atk_rifle_frac",      # players with rifle+ / 5
    "def_rifle_frac",
    "atk_health_norm",     # total health / 750
    "def_health_norm",
    "atk_loadout_norm",    # sum weapon quality / 20
    "def_loadout_norm",
    # ── Role-based alive (controllers/initiators matter most per round theory) ───
    "controllers_alive_atk",    # / 2  (typical max 2 controllers per team)
    "controllers_alive_def",
    "initiators_alive_atk",     # / 2
    "initiators_alive_def",
    # ── Spatial aggregates (0 when positions unavailable) ───────────────────────
    # Capture macro-level positioning: stacking, spread, engagement proximity.
    "atk_pos_coverage",    # fraction of alive atk players with detected positions
    "def_pos_coverage",
    "atk_centroid_x",      # mean x of alive+has_pos atk players
    "atk_centroid_y",
    "def_centroid_x",
    "def_centroid_y",
    "atk_spread",          # mean pairwise distance among alive atk players with positions
    "def_spread",
    "min_cross_dist",      # min distance between any alive atk and def player with positions
]

GLOBAL_FEAT_DIM = len(GLOBAL_FEAT_NAMES)  # 31

# Aggregate features — used by the baseline MLP and for analysis
AGGREGATE_FEATURE_NAMES = [
    # ── Baseline features (backward compat) ──────────────────────────────────
    "players_alive_atk", "players_alive_def",
    "total_health_atk", "total_health_def",
    "ults_ready_atk", "ults_ready_def",
    "spike_planted", "time_remaining", "round_number_norm",
    "score_diff", "atk_is_team1",
    # ── Role-based alive counts ───────────────────────────────────────────────
    # Conventional wisdom: losing controller/initiator > losing duelist
    "duelists_alive_atk", "duelists_alive_def",
    "initiators_alive_atk", "initiators_alive_def",
    "controllers_alive_atk", "controllers_alive_def",
    "sentinels_alive_atk", "sentinels_alive_def",
    # ── Economy features (derived from weapon name → tier mapping) ──────────
    "loadout_score_atk", "loadout_score_def",  # sum of weapon quality / 20 (max 5×4)
    "rifles_atk", "rifles_def",               # fraction of team with rifle+
    # ── Position coverage ────────────────────────────────────────────────────
    "positions_avail_atk", "positions_avail_def",
    # ── Spike timer ──────────────────────────────────────────────────────────
    "spike_time_norm",
]

N_AGG_FEATS = len(AGGREGATE_FEATURE_NAMES)
N_GLOBAL_FEATS = GLOBAL_FEAT_DIM


# ─── Round side calculation ───────────────────────────────────────────────────

def get_team1_side(round_num: int, starting_side: str) -> str:
    """
    Return team1's side for a given round number.
    Rounds 1-12: starting side. 13-24: flipped. 25+: alternates.
    """
    if round_num <= 12:
        block = 0
    elif round_num <= 24:
        block = 1
    else:
        block = 2 + (round_num - 25) % 2
    if block % 2 == 0:
        return starting_side
    return "defense" if starting_side == "attack" else "attack"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val == "" or val is None:
            return default
        f = float(val)
        return default if (f != f) else f  # NaN check without importing math
    except (ValueError, TypeError):
        return default


def _safe_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return False


def _safe_pos(val) -> tuple[float, bool]:
    """Returns (float_value, is_valid). Invalid on empty/NaN."""
    try:
        if val == "" or val is None:
            return 0.0, False
        f = float(val)
        if f != f:  # NaN
            return 0.0, False
        return float(f), True
    except (ValueError, TypeError):
        return 0.0, False


# ─── Economy classification ────────────────────────────────────────────────────

# Pistol round threshold: 1 rifle = quality 4.0, max 5 rifles = 20.0
# avg quality per alive player thresholds:
#   <= 1.2 → pistol  (only default sidearms / classic)
#   <= 2.2 → eco     (mostly sidearms, maybe 1 spectre)
#   <= 3.5 → force   (spectre/bulldog/shotgun mix, few rifles)
#   > 3.5  → full-buy (phantom/vandal/operator)
_ECON_PISTOL = 1.2
_ECON_ECO    = 2.2
_ECON_FORCE  = 3.5

def classify_economy(total_quality: float, n_alive: int) -> float:
    """
    Map team loadout quality → normalized economy type in [0, 1].
    0 = pistol, 1/3 = eco, 2/3 = force, 1 = full-buy.
    """
    if n_alive == 0:
        return 0.0
    avg = total_quality / n_alive
    if avg <= _ECON_PISTOL:
        return 0.0
    if avg <= _ECON_ECO:
        return 1 / 3
    if avg <= _ECON_FORCE:
        return 2 / 3
    return 1.0


# ─── Spatial aggregate helpers ────────────────────────────────────────────────

def _spatial_aggregates(
    positions: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """
    Given a list of (x, y) positions for alive players with detected positions,
    return (centroid_x, centroid_y, spread).

    spread = mean pairwise Euclidean distance (0 if < 2 players).
    Returns (0, 0, 0) if the list is empty.
    """
    n = len(positions)
    if n == 0:
        return 0.0, 0.0, 0.0
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    if n < 2:
        return cx, cy, 0.0
    total_dist = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            total_dist += (dx * dx + dy * dy) ** 0.5
            pairs += 1
    return cx, cy, total_dist / pairs


def _min_cross_distance(
    atk_positions: list[tuple[float, float]],
    def_positions: list[tuple[float, float]],
) -> float:
    """
    Minimum Euclidean distance between any attacker and any defender.
    Returns 0 if either team has no detected positions.
    """
    if not atk_positions or not def_positions:
        return 0.0
    min_d = float("inf")
    for ax, ay in atk_positions:
        for dx, dy in def_positions:
            d = ((ax - dx) ** 2 + (ay - dy) ** 2) ** 0.5
            if d < min_d:
                min_d = d
    return min_d


# ─── Map discovery ────────────────────────────────────────────────────────────

def find_map_dirs(
    data_dirs: list[Path],
    only_maps: set[str] | None = None,
) -> list[tuple[str, Path]]:
    """
    Yield (match_id, map_dir) for every processed map found under data_dirs.
    Layout expected: data_dir/<series_id>/<map_name>/output/

    only_maps: if given, restrict to entries matching "<series_id>/<map_name>" exactly.
    """
    results = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  Warning: data dir not found: {data_dir}")
            continue
        for series_dir in sorted(data_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            match_id = f"{data_dir.name}/{series_dir.name}"
            for map_dir in sorted(series_dir.iterdir()):
                if not map_dir.is_dir() or map_dir.name == "metadata":
                    continue
                key = f"{series_dir.name}/{map_dir.name}"
                if only_maps is not None and key not in only_maps:
                    continue
                output_dir = map_dir / "output"
                if (
                    (output_dir / "frame_states.csv").exists()
                    and (output_dir / "event_log.jsonl").exists()
                    and (map_dir / "metadata.json").exists()
                ):
                    results.append((match_id, map_dir))
    return results


# ─── Label extraction ─────────────────────────────────────────────────────────

def build_round_outcomes(event_log_path: Path) -> dict[int, str]:
    outcomes: dict[int, str] = {}
    with open(event_log_path, encoding="cp1252") as f:
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
    """Returns {round_number: label} where label=1 means attack wins."""
    labels: dict[int, int] = {}
    for round_num, winner in round_outcomes.items():
        team1_side = get_team1_side(round_num, team1_starting_side)
        if winner == team1_name:
            winning_side = team1_side
        elif winner == team2_name:
            winning_side = "defense" if team1_side == "attack" else "attack"
        else:
            continue  # winner name doesn't match metadata (data quality issue)
        labels[round_num] = 1 if winning_side == "attack" else 0
    return labels


# ─── Feature computation ──────────────────────────────────────────────────────

def compute_features(
    row: pd.Series,
    round_num: int,
    team1_name: str,
    team1_starting_side: str,
    player_teams: dict[int, str],
    agents_config: dict,
) -> dict:
    """
    Compute all feature arrays for a single ACTIVE_ROUND frame.

    Returns a dict of numpy arrays:
        agg_feats:       (N_AGG_FEATS,) float32
        player_feats:    (10, PLAYER_FEAT_DIM) float32
        agent_ids:       (10,) int32
        weapon_ids:      (10,) int32
        role_ids:        (10,) int32
        global_feats:    (N_GLOBAL_FEATS,) float32
        atk_mask:        (10,) float32 — 1 for attack players
        alive_mask:      (10,) float32 — 1 for alive players
    """
    team1_side = get_team1_side(round_num, team1_starting_side)
    atk_is_team1 = 1.0 if team1_side == "attack" else 0.0

    spike_timer_val = _safe_float(row.get("spike_timer", 0))
    spike_planted = 1.0 if spike_timer_val > 0 else 0.0
    spike_time_norm = min(spike_timer_val / 45.0, 1.0)
    game_timer = _safe_float(row.get("game_timer", 0))
    time_norm = min(game_timer / 100.0, 1.0)

    score_team1 = _safe_float(row.get("score_team1", 0))
    score_team2 = _safe_float(row.get("score_team2", 0))
    atk_score = score_team1 if team1_side == "attack" else score_team2
    def_score = score_team2 if team1_side == "attack" else score_team1
    score_diff = (atk_score - def_score) / 13.0
    round_norm = min(round_num, 30) / 30.0

    # Per-player arrays
    player_feats = np.zeros((10, PLAYER_FEAT_DIM), dtype=np.float32)
    agent_ids = np.zeros(10, dtype=np.int32)
    weapon_ids = np.zeros(10, dtype=np.int32)
    role_ids = np.zeros(10, dtype=np.int32)
    atk_mask = np.zeros(10, dtype=np.float32)
    alive_mask = np.zeros(10, dtype=np.float32)

    # Aggregate accumulators
    alive = {"atk": 0, "def": 0}
    total_health = {"atk": 0.0, "def": 0.0}
    ults_ready = {"atk": 0, "def": 0}
    role_alive: dict[str, dict[str, int]] = {r: {"atk": 0, "def": 0} for r in ROLE_IDS}
    loadout_quality = {"atk": 0.0, "def": 0.0}
    rifle_count = {"atk": 0, "def": 0}
    pos_count = {"atk": 0, "def": 0}
    # Spatial: collect (x, y) of alive players with detected positions per team
    positions: dict[str, list[tuple[float, float]]] = {"atk": [], "def": []}

    for pidx in range(10):
        prefix = f"player_{pidx}_"

        pteam = player_teams.get(pidx, "")
        is_atk = (pteam == team1_name) == (team1_side == "attack")
        side = "atk" if is_atk else "def"
        atk_mask[pidx] = 1.0 if is_atk else 0.0

        is_alive = _safe_bool(row.get(f"{prefix}alive", False))
        alive_mask[pidx] = 1.0 if is_alive else 0.0

        agent_name = str(row.get(f"{prefix}agent", "") or "").lower().strip()
        agent_ids[pidx] = AGENT_IDS.get(agent_name, 0)
        role = AGENT_ROLES.get(agent_name, "")
        role_ids[pidx] = ROLE_IDS.get(role, 0)

        weapon_name = str(row.get(f"{prefix}weapon", "") or "").lower().strip()
        weapon_ids[pidx] = WEAPON_IDS.get(weapon_name, 0)
        weapon_tier_str = WEAPON_TIER_BY_NAME.get(weapon_name, "unknown")

        health_val = _safe_float(row.get(f"{prefix}health", 0))
        armor_val = _safe_float(row.get(f"{prefix}armor", 0))

        # Normalize ability charges by agent's max charges (from config)
        agent_cfg = agents_config.get(agent_name, {})
        ab1_max = max(1, (agent_cfg.get("ability_1") or {}).get("max_charges", 1) or 1)
        ab2_max = max(1, (agent_cfg.get("ability_2") or {}).get("max_charges", 1) or 1)
        ab3_max = max(1, (agent_cfg.get("ability_3") or {}).get("max_charges", 1) or 1)
        ult_max = max(1, (agent_cfg.get("ultimate") or {}).get("max_charges", 7) or 7)

        ab1 = min(_safe_float(row.get(f"{prefix}ability_1", 0)) / ab1_max, 1.0)
        ab2 = min(_safe_float(row.get(f"{prefix}ability_2", 0)) / ab2_max, 1.0)
        ab3 = min(_safe_float(row.get(f"{prefix}ability_3", 0)) / ab3_max, 1.0)
        ult_norm = min(_safe_float(row.get(f"{prefix}ultimate_charges", 0)) / ult_max, 1.0)
        ult_ready_val = 1.0 if _safe_bool(row.get(f"{prefix}ultimate_full", False)) else 0.0

        pos_x, has_px = _safe_pos(row.get(f"{prefix}pos_x", ""))
        pos_y, has_py = _safe_pos(row.get(f"{prefix}pos_y", ""))
        has_pos = 1.0 if (has_px and has_py) else 0.0

        player_feats[pidx] = [
            1.0 if is_alive else 0.0,   # alive
            health_val / 150.0,         # health_norm
            armor_val / 50.0,           # armor_norm
            max(0.0, ab1),              # ability_1_norm
            max(0.0, ab2),              # ability_2_norm
            max(0.0, ab3),              # ability_3_norm
            max(0.0, ult_norm),         # ult_charge_norm
            ult_ready_val,              # ult_ready
            pos_x,                      # pos_x (0 if not detected)
            pos_y,                      # pos_y (0 if not detected)
            has_pos,                    # has_position
        ]

        # Accumulate aggregate features
        if is_alive:
            alive[side] += 1
            total_health[side] += health_val
            if ult_ready_val:
                ults_ready[side] += 1
            if role in role_alive:
                role_alive[role][side] += 1
            loadout_quality[side] += WEAPON_TIER_QUALITY.get(weapon_tier_str, 1.0)
            if weapon_tier_str in RIFLE_PLUS_TIERS:
                rifle_count[side] += 1
            if has_pos:
                positions[side].append((pos_x, pos_y))

        if has_pos:
            pos_count[side] += 1

    agg_feats = np.array([
        # Baseline features
        alive["atk"] / 5.0,
        alive["def"] / 5.0,
        total_health["atk"] / 750.0,
        total_health["def"] / 750.0,
        ults_ready["atk"] / 5.0,
        ults_ready["def"] / 5.0,
        spike_planted,
        time_norm,
        round_norm,
        score_diff,
        atk_is_team1,
        # Role-based (normalize by typical max per team — 3 duelists, 2 each for others)
        role_alive["duelist"]["atk"] / 3.0,
        role_alive["duelist"]["def"] / 3.0,
        role_alive["initiator"]["atk"] / 2.0,
        role_alive["initiator"]["def"] / 2.0,
        role_alive["controller"]["atk"] / 2.0,
        role_alive["controller"]["def"] / 2.0,
        role_alive["sentinel"]["atk"] / 2.0,
        role_alive["sentinel"]["def"] / 2.0,
        # Economy (max quality = 5 rifles = 20)
        loadout_quality["atk"] / 20.0,
        loadout_quality["def"] / 20.0,
        rifle_count["atk"] / 5.0,
        rifle_count["def"] / 5.0,
        # Position availability
        pos_count["atk"] / 5.0,
        pos_count["def"] / 5.0,
        # Spike timer
        spike_time_norm,
    ], dtype=np.float32)

    # Spatial aggregates (computed from alive players with detected positions)
    atk_cx, atk_cy, atk_spread = _spatial_aggregates(positions["atk"])
    def_cx, def_cy, def_spread = _spatial_aggregates(positions["def"])
    min_xdist = _min_cross_distance(positions["atk"], positions["def"])
    atk_pos_cov = pos_count["atk"] / max(1, alive["atk"])
    def_pos_cov = pos_count["def"] / max(1, alive["def"])

    # Economy type per team
    atk_econ = classify_economy(loadout_quality["atk"], alive["atk"])
    def_econ = classify_economy(loadout_quality["def"], alive["def"])

    global_feats = np.array([
        # Tactical / temporal
        atk_is_team1,
        round_norm,
        score_diff,
        time_norm,
        spike_planted,
        spike_time_norm,
        # Economy context
        atk_econ,
        def_econ,
        # Team aggregates
        alive["atk"] / 5.0,
        alive["def"] / 5.0,
        ults_ready["atk"] / 5.0,
        ults_ready["def"] / 5.0,
        rifle_count["atk"] / 5.0,
        rifle_count["def"] / 5.0,
        total_health["atk"] / 750.0,
        total_health["def"] / 750.0,
        loadout_quality["atk"] / 20.0,
        loadout_quality["def"] / 20.0,
        # Role-based alive (controller/initiator are highest-value roles)
        role_alive["controller"]["atk"] / 2.0,
        role_alive["controller"]["def"] / 2.0,
        role_alive["initiator"]["atk"] / 2.0,
        role_alive["initiator"]["def"] / 2.0,
        # Spatial aggregates
        atk_pos_cov,
        def_pos_cov,
        atk_cx,
        atk_cy,
        def_cx,
        def_cy,
        atk_spread,
        def_spread,
        min_xdist,
    ], dtype=np.float32)
    assert len(global_feats) == GLOBAL_FEAT_DIM, f"global_feats dim mismatch: {len(global_feats)}"

    return {
        "agg_feats": agg_feats,
        "player_feats": player_feats,
        "agent_ids": agent_ids,
        "weapon_ids": weapon_ids,
        "role_ids": role_ids,
        "global_feats": global_feats,
        "atk_mask": atk_mask,
        "alive_mask": alive_mask,
    }


# ─── Per-map processing ───────────────────────────────────────────────────────

def process_map(
    map_dir: Path,
    agents_config: dict,
) -> tuple[dict, str] | None:
    """
    Process one map directory. Returns (arrays_dict, map_name) or None if unusable.
    map_name is the lowercase map name (e.g. "ascent", "lotus").
    """
    output_dir = map_dir / "output"
    metadata_path = map_dir / "metadata.json"

    with open(metadata_path, encoding="cp1252") as f:
        metadata = json.load(f)

    teams = {t["name"]: t for t in metadata["teams"]}
    team_names = list(teams.keys())
    if len(team_names) < 2:
        print(f"  Skipping {map_dir.name}: < 2 teams in metadata")
        return None

    team1_name = team_names[0]
    team2_name = team_names[1]
    team1_starting_side = teams[team1_name]["starting_side"]

    map_name = metadata.get("map", "unknown").lower()

    round_outcomes = build_round_outcomes(output_dir / "event_log.jsonl")
    if not round_outcomes:
        print(f"  Skipping {map_dir.name}: no round_end events")
        return None

    round_labels = build_round_labels(round_outcomes, team1_name, team2_name, team1_starting_side)

    df = pd.read_csv(output_dir / "frame_states.csv", dtype=str, encoding="cp1252")
    df = df[df["phase"] == "ACTIVE_ROUND"].copy()
    if df.empty:
        print(f"  Skipping {map_dir.name}: no ACTIVE_ROUND frames")
        return None

    # Build player index -> team name from CSV (most common value = ground truth)
    player_teams: dict[int, str] = {}
    for pidx in range(10):
        team_col = f"player_{pidx}_team"
        if team_col in df.columns:
            col_vals = df[team_col].dropna()
            if len(col_vals) > 0:
                player_teams[pidx] = col_vals.value_counts().idxmax()

    accum: dict[str, list] = {
        "agg_feats": [], "player_feats": [], "agent_ids": [],
        "weapon_ids": [], "role_ids": [], "global_feats": [],
        "atk_mask": [], "alive_mask": [], "y": [],
    }
    skipped = 0

    for _, row in df.iterrows():
        rn_raw = row.get("round_number", "")
        try:
            rn = int(float(rn_raw))
        except (ValueError, TypeError):
            skipped += 1
            continue

        if rn not in round_labels:
            skipped += 1
            continue

        # Drop first ~2s of round (game_timer near 100 = buy-phase bleed)
        game_timer_val = _safe_float(row.get("game_timer", None), default=-1.0)
        if game_timer_val > 97.0:
            skipped += 1
            continue

        feats = compute_features(
            row, rn, team1_name, team1_starting_side, player_teams, agents_config
        )

        for k in ("agg_feats", "player_feats", "agent_ids", "weapon_ids",
                  "role_ids", "global_feats", "atk_mask", "alive_mask"):
            accum[k].append(feats[k])
        accum["y"].append(float(round_labels[rn]))

    if not accum["y"]:
        print(f"  Skipping {map_dir.name}: no labeled frames after filtering")
        return None

    result: dict[str, np.ndarray] = {}
    for k in ("agg_feats", "player_feats", "agent_ids", "weapon_ids",
              "role_ids", "global_feats", "atk_mask", "alive_mask", "y"):
        result[k] = np.stack(accum[k])

    n = len(accum["y"])
    atk_wins = int(result["y"].sum())
    print(
        f"  {map_dir.parent.name}/{map_dir.name}: "
        f"{n:,} frames, {atk_wins:,} atk wins, {len(round_labels)} rounds"
    )
    return result, map_name


# ─── Dataset split ────────────────────────────────────────────────────────────

def split_by_match(
    match_data: dict[str, dict],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict, dict, dict]:
    """
    Split by series (match_id) to prevent data leakage across rounds.

    Handles sparse maps gracefully:
      1 series  → all train, no val/test
      2 series  → 1 train + 1 val, no test
      3+ series → standard fractional split
    """
    rng = np.random.default_rng(seed)
    match_ids = sorted(match_data.keys())
    rng.shuffle(match_ids)

    n = len(match_ids)
    if n == 1:
        n_val, n_test = 0, 0
    elif n == 2:
        n_val, n_test = 1, 0
    else:
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))
        # Guarantee at least 1 train series
        while n - n_val - n_test < 1 and (n_val + n_test) > 0:
            if n_test > 0:
                n_test -= 1
            else:
                n_val -= 1

    train_ids = match_ids[: n - n_val - n_test]
    val_ids = match_ids[n - n_val - n_test: n - n_test]
    test_ids = match_ids[n - n_test:] if n_test > 0 else []

    array_keys = list(next(iter(match_data.values())).keys())

    def concat(ids: list[str]) -> dict:
        arrs: dict[str, list] = {k: [] for k in array_keys}
        for mid in ids:
            for k in array_keys:
                arrs[k].append(match_data[mid][k])
        return {k: np.concatenate(v) if v else np.array([]) for k, v in arrs.items()}

    return (
        concat(train_ids),
        concat(val_ids),
        concat(test_ids),
        {"train": train_ids, "val": val_ids, "test": test_ids},
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build per-map training datasets")
    parser.add_argument(
        "--data-dirs", nargs="+", type=Path,
        default=[Path("masters_london_2026")],
        help="Root directories of processed VOD outputs (accepts multiple)",
    )
    parser.add_argument(
        "--only-maps", nargs="+", default=None,
        metavar="SERIES/MAP",
        help=(
            "Restrict to specific maps, e.g. "
            "'nrg_vs_lev/map1_ascent'. "
            "If omitted, all maps with output data are included."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/dataset_masters_london"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    # Load agents config for per-ability max charges
    agents_config_path = Path("src/valoscribe/config/agents_champs2025.json")
    if not agents_config_path.exists():
        print(f"Error: agents config not found: {agents_config_path}", file=sys.stderr)
        sys.exit(1)
    with open(agents_config_path) as f:
        agents_config = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    only_maps = set(args.only_maps) if args.only_maps else None
    if only_maps:
        print(f"Filtering to {len(only_maps)} specified maps")

    print(f"Scanning: {[str(d) for d in args.data_dirs]}")
    map_dirs = find_map_dirs(args.data_dirs, only_maps=only_maps)
    n_series = len({mid for mid, _ in map_dirs})
    print(f"Found {len(map_dirs)} map directories across {n_series} series\n")

    # Collect per-map data: per_map[map_name][match_id] = arrays
    per_map: dict[str, dict[str, dict]] = {}

    for match_id, map_dir in map_dirs:
        out = process_map(map_dir, agents_config)
        if out is None:
            continue
        result, map_name = out
        if map_name not in per_map:
            per_map[map_name] = {}
        if match_id in per_map[map_name]:
            # Same series appears twice for this map (very rare) — concatenate
            for k in result:
                per_map[map_name][match_id][k] = np.concatenate(
                    [per_map[map_name][match_id][k], result[k]]
                )
        else:
            per_map[map_name][match_id] = result

    if not per_map:
        print("No usable data found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'-'*60}")
    print(f"Maps found: {sorted(per_map.keys())}\n")

    shared_metadata = {
        "aggregate_feature_names": AGGREGATE_FEATURE_NAMES,
        "player_feat_names": PLAYER_FEAT_NAMES,
        "global_feat_names": GLOBAL_FEAT_NAMES,
        "player_feat_dim": PLAYER_FEAT_DIM,
        "n_agg_feats": N_AGG_FEATS,
        "n_global_feats": N_GLOBAL_FEATS,
        "agent_ids": AGENT_IDS,
        "agent_roles": AGENT_ROLES,
        "role_ids": ROLE_IDS,
        "weapon_ids": WEAPON_IDS,
        "weapon_tier_by_name": WEAPON_TIER_BY_NAME,
        "weapon_tier_quality": WEAPON_TIER_QUALITY,
        "num_agents": NUM_AGENTS,
        "num_roles": NUM_ROLES,
        "num_weapons": NUM_WEAPONS,
    }

    for map_name in sorted(per_map.keys()):
        match_data = per_map[map_name]
        n_series_map = len(match_data)
        total_frames = sum(len(d["y"]) for d in match_data.values())
        total_atk = sum(float(d["y"].sum()) for d in match_data.values())

        print(f"Map: {map_name.upper()}")
        print(f"  Series: {n_series_map}  Frames: {total_frames:,}  Atk win rate: {total_atk / total_frames:.1%}")
        if n_series_map < 3:
            print(f"  Warning: only {n_series_map} series — val/test splits will be minimal or absent")

        train_data, val_data, test_data, splits = split_by_match(
            match_data, args.val_fraction, args.test_fraction
        )

        map_out = args.output_dir / map_name
        map_out.mkdir(parents=True, exist_ok=True)

        for split_name, split_dict in [("train", train_data), ("val", val_data), ("test", test_data)]:
            np.savez_compressed(map_out / f"{split_name}.npz", **split_dict)
            n = len(split_dict["y"]) if split_dict.get("y") is not None and len(split_dict.get("y", [])) > 0 else 0
            atk_w = int(split_dict["y"].sum()) if n > 0 else 0
            print(f"  {split_name}: {n:,} frames" + (f" ({atk_w / n:.1%} atk win rate)" if n else ""))

        map_metadata = {**shared_metadata, "map": map_name, "series": splits}
        with open(map_out / "metadata.json", "w") as f:
            json.dump(map_metadata, f, indent=2)
        with open(map_out / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)

        print(f"  Wrote: {map_out}/\n")

    print(f"Dataset saved to {args.output_dir}/")
    print(f"  One subdirectory per map: {sorted(per_map.keys())}")
    print(f"  Each map dir contains: train.npz, val.npz, test.npz, metadata.json, splits.json")
    print(f"  Per-player arrays: player_feats (N,10,{PLAYER_FEAT_DIM}), agent_ids, weapon_ids, role_ids")


if __name__ == "__main__":
    main()
