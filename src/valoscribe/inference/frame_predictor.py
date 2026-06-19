"""
frame_predictor.py — live round win probability from a live GameStateManager.

Reads directly from GameStateManager state after every ACTIVE_ROUND frame,
builds the same feature tensors as build_dataset.py, runs RoundPredictor,
and emits a smoothed P(attack wins) probability.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

# Resolve project root and pull in build_dataset constants and helpers.
# frame_predictor.py lives at src/valoscribe/inference/; root is 3 levels up.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from build_dataset import (  # noqa: E402
    AGENT_IDS,
    AGENT_ROLES,
    GLOBAL_FEAT_DIM,
    PLAYER_FEAT_DIM,
    RIFLE_PLUS_TIERS,
    ROLE_IDS,
    WEAPON_IDS,
    WEAPON_TIER_BY_NAME,
    WEAPON_TIER_QUALITY,
    classify_economy,
    get_team1_side,
    _min_cross_distance,
    _spatial_aggregates,
)

from valoscribe.models.round_predictor import ModelConfig, RoundPredictor
from valoscribe.orchestration.phase_detector import Phase

if TYPE_CHECKING:
    from valoscribe.orchestration.game_state_manager import GameStateManager


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(model_dir: Path, map_name: str) -> RoundPredictor:
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


# ── Feature extraction from live state ────────────────────────────────────────

def _f(val) -> float:
    """Safely coerce a tracker value to float; returns 0.0 for None/NaN."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if f != f else f  # NaN guard
    except (ValueError, TypeError):
        return 0.0


def compute_features_from_state(gsm: "GameStateManager", agents_config: dict) -> dict:
    """
    Compute all feature arrays from live GameStateManager state.

    Mirrors compute_features() in build_dataset.py exactly — same math, same
    normalization, same tensor shapes — but reads from live player_trackers
    instead of a CSV row.

    Returns the same dict as compute_features():
        player_feats:  (10, PLAYER_FEAT_DIM) float32
        agent_ids:     (10,) int32
        weapon_ids:    (10,) int32
        role_ids:      (10,) int32
        global_feats:  (GLOBAL_FEAT_DIM,) float32
        atk_mask:      (10,) float32
        alive_mask:    (10,) float32
        agg_feats:     (N_AGG_FEATS,) float32
    """
    rm = gsm.round_manager
    round_num = rm.current_round
    team1_name = rm.team_names[0]
    team1_starting_side = rm.starting_sides["team1"]
    team1_side = get_team1_side(round_num, team1_starting_side)
    atk_is_team1 = 1.0 if team1_side == "attack" else 0.0

    # Timers — cached by _write_frame_state on the same frame
    last_timers: dict = getattr(gsm, "_last_timers", {}) or {}
    spike_timer_val = float(last_timers.get("spike_timer") or 0.0)
    game_timer_val = float(last_timers.get("game_timer") or 0.0)
    spike_planted = 1.0 if spike_timer_val > 0 else 0.0
    spike_time_norm = min(spike_timer_val / 45.0, 1.0)
    time_norm = min(game_timer_val / 100.0, 1.0)

    # Scores
    score_t1 = float(rm.current_score["team1"])
    score_t2 = float(rm.current_score["team2"])
    atk_score = score_t1 if team1_side == "attack" else score_t2
    def_score = score_t2 if team1_side == "attack" else score_t1
    score_diff = (atk_score - def_score) / 13.0
    round_norm = min(round_num, 30) / 30.0

    # Per-player output arrays
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
    positions: dict[str, list[tuple[float, float]]] = {"atk": [], "def": []}

    trackers = gsm.player_trackers or []
    for pidx in range(10):
        if pidx >= len(trackers):
            break
        tracker = trackers[pidx]
        state = tracker.current_state
        meta = tracker.metadata

        pteam = meta.get("team", "")
        is_atk = (pteam == team1_name) == (team1_side == "attack")
        side = "atk" if is_atk else "def"
        atk_mask[pidx] = 1.0 if is_atk else 0.0

        is_alive = bool(state.get("alive", False))
        alive_mask[pidx] = 1.0 if is_alive else 0.0

        agent_name = str(meta.get("agent", "") or "").lower().strip()
        agent_ids[pidx] = AGENT_IDS.get(agent_name, 0)
        role = AGENT_ROLES.get(agent_name, "")
        role_ids[pidx] = ROLE_IDS.get(role, 0)

        weapon_name = str(state.get("weapon", "") or "").lower().strip()
        weapon_ids[pidx] = WEAPON_IDS.get(weapon_name, 0)
        weapon_tier_str = WEAPON_TIER_BY_NAME.get(weapon_name, "unknown")

        health_val = _f(state.get("health"))
        armor_val = _f(state.get("armor"))

        agent_cfg = agents_config.get(agent_name, {})
        ab1_max = max(1, (agent_cfg.get("ability_1") or {}).get("max_charges", 1) or 1)
        ab2_max = max(1, (agent_cfg.get("ability_2") or {}).get("max_charges", 1) or 1)
        ab3_max = max(1, (agent_cfg.get("ability_3") or {}).get("max_charges", 1) or 1)
        ult_max = max(1, (agent_cfg.get("ultimate") or {}).get("max_charges", 7) or 7)

        ab1 = min(_f(state.get("ability_1")) / ab1_max, 1.0)
        ab2 = min(_f(state.get("ability_2")) / ab2_max, 1.0)
        ab3 = min(_f(state.get("ability_3")) / ab3_max, 1.0)
        ult_info = state.get("ultimate") or {}
        ult_norm = min(_f(ult_info.get("charges", 0)) / ult_max, 1.0)
        ult_ready_val = 1.0 if bool(ult_info.get("is_full", False)) else 0.0

        raw_pos_x = state.get("pos_x")
        raw_pos_y = state.get("pos_y")
        has_pos = 1.0 if (raw_pos_x is not None and raw_pos_y is not None) else 0.0
        pos_x = float(raw_pos_x) if raw_pos_x is not None else 0.0
        pos_y = float(raw_pos_y) if raw_pos_y is not None else 0.0

        player_feats[pidx] = [
            1.0 if is_alive else 0.0,
            health_val / 150.0,
            armor_val / 50.0,
            max(0.0, ab1),
            max(0.0, ab2),
            max(0.0, ab3),
            max(0.0, ult_norm),
            ult_ready_val,
            pos_x,
            pos_y,
            has_pos,
        ]

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
        role_alive["duelist"]["atk"] / 3.0,
        role_alive["duelist"]["def"] / 3.0,
        role_alive["initiator"]["atk"] / 2.0,
        role_alive["initiator"]["def"] / 2.0,
        role_alive["controller"]["atk"] / 2.0,
        role_alive["controller"]["def"] / 2.0,
        role_alive["sentinel"]["atk"] / 2.0,
        role_alive["sentinel"]["def"] / 2.0,
        loadout_quality["atk"] / 20.0,
        loadout_quality["def"] / 20.0,
        rifle_count["atk"] / 5.0,
        rifle_count["def"] / 5.0,
        pos_count["atk"] / 5.0,
        pos_count["def"] / 5.0,
        spike_time_norm,
    ], dtype=np.float32)

    atk_cx, atk_cy, atk_spread = _spatial_aggregates(positions["atk"])
    def_cx, def_cy, def_spread = _spatial_aggregates(positions["def"])
    min_xdist = _min_cross_distance(positions["atk"], positions["def"])
    atk_pos_cov = pos_count["atk"] / max(1, alive["atk"])
    def_pos_cov = pos_count["def"] / max(1, alive["def"])
    atk_econ = classify_economy(loadout_quality["atk"], alive["atk"])
    def_econ = classify_economy(loadout_quality["def"], alive["def"])

    global_feats = np.array([
        atk_is_team1,
        round_norm,
        score_diff,
        time_norm,
        spike_planted,
        spike_time_norm,
        atk_econ,
        def_econ,
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
        role_alive["controller"]["atk"] / 2.0,
        role_alive["controller"]["def"] / 2.0,
        role_alive["initiator"]["atk"] / 2.0,
        role_alive["initiator"]["def"] / 2.0,
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


# ── FramePredictor ─────────────────────────────────────────────────────────────

class FramePredictor:
    """
    Live round win probability predictor.

    Wraps a trained RoundPredictor checkpoint and runs inference directly from
    a live GameStateManager state after each ACTIVE_ROUND frame. Applies a
    rolling median buffer to smooth single-frame detection noise.
    """

    def __init__(
        self,
        model_dir: Path,
        map_name: str,
        agents_config: Optional[dict] = None,
        smoothing_window: int = 3,
    ):
        """
        Args:
            model_dir:          Directory containing per-map model subdirs.
            map_name:           Map name matching the subdir (e.g. "fracture").
            agents_config:      Agent ability config dict. If None, loads default.
            smoothing_window:   Rolling median window size. 3 = 1.5s at 2fps.
        """
        self.map_name = map_name
        self.model = _load_model(Path(model_dir), map_name)
        self._buffer: deque[float] = deque(maxlen=smoothing_window)

        if agents_config is None:
            cfg_path = Path(__file__).parent.parent / "config" / "agents_champs2025.json"
            agents_config = json.loads(cfg_path.read_text())
        self.agents_config = agents_config

        self.last_prob: Optional[float] = None  # updated by predict_from_state; readable by capture loop

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[FramePredictor] {map_name} model loaded ({n_params:,} params) from {model_dir}")

    def predict_from_state(self, gsm: "GameStateManager", timestamp: float) -> Optional[float]:
        """
        Run inference on the current GameStateManager state.

        Returns smoothed P(attack wins) in [0, 1], or None when:
        - phase is not ACTIVE_ROUND
        - player trackers not yet initialized
        - game_timer > 97s (buy-phase bleed-through at round start)
        """
        if gsm.current_phase != Phase.ACTIVE_ROUND:
            return None
        if gsm.player_trackers is None:
            return None

        last_timers: dict = getattr(gsm, "_last_timers", {}) or {}
        game_timer = float(last_timers.get("game_timer") or 0.0)
        if game_timer > 97.0:
            return None

        feats = compute_features_from_state(gsm, self.agents_config)
        prob = self._run_model(feats)
        self._buffer.append(prob)
        smoothed = float(np.median(self._buffer))
        self.last_prob = smoothed
        return smoothed

    def _run_model(self, feats: dict) -> float:
        def t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype).unsqueeze(0)

        with torch.no_grad():
            prob = self.model.predict_proba(
                continuous_feats=t(feats["player_feats"]),
                agent_ids=t(feats["agent_ids"], dtype=torch.long),
                weapon_ids=t(feats["weapon_ids"], dtype=torch.long),
                role_ids=t(feats["role_ids"], dtype=torch.long),
                global_feats=t(feats["global_feats"]),
                atk_mask=t(feats["atk_mask"]),
                alive_mask=t(feats["alive_mask"]),
            )
        return float(prob.squeeze())
