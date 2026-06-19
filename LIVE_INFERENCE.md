# Live Round Win Probability — Architecture & Implementation Plan

This document describes how to extend the existing valoscribe pipeline to run the round win predictor in real time on a live broadcast or VOD, without pre-processing to `frame_states.csv` first.

---

## Current Architecture (Offline)

The existing pipeline has four stages:

```
Video frames
    │
    ▼
GameStateManager.process_frame()        ← runs all detectors, maintains state
    │
    ▼
OutputWriter.write_frame_state()        ← serializes state to frame_states.csv
    │
    ▼
build_dataset.py: compute_features()    ← reads CSV rows → numpy tensors
    │
    ▼
RoundPredictor.forward()                ← outputs P(attack wins)
```

The critical insight: `GameStateManager` already holds the full game state in memory after every `process_frame()` call. The CSV is just a serialized snapshot of that state. `compute_features()` in `build_dataset.py` then deserializes the CSV row back into tensors. For live inference, the CSV in the middle is unnecessary — we can go directly from live state to tensors to model.

**Target architecture:**

```
Video frames
    │
    ▼
GameStateManager.process_frame()        ← unchanged
    │
    ├──► OutputWriter (optional, same as before)
    │
    └──► FramePredictor.predict_from_state()   ← NEW
              │
              ▼
         RoundPredictor.forward()
              │
              ▼
         P(attack wins) → stdout / overlay / websocket
```

---

## What GameStateManager Already Knows

After every `process_frame()` call, the following state is available in memory:

### RoundManager (`gsm.round_manager`)
- `current_round` — round number (1-indexed)
- `current_phase` — `Phase.PREROUND`, `Phase.ACTIVE_ROUND`, `Phase.POST_ROUND`, etc.
- `score_team1`, `score_team2` — current score
- `game_timer` — seconds remaining in round (counts down from ~100)
- `spike_timer` — seconds remaining on spike, or None if not planted
- `team1_name`, `team2_name` — team names
- `team1_starting_side` — "attack" or "defense" for round 1 (used to derive side for all rounds)

### PlayerStateTracker × 10 (`gsm.player_trackers[i]`)
- `name` — player IGN
- `team` — team name
- `agent` — agent name (lowercase)
- `alive` — bool
- `health`, `armor` — int
- `ability_1_charges`, `ability_2_charges`, `ability_3_charges` — int
- `ultimate_charges`, `ultimate_full` — int, bool
- `weapon` — weapon name (lowercase), or None
- `pos_x`, `pos_y` — normalized map coordinates [0, 1], or None if not detected

### MinimapDetector (already ran inside process_frame)
- Positions are written directly into `PlayerStateTracker.pos_x` / `pos_y`

All of this is what `write_frame_state()` serializes into a CSV row. `compute_features()` then reads it back. The roundtrip through CSV is the only thing standing between the live state and the model.

---

## The Gaps to Fill

### Gap 1 — Feature extraction from live objects

`compute_features()` in `build_dataset.py` takes a `pd.Series` (a CSV row). We need a parallel function:

```python
def compute_features_from_state(gsm: GameStateManager, agents_config: dict) -> dict:
    """
    Same logic as compute_features() in build_dataset.py,
    but reads directly from live GameStateManager state instead of a CSV row.
    Returns the same dict of numpy arrays.
    """
```

Internally, instead of:
```python
health_val = _safe_float(row.get("player_0_health", 0))
```

it reads:
```python
health_val = float(gsm.player_trackers[0].health or 0)
```

The math is identical — same normalization, same spatial aggregates, same economy classification. Only the data source changes. This is approximately 100–150 lines.

**Key mapping from CSV columns to live attributes:**

| CSV column | Live source |
|---|---|
| `round_number` | `gsm.round_manager.current_round` |
| `game_timer` | `gsm.round_manager.game_timer` |
| `spike_timer` | `gsm.round_manager.spike_timer` |
| `score_team1` | `gsm.round_manager.score_team1` |
| `score_team2` | `gsm.round_manager.score_team2` |
| `player_N_alive` | `gsm.player_trackers[N].alive` |
| `player_N_health` | `gsm.player_trackers[N].health` |
| `player_N_armor` | `gsm.player_trackers[N].armor` |
| `player_N_agent` | `gsm.player_trackers[N].agent` |
| `player_N_weapon` | `gsm.player_trackers[N].weapon` |
| `player_N_ability_1` | `gsm.player_trackers[N].ability_1_charges` |
| `player_N_ultimate_charges` | `gsm.player_trackers[N].ultimate_charges` |
| `player_N_ultimate_full` | `gsm.player_trackers[N].ultimate_full` |
| `player_N_pos_x` | `gsm.player_trackers[N].pos_x` |
| `player_N_pos_y` | `gsm.player_trackers[N].pos_y` |
| `player_N_team` | `gsm.player_trackers[N].team` |

The `atk_is_team1` and `get_team1_side()` logic is unchanged — it uses `round_manager.team1_starting_side` and `round_manager.current_round`, both already available.

---

### Gap 2 — Knowing the attacking side

`compute_features()` calls `get_team1_side(round_num, team1_starting_side)` to determine which team is on attack for the current round. This handles the mid-game side swap (rounds 1–12 one side, 13–24 flipped, 25+ alternating OT).

`team1_starting_side` currently comes from `metadata.json` (scraped from VLR before processing). For live inference, this is available as `gsm.round_manager.team1_starting_side` — the `RoundManager` already infers this from the HUD during processing.

**OT edge case**: In overtime (rounds 25+), `get_team1_side()` correctly alternates sides each round. As long as `RoundManager` correctly increments `current_round` through OT, this works without any changes. Verify that `RoundManager` handles OT round counting before deploying on OT maps.

---

### Gap 3 — Map-to-model routing

We train one model per map (`models/masters_london/fracture/best_model.pt`, etc.). For live inference, we need to load the right checkpoint. Two options:

**Option A: Pre-specify the map (recommended for now)**

User passes `--map fracture` to the CLI. The `FramePredictor` loads `{model_dir}/{map}/best_model.pt` at startup and keeps it in memory for the session. Zero runtime overhead — model is loaded once.

```bash
valoscribe orchestrate process-vod match.mp4 metadata.json \
  --output output/ --fps 2 \
  --predict-model models/masters_london --map fracture
```

This matches the existing workflow — we already know the map before processing starts because it's in `metadata.json`.

**Option B: Auto-detect map from minimap**

The minimap crop (450×450) has a visually distinct layout per map. Options:
- Template match against a reference minimap image per map at startup
- CNN classifier on the first few PREROUND minimap crops
- Compare against the existing `minimap_mask_<mapname>.png` files

This is more complex and probably not necessary — `metadata.json` always contains the map name, and even for truly "live" use you know the map before the match starts. Auto-detection is a nice-to-have for a future fully-automated overlay.

---

### Gap 4 — No labels needed

In training, every frame is labeled with the round's eventual outcome. For live inference, the outcome is unknown — that's the whole point. The model just runs `forward()` on the current state and emits a probability. No label lookup needed.

The only implication: we only call the model when `phase == Phase.ACTIVE_ROUND`. During PREROUND, the game timer is at ~100s and most player states are in buy-phase (not useful for a round win predictor). The first ~2–3 seconds of ACTIVE_ROUND are also noisy (same filter as `build_dataset.py`: skip frames where `game_timer > 97`).

---

### Gap 5 — Probability smoothing

At 2fps, consecutive frames are nearly identical (same alive counts, same positions, same timer) so the probability series is already fairly smooth. However, detection errors can cause single-frame spikes:
- A player momentarily misdetected as dead → alive count drops → probability swings
- OCR noise on a timer digit → time_remaining jumps

Recommended: keep a rolling buffer of the last N predictions and output a smoothed value (median or exponential moving average). N=3 (1.5 seconds at 2fps) is usually enough to eliminate single-frame noise without adding perceptible lag.

For a real-time overlay at higher fps (e.g. 10fps), a longer window (N=10–15) makes sense.

---

## Implementation Plan

### New file: `src/valoscribe/inference/frame_predictor.py`

```python
class FramePredictor:
    def __init__(self, model_dir: Path, map_name: str, agents_config: dict,
                 smoothing_window: int = 3):
        # Load model checkpoint
        self.model = load_model(model_dir, map_name)
        self.model.eval()
        self.agents_config = agents_config
        self._buffer: deque[float] = deque(maxlen=smoothing_window)

    def predict_from_state(self, gsm) -> float | None:
        """
        Run inference on the current GameStateManager state.
        Returns smoothed P(attack wins), or None if phase is not ACTIVE_ROUND
        or game_timer > 97 (buy-phase bleed).
        """
        if gsm.round_manager.current_phase != Phase.ACTIVE_ROUND:
            return None
        if (gsm.round_manager.game_timer or 0) > 97:
            return None

        feats = compute_features_from_state(gsm, self.agents_config)
        prob = self._run_model(feats)
        self._buffer.append(prob)
        return float(np.median(self._buffer))

    def _run_model(self, feats: dict) -> float:
        # Same tensor construction as predict_frames.py
        ...
```

### Changes to `GameStateManager`

In `process_frame()`, after calling `write_frame_state()`:

```python
# Existing code
self.output_writer.write_frame_state(...)

# New: live inference (only if FramePredictor is configured)
if self.frame_predictor is not None:
    prob = self.frame_predictor.predict_from_state(self)
    if prob is not None:
        self._emit_prediction(prob)
```

`_emit_prediction()` can write to stdout, a websocket, a file, or an overlay — pluggable output.

### Changes to `orchestrate.py`

Add two new CLI options:

```python
predict_model: Optional[Path] = typer.Option(
    None, "--predict-model",
    help="Path to model directory (e.g. models/masters_london). "
         "If set, runs live round win probability on each ACTIVE_ROUND frame.",
)
map_name: Optional[str] = typer.Option(
    None, "--map",
    help="Map name for model routing (e.g. 'fracture'). "
         "Required when --predict-model is set. Defaults to map in metadata.json.",
)
```

If `predict_model` is set:
1. Read map name from `metadata.json` if `--map` not provided
2. Load `FramePredictor(predict_model, map_name, agents_config)`
3. Pass it to `GameStateManager.__init__`

---

## Data Flow for a Live Match

1. **Before the match**: Scrape VLR with `generate_match_metadata.py` to get team names, agents, and starting sides. This takes ~5 seconds and can be done as soon as lineups are announced.

2. **At process startup**: Load the map's model checkpoint into `FramePredictor`. The checkpoint is ~200KB so this is instant.

3. **Every frame** (at whatever fps the pipeline runs):
   - All existing detectors run: health, alive, abilities, ult, weapon, position
   - `GameStateManager` updates all `PlayerStateTracker` objects
   - After state update, `FramePredictor.predict_from_state()` is called
   - If `ACTIVE_ROUND` and `game_timer <= 97`: build tensors, run model forward pass, emit probability
   - Otherwise: emit nothing (PREROUND / POST_ROUND / etc.)

4. **Output**: A timestamped probability stream, e.g.:
   ```
   [round=14  timer=67s]  ATK 38.2%  DEF 61.8%
   [round=14  timer=65s]  ATK 37.9%  DEF 62.1%
   [round=14  timer=65s]  SPIKE PLANTED  ATK 72.4%  DEF 27.6%
   ```

---

## Performance Considerations

**CPU inference**: The model is 55k parameters — a forward pass takes <1ms on CPU. At 2fps, this adds negligible overhead. Even at 30fps it would be fast enough to run in real time on any modern machine.

**Memory**: The model checkpoint is ~200KB. One model is kept in memory per session (one map at a time).

**Bottleneck**: The bottleneck is still the existing detectors (minimap template matching, OCR for timers/scores), not the model. This is unchanged.

**GPU**: Not needed for inference at this model size. If GPU is available, `torch.device("cuda")` can be set in `FramePredictor.__init__` for marginal speedup.

---

## What the Pipeline Does NOT Handle Yet

**1. Agent auto-detection on first frame**

Currently, agent assignments (which player plays which agent) come from `metadata.json`. In practice, the game HUD doesn't label agents by player name mid-round — it only shows them in the buy phase or pre-game screen. The existing pipeline handles this via `metadata.json` scraped from VLR. For a fully live use case (no VLR ID known in advance), we would need to detect agent portraits from the HUD at match start.

This is already partially done — `MinimapDetector` uses per-agent minimap icon templates for position detection. The agent identity for each player slot is resolved at startup from metadata, not from per-frame detection.

**2. Weapon detection confidence**

The weapon detector occasionally misses (emits `None` or `nan` for a player). In the offline pipeline, missing weapons fall back to `unknown` weapon embedding (index 0). This is the same behavior for live inference — no special handling needed, but it does slightly degrade accuracy for players whose weapons aren't detected.

**3. Side detection from HUD (vs metadata)**

`team1_starting_side` currently comes from metadata. A fully metadata-free implementation would need to infer starting sides from the HUD (the side icon next to each team's name in the scoreboard, visible during buy phase). This is doable but adds a detector.

**4. Multiple concurrent maps**

For a broadcast covering multiple matches in parallel (e.g. a bracket stage where two maps play simultaneously), you'd need multiple `FramePredictor` instances with different checkpoints. The architecture supports this — just instantiate one per map.

---

## Testing the Live Path Before Full Integration

Before wiring into `GameStateManager`, you can validate the feature extraction gap independently:

1. Take one already-processed map (has `frame_states.csv`)
2. For a few frames, compute features both ways:
   - `compute_features(csv_row, ...)` — existing offline path
   - `compute_features_from_state(gsm_replay, ...)` — new live path, replaying state from CSV
3. Assert the tensors are identical (within float32 precision)

This gives confidence that the live path produces the same inputs the model was trained on before deploying it on an actual live feed.

---

## Summary

| Component | Status | Work needed |
|---|---|---|
| Frame detection (health, alive, abilities, ult, weapon, position) | Done | None |
| Round/phase/score/timer tracking | Done | None |
| `compute_features()` logic | Done (reads CSV) | Port to read from live state (~100 lines) |
| RoundPredictor model | Done | None |
| `FramePredictor` class | Not started | ~150 lines |
| `GameStateManager` integration | Not started | ~20 lines |
| `orchestrate.py` CLI flags | Not started | ~15 lines |
| Metadata scrape (team names, starting sides) | Done | None (already in workflow) |
| Map → model routing | Done implicitly | Add `--map` flag |
| Probability smoothing | Not started | ~10 lines (rolling buffer) |

**Total new code: ~300 lines.** The architecture is already there. This is integration work, not a redesign.
