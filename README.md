# xValorant — Live Round Win Predictor

Real-time round win probability for professional Valorant broadcasts. Point it at a fullscreen stream and it outputs P(attack wins) live, updated at 2 fps, displayed in an always-on-top overlay.

[![Demo](https://img.youtube.com/vi/xKsUO7c6Hbw/maxresdefault.jpg)](https://youtu.be/xKsUO7c6Hbw)

### 🎥 Showcase

[![Showcase](https://drive.google.com/thumbnail?id=1asZzbtpJRzkTrqYvJhKXxCW4k8lsZSl8&sz=w1280)](https://drive.google.com/file/d/1asZzbtpJRzkTrqYvJhKXxCW4k8lsZSl8/view?usp=sharing)

▶ **[Watch the full showcase](https://drive.google.com/file/d/1asZzbtpJRzkTrqYvJhKXxCW4k8lsZSl8/view?usp=sharing)** (Google Drive)

---

## How It Works

Valorant has no public positional API. Everything here runs purely from broadcast video — no game client, no Riot API, no external data feeds.

The pipeline has three stages:

**1. Computer vision (valoscribe)** — reads the spectator HUD in real time: health, armor, alive status, ability charges, ultimate charge, weapon, minimap position, spike timer, and round clock for all 10 players simultaneously.

**2. Feature engineering** — constructs a per-frame feature vector (economy, player states, map position aggregates, game context) matching the format the model was trained on.

**3. DeepSets neural network** — permutation-invariant per-player encoder + global head outputs a single probability: P(attack team wins this round). Smoothed over a 3-frame rolling median to reduce jitter.

The live overlay updates every 0.5 s, colors each team's label on a red→yellow→green gradient, and auto-detects which team is attacking each round (including halftime swap and overtime).

---

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows installer linked)
- A 1080p display running the broadcast fullscreen (or windowed at native 1080p)

```bash
git clone https://github.com/SphinxNumberNine/xValorant.git
cd xValorant
uv sync
```

---

## Running the Live Predictor

### Step 1 — Get match metadata

The predictor needs to know which agents each player is playing and which team starts on which side. Pull this from VLR.gg before the match:

```bash
python scripts/generate_match_metadata.py <vlr_match_id> <map_number>
# Example:
python scripts/generate_match_metadata.py 12345 1
# Writes to: masters_london_2026/<team_a>_vs_<team_b>/map1_<mapname>/metadata.json
```

### Step 2 — Start the overlay

```bash
python scripts/live_capture.py \
  --metadata masters_london_2026/fut_esports_vs_nrg/map1_lotus/metadata.json \
  --model-dir models/masters_london \
  --monitor 1
```

The overlay appears in the top-right corner of your screen. Click and drag it anywhere to reposition. Press **swap sides** to flip the left/right label order if the team names appear on the wrong side.

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--monitor` | `1` | Display index to capture (1 = primary) |
| `--fps` | `2.0` | Capture rate in frames per second |
| `--map` | from metadata | Map name override (e.g. `lotus`) |
| `--no-overlay` | off | Console-only mode, no tkinter window |
| `--quiet` | off | Suppress valoscribe processing logs |
| `--output` | `output/live` | Directory for debug `frame_states.csv` |

Close the overlay window or press Ctrl+C to stop.

### Notes on startup

- The predictor needs to see one **buy phase** screen before it can initialize player tracking. Expect ~10–30 seconds of "waiting for ACTIVE_ROUND" at startup — this is normal.
- The overlay shows "Waiting for ACTIVE\_ROUND..." until the first round begins. Start the script before the buy phase for immediate pickup.
- Predictions are automatically suppressed during buy phases and the first 97 seconds of bleedover.

---

## Trained Models

Pre-trained per-map models for **Masters London 2026** are included in `models/masters_london/`:

| Map | Val AUC | Test AUC | Test Brier |
|-----|---------|----------|------------|
| Breeze | 0.889 | 0.822 | 0.175 |
| Pearl | 0.688 | 0.786 | 0.170 |
| Lotus | 0.784 | 0.766 | 0.189 |
| Haven | 0.598 | 0.747 | 0.243 |
| Fracture | 0.806 | 0.709 | 0.208 |
| Ascent | 0.819 | 0.667 | 0.243 |
| Split | 0.834 | 0.665 | 0.238 |

Trained on professional VCT broadcast data (Masters London 2026 + VCT Americas Stage 1 playoffs), 45 maps across 15 series. Test AUC is measured on held-out matches (split by match, never by round). Performance may degrade on maps with limited training data or compositions uncommon at the training event.

---

## Training Your Own Model

To train on a new tournament or extend coverage, run the full pipeline:

### 1. Process VODs

Scrape metadata and process match videos with valoscribe:

```bash
# Process a single series from VLR.gg
./scripts/process_vlr_series.sh https://www.vlr.gg/12345/team-a-vs-team-b

# Or process many series in parallel
./scripts/process_all_series_parallel.sh scripts/matches.txt 6
```

Each processed map produces `frame_states.csv` (frame-by-frame player states) and `event_log.jsonl` (timestamped game events).

### 2. Build the dataset

```bash
python scripts/build_dataset.py \
  --data-dirs masters_london_2026 \
  --output-dir data/dataset_my_event
```

Labels each ACTIVE\_ROUND frame with the round outcome, engineers features, and splits by match (never by round) into per-map train/val/test `.npz` files. Pass several roots to `--data-dirs` to combine events.

### 3. Train

```bash
python scripts/train_model.py \
  --dataset-dir data/dataset_my_event \
  --output-dir models/my_event
```

Trains one per-map DeepSets transformer for every map in the dataset (it loops over all maps — there is no `--map` flag). The best checkpoint per map is saved to `models/my_event/<map>/best_model.pt`.

### 4. Evaluate

Per-map validation/test **Brier, log loss, and AUC** plus reliability diagrams (`reliability_{val,test}.png`) are written automatically during the training run above — there is no separate evaluation pass for the DeepSets models. (`scripts/evaluate_model.py` scores only the legacy XGBoost baseline.)

---

## Architecture

```
Broadcast video (1080p, 2 fps)
        │
        ▼
  valoscribe CV pipeline
  ├── Template matching: health, armor, armor tier, alive, spike, round clock
  ├── OCR: credits, ultimate charge
  ├── Agent detection: preround scoreboard + active HUD icons
  ├── Weapon detection: template matching on HUD weapon slot
  └── Minimap detector: agent icon matching → normalized (x, y) positions
        │
        ▼
  compute_features_from_state()
  ├── Per-player (×10): health, armor, alive, ability charges, ult charge,
  │   weapon tier, credits, position (x, y), has_position flag
  ├── Team aggregates: total health, alive count, ults ready, total credits
  └── Global: map, side, round number, score diff, game timer, spike planted, spike timer
        │
        ▼
  DeepSets + Transformer
  ├── Shared MLP per player → h_i
  ├── Split into ATK/DEF sets → mean pool per team
  ├── Concat with global features
  └── MLP head → sigmoid → P(attack wins)
        │
        ▼
  Rolling median (window=3) → overlay + console
```

---

## Limitations

- **1080p broadcast HUD only.** The coordinate config and templates are calibrated for the standard VCT spectator HUD at 1080p. Player-POV streams and non-standard broadcast overlays won't work.
- **Requires one buy phase to initialize.** Agent identities are read from the scoreboard during PREROUND. If you start mid-round, the predictor waits for the next buy phase.
- **Per-tournament models.** The included models are trained on Masters London 2026 data. Cross-tournament generalization is untested; retrain on new event data for best results.
- **Maps outside the pool.** Only maps present in the training data have models. Requesting a model for an unseen map returns an error.
- **Processing speed.** VOD processing runs at ~20–40 minutes per map at 4 FPS on a modern laptop. Batch processing multiple matches is supported via `process_all_series_parallel.sh`.

---

## Repo Layout

```
scripts/
  live_capture.py          # Live overlay — main entry point
  generate_match_metadata.py
  build_dataset.py
  train_model.py
  evaluate_model.py
  process_vlr_series.sh
  predict_frames.py        # Run predictor on a processed VOD offline

src/valoscribe/
  commands/                # CLI (valoscribe orchestrate, scrape, detect, extract)
  detectors/               # CV detectors (health, weapon, minimap, agent, …)
  inference/               # FramePredictor — live inference wrapper
  models/                  # RoundPredictor PyTorch model definition
  orchestration/           # GameStateManager, phase detection, state tracking
  config/                  # HUD coordinate configs, minimap homographies
  templates/               # Template images for all detectors

models/
  masters_london/          # Per-map trained checkpoints (in use)
  baseline/                # XGBoost baseline (reference)

data/
  dataset_masters_london/  # Pre-built train/val/test splits (.npz)
```
