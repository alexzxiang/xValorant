"""
Extract weapon crops for frames labeled 'unknown' so missing weapon templates
(shorty, bucky, marshal, ares, etc.) can be identified and added.

Samples unknown-weapon frames evenly across each VOD, extracts the weapon icon
crop from each player, and saves PNGs to vods/weapon_review/<map>/.

Usage:
    python scripts/extract_unknown_weapon_crops.py

After running, look through vods/weapon_review/ and copy+rename any crops that
show a missing weapon into src/valoscribe/templates/weapons/<name>N.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd
from valoscribe.detectors.cropper import Cropper


VODS = [
    {
        "name": "haven",
        "video": Path("vods/DRX vs. NRG — VALORANT Champions Paris — Group Stage — Map 01.f399.mp4"),
        "csv":   Path("champs2025_processed_vods/542208_drx_vs_nrg/map1_haven/output/frame_states.csv"),
    },
    {
        "name": "ascent",
        "video": Path("vods/PRX vs. GX — VALORANT Champions Paris — Group Stage — Map 01 - VALORANT Champions Tour (1080p).mp4"),
        "csv":   Path("champs2025_processed_vods/542197_paper_rex_vs_giantx/map1_ascent/output/frame_states.csv"),
    },
    {
        "name": "lotus",
        "video": Path("vods/PRX vs. GX — VALORANT Champions Paris — Group Stage — Map 02 - VALORANT Champions Tour (1080p).mp4"),
        "csv":   Path("champs2025_processed_vods/542197_paper_rex_vs_giantx/map2_lotus/output/frame_states.csv"),
    },
]

# How many unknown crops to extract per VOD (spread evenly across rounds)
N_SAMPLES = 80
# Minimum health so we're not looking at dead player rows
MIN_HEALTH = 50

OUTPUT_ROOT = Path("vods/weapon_review")


def main():
    cropper = Cropper()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for vod in VODS:
        name = vod["name"]
        video_path = vod["video"]
        csv_path = vod["csv"]

        if not video_path.exists():
            print(f"[{name}] Video not found: {video_path}")
            continue
        if not csv_path.exists():
            print(f"[{name}] CSV not found: {csv_path}")
            continue

        out_dir = OUTPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path, dtype=str)
        active = df[df["phase"] == "ACTIVE_ROUND"].copy()

        # Collect (timestamp, player_idx, side) for unknown-weapon alive players
        candidates = []
        wpn_cols = [c for c in df.columns if c.endswith("_weapon")]
        for _, row in active.iterrows():
            for pidx in range(10):
                w = str(row.get(f"player_{pidx}_weapon", "")).lower().strip()
                alive = str(row.get(f"player_{pidx}_alive", "")).lower()
                hp_str = str(row.get(f"player_{pidx}_health", "0"))
                try:
                    hp = int(float(hp_str))
                except (ValueError, TypeError):
                    hp = 0
                if w == "unknown" and alive == "true" and hp >= MIN_HEALTH:
                    try:
                        ts = float(row["timestamp"])
                    except (ValueError, TypeError):
                        continue
                    side = "left" if pidx < 5 else "right"
                    candidates.append((ts, pidx, side))

        if not candidates:
            print(f"[{name}] No unknown-weapon frames found.")
            continue

        # Sample evenly
        step = max(1, len(candidates) // N_SAMPLES)
        samples = candidates[::step][:N_SAMPLES]
        print(f"[{name}] {len(candidates)} unknown-weapon frames -> sampling {len(samples)}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{name}] Could not open video.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        saved = 0

        for i, (ts, pidx, side) in enumerate(samples):
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            player_crops = cropper.crop_player_info(frame)
            if pidx >= len(player_crops):
                continue
            crop_data = player_crops[pidx]
            if crop_data.get("side") != side:
                continue
            weapon_crop = crop_data.get("weapon")
            if weapon_crop is None or weapon_crop.size == 0:
                continue

            # Filter out credits-display frames: credits text produces very bright
            # white pixels (>240) in isolated clusters.  Weapon icons spread their
            # brightness more evenly.  If >20% of pixels are near-white → skip.
            gray = cv2.cvtColor(weapon_crop, cv2.COLOR_BGR2GRAY) if len(weapon_crop.shape) == 3 else weapon_crop
            very_bright = np.sum(gray > 220)
            total_px = gray.size
            if very_bright / total_px > 0.20:
                # Looks like a credits/text display, not a weapon icon
                continue

            # Also skip near-blank crops (player offscreen / HUD hidden)
            if gray.mean() < 10:
                continue

            # Save at 4× scale so it's easy to see
            h, w = weapon_crop.shape[:2]
            big = cv2.resize(weapon_crop, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
            fname = out_dir / f"{name}_p{pidx}_{ts:.1f}.png"
            cv2.imwrite(str(fname), big)
            saved += 1

        cap.release()
        print(f"[{name}] Saved {saved} crops to {out_dir}/")

    print(f"\nDone. Review crops in {OUTPUT_ROOT}/")
    print("Rename any shorty/bucky/marshal/ares crops to src/valoscribe/templates/weapons/<name>N.png")


if __name__ == "__main__":
    main()
