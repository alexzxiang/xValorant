"""
Bootstrap weapon icon templates from processed VOD debug frames.

Reads frame_states.csv to find moments where each weapon is clearly held
(player alive, known weapon from killfeed), then crops the weapon region
and saves it as a template PNG.

Run this ONCE after processing your first few VODs with valoscribe.
The templates are saved to src/valoscribe/templates/weapons/<weapon>.png.

Usage:
    python scripts/extract_weapon_templates.py \
        --data-dir champs2025_processed_vods \
        --video-dir /path/to/vod/files \
        --output-dir src/valoscribe/templates/weapons

Requirements:
    - Processed frame_states.csv with at least 1 map of data
    - The original video file accessible (to extract actual frames)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pandas not installed. Run: pip install pandas", file=sys.stderr)
    sys.exit(1)


# All Valorant weapon names — templates will be extracted for each
ALL_WEAPONS = [
    "classic", "shorty", "frenzy", "ghost", "sheriff",  # Sidearms
    "stinger", "spectre",                                # SMGs
    "bucky", "judge",                                    # Shotguns
    "bulldog", "guardian", "phantom", "vandal",         # Rifles
    "marshal", "outlaw", "operator",                     # Snipers
    "ares", "odin",                                      # Heavy
]


def extract_templates_from_video(
    video_path: Path,
    frame_csv: Path,
    output_dir: Path,
    target_weapons: set[str],
    min_health: int = 50,
    n_samples_per_weapon: int = 5,
) -> dict[str, int]:
    """
    Extract weapon icon crops from a video at timestamps where each weapon
    appears in the frame_states.csv (player alive + weapon known).

    Returns dict of {weapon_name: n_saved}.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from valoscribe.detectors.cropper import Cropper

    df = pd.read_csv(frame_csv, dtype=str)
    df_active = df[df["phase"] == "ACTIVE_ROUND"].copy()
    if df_active.empty:
        return {}

    cropper = Cropper()
    saved = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)

    for weapon in target_weapons:
        if weapon not in ALL_WEAPONS:
            continue
        if weapon in saved and saved[weapon] >= n_samples_per_weapon:
            continue

        # Find rows where this weapon appears
        samples = []
        for _, row in df_active.iterrows():
            for pidx in range(10):
                w = str(row.get(f"player_{pidx}_weapon", "")).lower().strip()
                alive = str(row.get(f"player_{pidx}_alive", "")).lower()
                hp_str = str(row.get(f"player_{pidx}_health", "0"))
                try:
                    hp = int(float(hp_str))
                except (ValueError, TypeError):
                    hp = 0

                if w == weapon and alive == "true" and hp >= min_health:
                    try:
                        ts = float(row["timestamp"])
                    except (ValueError, TypeError):
                        continue
                    side = "left" if pidx < 5 else "right"
                    samples.append((ts, pidx, side))

                if len(samples) >= n_samples_per_weapon:
                    break
            if len(samples) >= n_samples_per_weapon:
                break

        if not samples:
            continue

        weapon_crops = []
        for ts, pidx, side in samples[:n_samples_per_weapon]:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            player_crops = cropper.crop_player_info(frame)
            if pidx >= len(player_crops):
                continue
            crop_data = player_crops[pidx]
            weapon_crop = crop_data.get("weapon")
            if weapon_crop is None or weapon_crop.size == 0:
                continue

            weapon_crops.append(weapon_crop)

        if not weapon_crops:
            continue

        # Use the median crop as the template (most representative)
        # Convert all to grayscale, upscale 2×, then pick the most centered one
        processed = []
        for wc in weapon_crops:
            gray = cv2.cvtColor(wc, cv2.COLOR_BGR2GRAY) if len(wc.shape) == 3 else wc.copy()
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            processed.append(gray)

        # Save the first good crop as the template
        template = processed[0]
        out_path = output_dir / f"{weapon}.png"
        cv2.imwrite(str(out_path), template)
        saved[weapon] = 1
        print(f"  Saved template: {weapon}.png ({template.shape[1]}×{template.shape[0]})")

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract weapon icon templates")
    parser.add_argument("--data-dir", type=Path, default=Path("champs2025_processed_vods"))
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory containing .mp4 files")
    parser.add_argument("--output-dir", type=Path, default=Path("src/valoscribe/templates/weapons"))
    parser.add_argument("--weapons", nargs="*", default=None, help="Specific weapons to extract")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_weapons = set(args.weapons) if args.weapons else set(ALL_WEAPONS)

    # Find already-existing templates
    existing = {p.stem.lower() for p in args.output_dir.glob("*.png")}
    target_weapons -= existing
    if not target_weapons:
        print("All weapon templates already exist.")
        return

    print(f"Extracting templates for: {sorted(target_weapons)}")

    # Find video files and their corresponding CSV
    video_dir = args.video_dir or args.data_dir

    total_saved = {}
    for series_dir in sorted(args.data_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        for map_dir in sorted(series_dir.iterdir()):
            if not map_dir.is_dir() or map_dir.name == "metadata":
                continue
            frame_csv = map_dir / "output" / "frame_states.csv"
            if not frame_csv.exists():
                continue

            # Find matching video file
            video_path = None
            for ext in [".mp4", ".mkv", ".ts"]:
                candidate = video_dir / f"{map_dir.name}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
                # Try series/map pattern
                candidate2 = video_dir / series_dir.name / f"{map_dir.name}{ext}"
                if candidate2.exists():
                    video_path = candidate2
                    break

            if video_path is None:
                # Check if video is directly in map_dir
                for ext in [".mp4", ".mkv", ".ts"]:
                    candidate = map_dir / f"video{ext}"
                    if candidate.exists():
                        video_path = candidate
                        break

            if video_path is None:
                continue

            remaining = target_weapons - set(total_saved.keys())
            if not remaining:
                break

            print(f"\nProcessing {series_dir.name}/{map_dir.name}...")
            saved = extract_templates_from_video(
                video_path, frame_csv, args.output_dir, remaining
            )
            total_saved.update(saved)

        if not (target_weapons - set(total_saved.keys())):
            break

    print(f"\nExtracted {len(total_saved)} templates: {sorted(total_saved.keys())}")
    missing = target_weapons - set(total_saved.keys())
    if missing:
        print(f"Could not extract: {sorted(missing)}")
        print("  (These weapons may not appear in the processed VODs)")


if __name__ == "__main__":
    main()
