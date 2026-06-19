"""
Bootstrap weapon icon templates by sampling raw crops from the video.

The existing extract_weapon_templates.py is circular: it needs weapon
detection to label frames, but detection needs templates. This script
breaks that loop by sampling frames blindly and clustering by visual
similarity so a human can label each cluster.

Workflow:
  1. Run this script against the VOD:
       python scripts/bootstrap_weapon_templates.py \
           --video "vods/DRX vs. NRG..." \
           --start 67 --end 2644 \
           --output-dir weapon_crops

  2. Inspect weapon_crops/groups/:
       Each group_NNN.png is a representative of a visually distinct icon.
       Rename + copy the best representative for each weapon to:
           src/valoscribe/templates/weapons/<weapon_name>.png

  3. Verify with the gallery:
       Open weapon_crops/gallery.png for a quick visual overview.

Usage:
    python scripts/bootstrap_weapon_templates.py \
        --video "vods/DRX vs. NRG — VALORANT Champions Paris — Group Stage — Map 01.f399.mp4" \
        --start 67 --end 2644 \
        --output-dir weapon_crops \
        --interval 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.detectors.cropper import Cropper


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(crop: np.ndarray) -> np.ndarray:
    """
    Grayscale + CLAHE normalisation + 2× upscale.

    CLAHE equalises local contrast, which removes the tint difference between
    attack (orange) and defense (cyan) weapon icons when converted to grayscale.
    Without this, the same weapon on opposite sides clusters separately.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    h, w = gray.shape
    return cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def is_empty(crop: np.ndarray, min_std: float = 8.0) -> bool:
    """Return True if crop is mostly uniform (black/dead player region)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    return float(np.std(gray)) < min_std


def crop_quality(crop: np.ndarray) -> float:
    """Higher = more information content (pick as group representative)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    return float(np.std(gray)) + float(np.mean(gray)) / 10.0


# ── Clustering ────────────────────────────────────────────────────────────────

class CropGroup:
    def __init__(self, representative: np.ndarray, preprocessed: np.ndarray):
        self.representative = representative.copy()
        self.preprocessed = preprocessed.copy()
        self.count = 1
        self.best_quality = crop_quality(representative)

    def matches(self, other_pre: np.ndarray, threshold: float) -> bool:
        tmpl = self.preprocessed
        img = other_pre
        if tmpl.shape != img.shape:
            return False
        result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if float(max_val) >= threshold:
            return True
        # Also try horizontally flipped — right-side icons are mirrored
        flipped = cv2.flip(img, 1)
        result2 = cv2.matchTemplate(flipped, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, _ = cv2.minMaxLoc(result2)
        return float(max_val2) >= threshold

    def update_representative(self, crop: np.ndarray, pre: np.ndarray) -> None:
        q = crop_quality(crop)
        if q > self.best_quality:
            self.best_quality = q
            self.representative = crop.copy()
            self.preprocessed = pre.copy()
        self.count += 1


def cluster_crops(
    crops_and_originals: list[tuple[np.ndarray, np.ndarray]],
    match_threshold: float = 0.72,
) -> list[CropGroup]:
    """
    Iteratively cluster (preprocessed, original) pairs.
    Two crops go in the same group if template-match score >= threshold.
    """
    groups: list[CropGroup] = []

    for original, preprocessed in crops_and_originals:
        matched = False
        for group in groups:
            if group.matches(preprocessed, match_threshold):
                group.update_representative(original, preprocessed)
                matched = True
                break
        if not matched:
            groups.append(CropGroup(original, preprocessed))

    # Sort by frequency (most common weapons first)
    groups.sort(key=lambda g: g.count, reverse=True)
    return groups


# ── Gallery ───────────────────────────────────────────────────────────────────

def make_gallery(
    groups: list[CropGroup],
    cols: int = 10,
    target_w: int = 90,
    target_h: int = 34,
) -> np.ndarray:
    """Compose a grid image of all group representatives with count labels."""
    rows = (len(groups) + cols - 1) // cols
    pad = 4
    label_h = 14
    cell_h = target_h + label_h + pad * 2
    cell_w = target_w + pad * 2

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    canvas[:] = 40  # dark grey background

    for i, group in enumerate(groups):
        row, col = divmod(i, cols)
        y0 = row * cell_h + pad
        x0 = col * cell_w + pad

        rep = group.representative
        rep_resized = cv2.resize(rep, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + target_h, x0:x0 + target_w] = rep_resized

        label = f"{i:03d} n={group.count}"
        cv2.putText(
            canvas, label,
            (x0, y0 + target_h + label_h - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
        )

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap weapon templates from video")
    parser.add_argument("--video", type=Path, required=True, help="Path to .mp4 VOD file")
    parser.add_argument("--start",    type=float, default=0.0,   help="Start time (seconds)")
    parser.add_argument("--end",      type=float, default=None,  help="End time (seconds)")
    parser.add_argument("--interval", type=float, default=15.0,  help="Sample every N seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("weapon_crops"))
    parser.add_argument("--threshold", type=float, default=0.72, help="Template match threshold for clustering")
    parser.add_argument("--min-std",   type=float, default=8.0,  help="Min std to treat crop as non-empty")
    parser.add_argument("--template-dir", type=Path,
                        default=Path("src/valoscribe/templates/weapons"),
                        help="Destination for final templates")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    end = min(args.end, duration) if args.end else duration

    print(f"Video: {fps:.2f} fps, {duration:.1f}s total")
    print(f"Sampling {args.start:.0f}s–{end:.0f}s every {args.interval:.0f}s")

    cropper = Cropper()

    all_crops: list[tuple[np.ndarray, np.ndarray]] = []
    timestamps = np.arange(args.start, end, args.interval)
    n_empty = 0

    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        player_crops = cropper.crop_player_info(frame)
        for pidx, crop_data in enumerate(player_crops):
            weapon_crop = crop_data.get("weapon")
            if weapon_crop is None or weapon_crop.size == 0:
                continue
            if is_empty(weapon_crop, min_std=args.min_std):
                n_empty += 1
                continue
            pre = preprocess(weapon_crop)
            all_crops.append((weapon_crop, pre))

    cap.release()
    print(f"Extracted {len(all_crops)} non-empty weapon crops ({n_empty} empty skipped)")

    if not all_crops:
        print("No weapon crops found — check --start/--end range or video path.")
        sys.exit(1)

    print(f"Clustering with threshold={args.threshold}...")
    groups = cluster_crops(all_crops, match_threshold=args.threshold)
    print(f"Found {len(groups)} distinct visual groups")

    # Save representative for each group
    for i, group in enumerate(groups):
        out_path = groups_dir / f"group_{i:03d}_n{group.count}.png"
        cv2.imwrite(str(out_path), group.representative)

    # Gallery
    gallery = make_gallery(groups)
    gallery_path = args.output_dir / "gallery.png"
    cv2.imwrite(str(gallery_path), gallery)

    print(f"\nSaved {len(groups)} group representatives to {groups_dir}/")
    print(f"Gallery: {gallery_path}")
    print()
    print("Top groups (most frequent):")
    for i, g in enumerate(groups[:20]):
        print(f"  group_{i:03d}_n{g.count}.png  ({g.count} crops)")

    print()
    print("Next steps:")
    print("  1. Open weapon_crops/gallery.png — each cell is a unique icon")
    print("  2. Identify each group by visually comparing to in-game weapon icons")
    print(f"  3. Copy + rename to {args.template_dir}/<weapon_name>.png")
    print("     e.g.:  cp weapon_crops/groups/group_000_n87.png src/valoscribe/templates/weapons/phantom.png")
    print()
    print("Weapon names (from TemplateWeaponDetector):")
    print("  Sidearms: classic, shorty, frenzy, ghost, sheriff")
    print("  SMGs:     stinger, spectre")
    print("  Shotguns: bucky, judge")
    print("  Rifles:   bulldog, guardian, phantom, vandal")
    print("  Snipers:  marshal, outlaw, operator")
    print("  Heavy:    ares, odin")
    print("  Melee:    melee")


if __name__ == "__main__":
    main()
