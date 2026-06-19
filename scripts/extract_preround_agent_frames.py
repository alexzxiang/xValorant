"""
Extract buy-phase (preround) scoreboard frames and per-slot crops so you can
add missing agent portrait templates.

For each specified timestamp range the script saves:
  - One full scoreboard screenshot (1920×1080) named <map>_<ts>s.png
  - 10 individual agent-slot crops (the portrait region each detector reads)
    named <map>_slot<N>_<ts>s.png

After running, identify the slots that show the missing agent and copy/rename
those crops to:
  src/valoscribe/templates/preround_agents/attack/<agent>_atk.jpg
  src/valoscribe/templates/preround_agents/defense/<agent>_def.jpg

Usage:
    python scripts/extract_preround_agent_frames.py \\
        --video  "vods/NRG vs. LEV - FULL MATCH ...mp4" \\
        --timestamps 115 120 125 130 135 140 145 \\
        --output vods/preround_frames/ascent
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.detectors.cropper import Cropper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--timestamps", nargs="+", type=float, required=True,
        help="Video timestamps (seconds) to extract, e.g. 115 120 125 130 135 140",
    )
    parser.add_argument("--output", type=Path, default=Path("vods/preround_frames"))
    parser.add_argument(
        "--label", default="frame",
        help="Short label prefix for filenames (e.g. 'ascent')",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    cropper = Cropper()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Could not open video", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps:.2f}")

    for ts in args.timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Could not read frame at {ts}s")
            continue

        # Save full frame
        full_path = args.output / f"{args.label}_{ts:.0f}s.png"
        cv2.imwrite(str(full_path), frame)
        print(f"  [{ts:.0f}s] Full frame -> {full_path.name}")

        # Save per-slot agent_icon crops (4× upscaled) using the preround cropper
        preround_crops = cropper.crop_player_info_preround(frame)
        n_saved = 0
        for slot in range(10):
            if slot >= len(preround_crops):
                continue
            agent_icon = preround_crops[slot].get("agent_icon")
            if agent_icon is None or agent_icon.size == 0:
                continue
            h, w = agent_icon.shape[:2]
            big = cv2.resize(agent_icon, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
            side = "L" if slot < 5 else "R"
            slot_path = args.output / f"{args.label}_slot{slot}{side}_{ts:.0f}s.png"
            cv2.imwrite(str(slot_path), big)
            n_saved += 1

        print(f"         {n_saved}/10 slot crops saved.")

    cap.release()
    print(f"\nDone. Review frames in {args.output}/")
    print("Identify missing agent icons and save at 1x scale as:")
    print("  src/valoscribe/templates/preround_agents/attack/<agent>_atk.jpg")
    print("  src/valoscribe/templates/preround_agents/defense/<agent>_def.jpg")


if __name__ == "__main__":
    main()
