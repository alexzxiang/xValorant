"""
Calibrate minimap homography matrices.

Interactive tool: for a given map, lets you click reference points on a
minimap image and input their known normalized map coordinates,
then computes the homography H and saves it to minimap_homographies.json.

Usage (recommended — use the clean 450x450 map PNG):
    python scripts/calibrate_minimap.py \
        --map haven \
        --map-image maps/haven.png

Usage (from a full 1080p broadcast frame):
    python scripts/calibrate_minimap.py \
        --map haven \
        --frame-path debug_frames/haven_frame_1000.png

Controls:
    - Left-click to mark a reference point
    - Enter its normalized map coordinate (x y) at the terminal prompt
    - Press 'c' to compute and save the homography (needs ≥4 points)
    - Press 'r' to reset all points
    - Press 'q' to quit without saving

Normalized coordinates [0,1]²:
    (0,0) = top-left of the minimap  (0,1) = bottom-left
    (1,0) = top-right                (1,1) = bottom-right
    Attacker spawn is at the south (bottom), so roughly y≈0.9

Output:
    Updates src/valoscribe/config/minimap_homographies.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HOMOGRAPHIES_PATH = Path("src/valoscribe/config/minimap_homographies.json")
MINIMAP_REGION = {"x": 10, "y": 20, "width": 450, "height": 450}


class MinimapCalibrator:
    def __init__(self, map_name: str, frame_path: Path | None = None, map_image_path: Path | None = None):
        self.map_name = map_name

        if map_image_path is not None:
            # Direct map PNG (450x450) — no cropping needed
            self.minimap = cv2.imread(str(map_image_path))
            if self.minimap is None:
                raise FileNotFoundError(f"Could not load map image: {map_image_path}")
            h, w = self.minimap.shape[:2]
            if h != 450 or w != 450:
                print(f"Warning: expected 450x450, got {w}x{h}. Resizing.")
                self.minimap = cv2.resize(self.minimap, (450, 450))
        elif frame_path is not None:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Could not load frame: {frame_path}")
            r = MINIMAP_REGION
            self.minimap = frame[r["y"]:r["y"]+r["height"], r["x"]:r["x"]+r["width"]].copy()
        else:
            raise ValueError("Provide either --map-image or --frame-path")

        self.display = self.minimap.copy()

        self.src_points: list[list[float]] = []  # minimap pixel coords
        self.dst_points: list[list[float]] = []  # normalized map coords [0,1]

    def run(self) -> np.ndarray | None:
        window = f"Minimap calibration: {self.map_name}"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_click)

        print("\nCalibration instructions:")
        print("  1. Left-click a reference point on the minimap")
        print("  2. Enter its normalized map coordinate (e.g. '0.23 0.45')")
        print("  3. Repeat for ≥4 points")
        print("  Press 'c' to compute, 'r' to reset, 'q' to quit\n")

        while True:
            cv2.imshow(window, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("Quit without saving.")
                cv2.destroyAllWindows()
                return None

            elif key == ord('r'):
                self.src_points.clear()
                self.dst_points.clear()
                self.display = self.minimap.copy()
                print("Reset all points.")

            elif key == ord('c'):
                if len(self.src_points) < 4:
                    print(f"Need ≥4 points (have {len(self.src_points)})")
                    continue
                H, status = cv2.findHomography(
                    np.array(self.src_points, dtype=np.float64),
                    np.array(self.dst_points, dtype=np.float64),
                    cv2.RANSAC,
                    5.0,
                )
                if H is None:
                    print("Homography computation failed — check your reference points.")
                    continue
                print(f"\nHomography computed from {status.sum()}/{len(self.src_points)} inliers:")
                print(H)
                cv2.destroyAllWindows()
                return H

    def _on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"\nClicked minimap pixel ({x}, {y})")
        coord_str = input("Enter normalized map coord (x y) [0.0-1.0]: ").strip()
        try:
            mx, my = map(float, coord_str.split())
            if not (0.0 <= mx <= 1.0 and 0.0 <= my <= 1.0):
                print("Coordinates must be in [0,1]. Ignoring.")
                return
        except ValueError:
            print("Invalid input. Expected two floats, e.g. '0.25 0.63'. Ignoring.")
            return

        self.src_points.append([float(x), float(y)])
        self.dst_points.append([mx, my])

        # Draw the point
        cv2.circle(self.display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            self.display, f"{len(self.src_points)}: ({mx:.2f},{my:.2f})",
            (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
        print(f"  Point {len(self.src_points)}: minimap ({x},{y}) → map ({mx:.3f},{my:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Calibrate minimap homography")
    parser.add_argument("--map", required=True, help="Map name (e.g. haven, ascent)")
    parser.add_argument("--map-image", type=Path, default=None,
                        help="Path to a 450x450 map PNG (e.g. maps/haven.png) — preferred over --frame-path")
    parser.add_argument("--frame-path", type=Path, default=None,
                        help="Path to a full 1080p broadcast frame (minimap will be cropped from it)")
    parser.add_argument("--output", type=Path, default=HOMOGRAPHIES_PATH)
    args = parser.parse_args()

    if args.map_image is None and args.frame_path is None:
        parser.error("Provide either --map-image maps/<map>.png or --frame-path <1080p_frame.png>")

    calibrator = MinimapCalibrator(args.map, frame_path=args.frame_path, map_image_path=args.map_image)
    H = calibrator.run()

    if H is None:
        sys.exit(1)

    # Load existing homographies
    if args.output.exists():
        with open(args.output) as f:
            data = json.load(f)
    else:
        data = {}

    # Strip comment keys and save
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
    clean_data[args.map] = H.tolist()

    # Preserve comment keys
    output_data = {}
    for k, v in data.items():
        if k.startswith("_"):
            output_data[k] = v
    output_data.update(clean_data)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved homography for '{args.map}' to {args.output}")

    # Quick sanity check: transform the src points and compare
    print("\nSanity check (src → transformed → expected):")
    src = np.array(calibrator.src_points, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    for i, (t, d) in enumerate(zip(transformed, calibrator.dst_points)):
        print(f"  Point {i+1}: ({t[0]:.3f}, {t[1]:.3f}) vs expected ({d[0]:.3f}, {d[1]:.3f})")


if __name__ == "__main__":
    main()
