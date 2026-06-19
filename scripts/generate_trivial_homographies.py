"""
Generate trivial homography matrices for all known Valorant maps.

The broadcast minimap is cropped at x=10, y=20, w=450, h=450.
The maps/ PNG files are also 450x450 in the same orientation.
So the transform from minimap pixel (px, py) to normalized [0,1]^2 is:
    nx = px / 450,  ny = py / 450
which corresponds to H = diag(1/450, 1/450, 1).

This trivial homography is correct when:
  - The broadcast minimap and the map PNG share the same coordinate space
  - There is no rotation or perspective distortion between them

If per-map refinement is needed later, run:
    python scripts/calibrate_minimap.py --map <name> --map-image maps/<name>.png
and the interactive tool will overwrite the entry for that map.
"""

import json
from pathlib import Path

HOMOGRAPHIES_PATH = Path("src/valoscribe/config/minimap_homographies.json")

# 1/450 scale: maps minimap pixel [0,449] to normalized [0, 0.9978] ≈ [0,1]
S = 1.0 / 450.0
TRIVIAL_H = [
    [S,   0.0, 0.0],
    [0.0, S,   0.0],
    [0.0, 0.0, 1.0],
]

# All maps that appear in Champions 2025 data + maps/ folder
MAPS = [
    "ascent",
    "bind",
    "breeze",
    "corrode",   # also called "abyss" in older VODs (check metadata)
    "abyss",
    "fracture",
    "haven",
    "icebox",
    "lotus",
    "pearl",
    "split",
    "sunset",
]


def main():
    # Load existing file to preserve comment keys
    if HOMOGRAPHIES_PATH.exists():
        with open(HOMOGRAPHIES_PATH) as f:
            data = json.load(f)
    else:
        data = {}

    comment_keys = {k: v for k, v in data.items() if k.startswith("_")}
    existing = {k: v for k, v in data.items() if not k.startswith("_")}

    added, skipped = [], []
    for map_name in MAPS:
        if map_name in existing:
            skipped.append(map_name)
        else:
            existing[map_name] = TRIVIAL_H
            added.append(map_name)

    output = {}
    output.update(comment_keys)
    output.update(existing)

    with open(HOMOGRAPHIES_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Written to {HOMOGRAPHIES_PATH}")
    print(f"Added ({len(added)}):   {', '.join(added) or 'none'}")
    print(f"Kept   ({len(skipped)}): {', '.join(skipped) or 'none'}")
    print()
    print("Homography (trivial 1/450 scale):")
    for row in TRIVIAL_H:
        print(" ", row)
    print()
    print("To refine a specific map interactively:")
    print("  python scripts/calibrate_minimap.py --map haven --map-image maps/haven.png")


if __name__ == "__main__":
    main()
