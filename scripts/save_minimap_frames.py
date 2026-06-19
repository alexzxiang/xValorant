"""Save minimap crops at various match timestamps for overlay calibration."""
import cv2, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.detectors.cropper import Cropper

cap = cv2.VideoCapture(glob.glob("vods/*.mp4")[0])
fps = cap.get(cv2.CAP_PROP_FPS)
cropper = Cropper()

out_dir = Path("vods/minimap_frames")
out_dir.mkdir(exist_ok=True)

# Spread across the match: pregame, early, mid, late rounds, overtime
timestamps = [70, 85, 100, 130, 160, 200, 250, 300, 400, 500, 600, 700, 800]

for t in timestamps:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if not ret:
        continue
    mm = cropper.crop_simple_region(frame, "minimap")
    if mm.size == 0:
        continue
    path = out_dir / f"minimap_t{t:04d}.png"
    cv2.imwrite(str(path), mm)
    print(f"Saved {path}  ({mm.shape[1]}x{mm.shape[0]})")

cap.release()
print(f"\nAll saved to {out_dir}/")
print(f"Create a mask PNG the same size ({mm.shape[1]}x{mm.shape[0]}) — white where the map is, black for background.")
print("Save it as: src/valoscribe/config/minimap_mask_haven.png")
