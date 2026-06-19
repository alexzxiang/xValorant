"""
Identify positions of static A/B/C site labels on Haven's minimap
so we can hard-code them as exclusion zones.

Saves a labeled image so we can verify visually.
"""
import cv2
import glob
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.detectors.cropper import Cropper

vod = glob.glob("vods/*.mp4")[0]
cap = cv2.VideoCapture(vod)
fps = cap.get(cv2.CAP_PROP_FPS)
cropper = Cropper()

# Sample many frames across the whole game and find blobs that are ALWAYS present
# Attack range: H=[0-20] | H=[155-180], S=[70-255], V=[70-255]
# Defense range: H=[70-95], S=[70-200], V=[150-255]

ATK_LO  = np.array([0,  70, 70], dtype=np.uint8)
ATK_HI  = np.array([20, 255, 255], dtype=np.uint8)
ATK_LO2 = np.array([155, 70, 70], dtype=np.uint8)
ATK_HI2 = np.array([180, 255, 255], dtype=np.uint8)
DEF_LO  = np.array([70, 70, 150], dtype=np.uint8)
DEF_HI  = np.array([95, 200, 255], dtype=np.uint8)
QUANT = 8  # quantize positions to 8px grid

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def find_blobs(hsv, lo, hi, lo2=None, hi2=None):
    mask = cv2.inRange(hsv, lo, hi)
    if lo2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < 20 or a > 2000:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        p = cv2.arcLength(cnt, True)
        circ = 4 * 3.14159 * a / (p * p) if p > 0 else 0
        blobs.append((int(round(cx/QUANT)*QUANT), int(round(cy/QUANT)*QUANT), a, circ))
    return blobs

# Sample 40 timestamps evenly across the match (t=82s to t=2600s)
from collections import Counter
atk_hits = Counter()
def_hits = Counter()
n_samples = 0

STEP = 60  # every 60 seconds
for t in range(82, 2600, STEP):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if not ret:
        continue
    minimap = cropper.crop_simple_region(frame, "minimap")
    if minimap.size == 0:
        continue
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    n_samples += 1

    for cx, cy, a, circ in find_blobs(hsv, ATK_LO, ATK_HI, ATK_LO2, ATK_HI2):
        atk_hits[(cx, cy)] += 1
    for cx, cy, a, circ in find_blobs(hsv, DEF_LO, DEF_HI):
        def_hits[(cx, cy)] += 1

cap.release()

THRESHOLD = 0.50  # appear in ≥50% of samples = site label, not a player

print(f"Sampled {n_samples} frames (every {STEP}s)")
print(f"\nPersistent ATTACK blobs (>= {THRESHOLD*100:.0f}% of frames):")
for (cx, cy), count in sorted(atk_hits.items(), key=lambda x: -x[1]):
    if count / n_samples >= THRESHOLD:
        print(f"  ({cx}, {cy})  seen in {count}/{n_samples} = {count/n_samples*100:.0f}%")

print(f"\nPersistent DEFENSE blobs (>= {THRESHOLD*100:.0f}% of frames):")
for (cx, cy), count in sorted(def_hits.items(), key=lambda x: -x[1]):
    if count / n_samples >= THRESHOLD:
        print(f"  ({cx}, {cy})  seen in {count}/{n_samples} = {count/n_samples*100:.0f}%")

# Visualize on the minimap
cap2 = cv2.VideoCapture(glob.glob("vods/*.mp4")[0])
fps2 = cap2.get(cv2.CAP_PROP_FPS)
cap2.set(cv2.CAP_PROP_POS_FRAMES, int(100 * fps2))
ret, frame = cap2.read()
cap2.release()
minimap = cropper.crop_simple_region(frame, "minimap")
vis = minimap.copy()

THRESHOLD_SHOW = 0.40
for (cx, cy), count in atk_hits.items():
    if count / n_samples >= THRESHOLD_SHOW:
        cv2.circle(vis, (cx, cy), 16, (0, 0, 255), 2)
        cv2.putText(vis, f"A{count}", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
for (cx, cy), count in def_hits.items():
    if count / n_samples >= THRESHOLD_SHOW:
        cv2.circle(vis, (cx, cy), 16, (0, 220, 80), 2)
        cv2.putText(vis, f"D{count}", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,220,80), 1)

cv2.imwrite("debug_exclusion_zones.png", vis)
print("\nSaved debug_exclusion_zones.png")
