# Spike Template

This directory contains the template image for the spike icon used in template matching spike detection.

## Required File

Place the following template image in this directory:
- `spike.png` - Template of the spike icon that appears in the timer region when spike is planted

## Template Requirements

- **Format**: PNG (grayscale preferred, but color works)
- **Content**: Clean, cropped spike icon from Valorant HUD timer region
- **Preprocessing**: Should match the preprocessing applied during detection:
  - Upscaled 3x with INTER_CUBIC
  - Otsu's automatic binary threshold
  - White icon on black background (no inversion)
- **Size**: Should be consistent with the preprocessed timer crops (typically after 3x upscale)

## Purpose

The spike icon appears in the timer region when the spike is planted. Detecting this icon helps identify:
- Readable game frames during post-plant phase
- Distinction between timer (pre-plant) and spike icon (post-plant)
- Valid game state even when timer is not visible

## Extraction Tips

You can extract the template by:
1. Finding a frame where the spike is planted (spike icon visible in timer region)
2. Using `extract-timer-crops` or manually cropping the timer region
3. Ensuring the spike icon is clearly visible and centered
4. Preprocessing the crop to match detection preprocessing (grayscale, 3x upscale, Otsu threshold)
5. Saving as `spike.png`

## Example Usage

The spike detector is used in combination with timer detector to identify readable game frames:

```python
# Check if frame shows active gameplay
timer_info = timer_detector.detect(frame)
if timer_info:
    # Readable game frame - pre-plant phase
    is_game_frame = True
    time_remaining = timer_info.time_seconds
else:
    spike_info = spike_detector.detect(frame)
    if spike_info and spike_info.spike_planted:
        # Readable game frame - post-plant phase
        is_game_frame = True
    else:
        # Not a readable game frame (buy phase, between rounds, etc.)
        is_game_frame = False
```
