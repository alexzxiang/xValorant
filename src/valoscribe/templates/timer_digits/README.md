# Timer Digit Templates

This directory contains template images for digits 0-9 used in template matching timer detection.

## Required Files

Place the following template images in this directory:
- `0.png` - Template for digit 0
- `1.png` - Template for digit 1
- `2.png` - Template for digit 2
- `3.png` - Template for digit 3
- `4.png` - Template for digit 4
- `5.png` - Template for digit 5
- `6.png` - Template for digit 6
- `7.png` - Template for digit 7
- `8.png` - Template for digit 8
- `9.png` - Template for digit 9

**Note:** Colon (:) and period (.) templates are NOT needed. The timer format is inferred from the number of detected digits:
- **3 digits** = m:ss format (e.g., "1:45" → detects "145" → 1 minute 45 seconds)
- **4 digits** = ss.ms format (e.g., "09.67" → detects "0967" → 9.67 seconds)

## Template Requirements

- **Format**: PNG (grayscale preferred, but color works)
- **Content**: Clean, cropped character from Valorant HUD timer display
- **Preprocessing**: Should match the preprocessing applied during detection:
  - Upscaled 3x with INTER_CUBIC
  - Otsu's automatic binary threshold (handles both white and red text)
  - White text on black background (no inversion)
- **Size**: Should be consistent with the preprocessed timer crops (typically ~50-100px wide after 3x upscale)

## Note on Red Timer Text

When the timer runs low, the text color changes to red. The preprocessing uses Otsu's automatic thresholding to handle both white text (normal) and red text (low timer), so you should extract templates from frames with white text for consistency.

## Extraction Tips

You can extract templates by:
1. Using `extract-timer-crops` to save preprocessed crops
2. Finding frames with clear, well-detected individual digits (0-9)
3. Manually cropping individual digits from the preprocessed images
4. Saving as PNG files named by the digit they represent (0.png - 9.png)

## Example Usage

```bash
# Extract timer crops every 10 seconds from a VOD
python -m valoscribe extract-timer-crops video.mp4 -o ./timer_crops -i 10

# Browse the preprocessed images to find good examples of each digit
# Manually crop and save individual digits as 0.png, 1.png, ..., 9.png
```
