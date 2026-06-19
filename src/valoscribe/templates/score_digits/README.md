# Score Digit Templates

This directory contains template images for digits 0-9 used in template matching score detection.

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

## Template Requirements

- **Format**: PNG (grayscale preferred, but color works)
- **Content**: Clean, cropped digit from Valorant HUD score display
- **Preprocessing**: Should match the preprocessing applied during detection:
  - Upscaled 3x with INTER_CUBIC
  - Binary threshold at 127
  - Inverted if needed (dark text on light background)
- **Size**: Should be consistent with the preprocessed score crops (typically ~50-100px wide after 3x upscale)

## Extraction Tips

You can extract templates by:
1. Using `detect-score --show-debug` to save preprocessed crops
2. Finding frames with clear, well-detected single digits (0-9)
3. Manually cropping individual digits from the preprocessed images
4. Saving as PNG files named by the digit they represent
