# Credits Icon Template

This directory contains the credits icon template used for player alive/dead detection.

## Template File

- `credits_icon.png` - Credits icon template (preprocessed: white icon on black background)

## How It Works

The credits icon appears in the player info region when a player is alive. When a player dies, the credits information moves/disappears, causing the icon to not be found in the expected location.

This makes credits icon detection a reliable indicator of player alive/dead status:
- **Icon found** → Player is alive
- **Icon not found** → Player is dead

## Extracting Template

Use the `extract-credits-crops` command to save preprocessed credits crops from a VOD:

```bash
# Extract from a single player
valoscribe extract-credits-crops video.mp4 --player 0 --output ./my_credits

# Extract from all players
valoscribe extract-credits-crops video.mp4 --player -1 --interval 10
```

This will save:
- `*_credits_original.png` - Raw crops from the video
- `*_credits_preprocessed.png` - Preprocessed versions (use these for the template)

## Creating the Template

1. Review the preprocessed images in the output directory
2. Select a clear, well-defined credits icon
3. Crop just the icon (not the surrounding area)
4. Save as `credits_icon.png` in this directory
5. Test with `detect-credits-template` command

## Testing the Template

```bash
# Test on a single player
valoscribe detect-credits-template video.mp4 --player 0 --max-frames 10

# Test on all players
valoscribe detect-credits-template video.mp4 --player -1 --max-frames 5

# Adjust confidence threshold if needed
valoscribe detect-credits-template video.mp4 --player -1 --min-confidence 0.6
```

## Template Requirements

- Grayscale image
- White icon on black background (from Otsu's thresholding)
- Should match the preprocessed crops exactly
- Icon should be clearly visible and not too small
