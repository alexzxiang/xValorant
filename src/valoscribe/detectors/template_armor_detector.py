"""
Template-based armor detector for Valorant HUD.

Uses template matching to detect player armor values from the scoreboard.
Allows small overlap (1-2px) between digit matches due to small text size.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import ArmorInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class TemplateArmorDetector:
    """
    Template matching-based detector for extracting player armor.

    Uses pre-made templates of digits 0-9 to match against armor regions
    instead of OCR, providing more robust detection. Allows small overlap
    between digit matches (1-2px) since armor text is very small.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = cv2.TM_CCOEFF_NORMED,
        overlap_tolerance_px: int = 2,
    ):
        """
        Initialize template armor detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_dir: Directory containing digit templates (0.png - 9.png)
                         If None, uses default: src/valoscribe/templates/armor/
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
            overlap_tolerance_px: Allow up to N pixels of overlap between matches (default: 2)
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method
        self.overlap_tolerance_px = overlap_tolerance_px

        # Set default template directory if not provided
        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "armor"

        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

        log.info(
            f"Template armor detector initialized (min_confidence: {min_confidence}, "
            f"templates: {len(self.templates)}, method: {match_method}, "
            f"overlap_tolerance: {overlap_tolerance_px}px)"
        )

    def _load_templates(self) -> dict[str, np.ndarray]:
        """
        Load digit templates from template directory.

        Expects files named: 0.png, 1.png, 2.png, ..., 9.png

        Returns:
            Dictionary mapping digit strings to template images
        """
        templates = {}

        if not self.template_dir.exists():
            log.warning(f"Template directory does not exist: {self.template_dir}")
            return templates

        for digit in range(10):
            template_path = self.template_dir / f"{digit}.png"

            if not template_path.exists():
                log.warning(f"Template not found: {template_path}")
                continue

            # Load template in grayscale
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

            if template is None:
                log.warning(f"Failed to load template: {template_path}")
                continue

            templates[str(digit)] = template
            log.debug(f"Loaded template {digit}: {template.shape}")

        if len(templates) == 0:
            log.error(f"No templates loaded from {self.template_dir}")
        else:
            log.info(f"Loaded {len(templates)} digit templates")

        return templates

    def detect(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> Optional[ArmorInfo]:
        """
        Detect armor for a specific player using template matching.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard ("left" or "right")

        Returns:
            ArmorInfo if successfully detected, None otherwise
        """
        # Check if templates are loaded
        if len(self.templates) == 0:
            log.error("No templates loaded, cannot detect armor")
            return None

        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            log.warning(f"Player index {player_index} out of range (max: {len(player_crops) - 1})")
            return None

        player_crop_data = player_crops[player_index]

        # Check side
        if player_crop_data["side"] != side:
            log.warning(
                f"Player {player_index} is on {player_crop_data['side']} side, "
                f"but {side} was requested"
            )
            return None

        # Get armor crop
        if "armor" not in player_crop_data:
            log.warning(f"Armor region not found in player crop data")
            return None

        armor_crop = player_crop_data["armor"]

        if armor_crop.size == 0:
            log.warning(f"Armor crop is empty for player {player_index}")
            return None

        # Match templates
        result = self._match_armor_region(armor_crop)

        if result is None:
            log.debug(f"Template matching failed for player {player_index}")
            return None

        armor, confidence, raw_text = result

        # Validate armor range (0-50 for light armor, 0-25 for heavy armor, max 50)
        if not (0 <= armor <= 50):
            log.warning(f"Armor out of range for player {player_index}: {armor}")
            return None

        log.debug(f"Player {player_index} armor: {armor} (confidence: {confidence:.2f})")

        return ArmorInfo(
            armor=armor,
            confidence=confidence,
            raw_text=raw_text,
        )

    def _match_armor_region(
        self, crop: np.ndarray
    ) -> Optional[tuple[int, float, str]]:
        """
        Match templates against an armor region to extract the armor value.

        Handles 1-2 digit armor values (0-50).

        Args:
            crop: Cropped armor region

        Returns:
            Tuple of (armor, confidence, raw_text) or None if matching fails
        """
        # Preprocess crop
        preprocessed = self._preprocess_crop(crop)

        # Find all digit matches in the region
        matches = self._find_all_digit_matches(preprocessed)

        if len(matches) == 0:
            return None

        # Filter matches with small overlap tolerance (1-2px allowed)
        filtered_matches = self._filter_overlapping_matches(matches)

        if len(filtered_matches) == 0:
            return None

        # Sort by x-position (left to right) to get correct digit order
        filtered_matches.sort(key=lambda m: m["x"])

        # Limit to 2 digits max (armor can't be more than 2 digits: 0-50)
        if len(filtered_matches) > 2:
            log.warning(f"Too many digits detected ({len(filtered_matches)}), keeping best 2")
            # Keep the 2 highest confidence matches
            filtered_matches = sorted(filtered_matches, key=lambda m: m["confidence"], reverse=True)[:2]
            # Re-sort by position after keeping top 2
            filtered_matches.sort(key=lambda m: m["x"])

        # Construct armor value from digits
        digits = [m["digit"] for m in filtered_matches]
        raw_text = "".join(digits)

        try:
            armor = int(raw_text)
        except ValueError:
            log.warning(f"Failed to parse armor from matched digits: {raw_text}")
            return None

        # Fix OCR errors: treat 58 as 50 (common misread due to small text)
        if armor == 58:
            log.debug(f"Correcting armor value 58 -> 50 (OCR error)")
            armor = 50
            raw_text = "50"

        # Use minimum confidence across all matched digits
        confidence = min(m["confidence"] for m in filtered_matches)

        # Check confidence threshold
        if confidence < self.min_confidence:
            log.debug(
                f"Template match confidence {confidence:.2f} below threshold {self.min_confidence}"
            )
            return None

        return armor, confidence, raw_text

    def _find_all_digit_matches(self, image: np.ndarray) -> list[dict]:
        """
        Find all digit template matches in an image.

        Args:
            image: Preprocessed grayscale image

        Returns:
            List of match dictionaries with keys: digit, confidence, x, y, w, h
        """
        matches = []

        for digit, template in self.templates.items():
            # Run template matching
            result = cv2.matchTemplate(image, template, self.match_method)

            # Find all matches above threshold
            threshold = self.min_confidence
            locations = np.where(result >= threshold)

            # Get template dimensions
            h, w = template.shape

            # Store each match
            for y, x in zip(*locations):
                confidence = result[y, x]
                matches.append(
                    {
                        "digit": digit,
                        "confidence": float(confidence),
                        "x": int(x),
                        "y": int(y),
                        "w": w,
                        "h": h,
                    }
                )

        return matches

    def _filter_overlapping_matches(
        self, matches: list[dict]
    ) -> list[dict]:
        """
        Filter out overlapping matches, keeping only the best match per region.

        Allows small overlap (overlap_tolerance_px) to accommodate small armor text.

        Args:
            matches: List of match dictionaries

        Returns:
            Filtered list of matches
        """
        if len(matches) == 0:
            return []

        # Sort by confidence (descending)
        sorted_matches = sorted(matches, key=lambda m: m["confidence"], reverse=True)

        filtered = []

        for match in sorted_matches:
            # Check if this match overlaps with any already accepted match
            overlaps = False

            for accepted in filtered:
                # Calculate horizontal overlap
                x1_start, x1_end = match["x"], match["x"] + match["w"]
                x2_start, x2_end = accepted["x"], accepted["x"] + accepted["w"]

                overlap_start = max(x1_start, x2_start)
                overlap_end = min(x1_end, x2_end)
                overlap_width = max(0, overlap_end - overlap_start)

                # Allow small overlap (overlap_tolerance_px)
                if overlap_width > self.overlap_tolerance_px:
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(match)

        return filtered

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess armor crop for template matching.

        Produces white text on black background to match templates.

        Args:
            crop: Cropped armor region

        Returns:
            Preprocessed grayscale binary image (white text on black)
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Upscale for better matching (2x to match template extraction)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # Use Otsu's thresholding - handles varying brightness
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def detect_with_debug(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> tuple[Optional[ArmorInfo], np.ndarray, dict]:
        """
        Detect armor and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard

        Returns:
            Tuple of (ArmorInfo or None, armor_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            return None, np.array([]), {}

        player_crop_data = player_crops[player_index]

        if "armor" not in player_crop_data:
            return None, np.array([]), {}

        armor_crop = player_crop_data["armor"]

        if armor_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess using the same method as detection
        armor_preprocessed = self._preprocess_crop(armor_crop)

        # Find all matches for debug visualization
        matches = self._find_all_digit_matches(armor_preprocessed)

        debug_info = {
            "all_matches": matches,
            "filtered_matches": self._filter_overlapping_matches(matches),
        }

        # Run detection
        armor_info = self.detect(frame, player_index, side)

        return armor_info, armor_preprocessed, debug_info
