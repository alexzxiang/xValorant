"""
Template-based health detector for Valorant HUD.

Uses template matching to detect player health values from the scoreboard.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import HealthInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class TemplateHealthDetector:
    """
    Template matching-based detector for extracting player health.

    Uses pre-made templates of digits 0-9 to match against health regions
    instead of OCR, providing more robust detection.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        """
        Initialize template health detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_dir: Directory containing digit templates (0.png - 9.png)
                         If None, uses default: src/valoscribe/templates/health_digits/
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        # Set default template directory if not provided
        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "health_digits"

        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

        log.info(
            f"Template health detector initialized (min_confidence: {min_confidence}, "
            f"templates: {len(self.templates)}, method: {match_method})"
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
    ) -> Optional[HealthInfo]:
        """
        Detect health for a specific player using template matching.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard ("left" or "right")

        Returns:
            HealthInfo if successfully detected, None otherwise
        """
        # Check if templates are loaded
        if len(self.templates) == 0:
            log.error("No templates loaded, cannot detect health")
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

        # Get health crop
        if "health" not in player_crop_data:
            log.warning(f"Health region not found in player crop data")
            return None

        health_crop = player_crop_data["health"]

        if health_crop.size == 0:
            log.warning(f"Health crop is empty for player {player_index}")
            return None

        # Match templates
        result = self._match_health_region(health_crop)

        if result is None:
            log.debug(f"Template matching failed for player {player_index}")
            return None

        health, confidence, raw_text = result

        # Validate health range (0-150, allowing for overheal abilities)
        if not (0 <= health <= 150):
            log.warning(f"Health out of range for player {player_index}: {health}")
            return None

        log.debug(f"Player {player_index} health: {health} (confidence: {confidence:.2f})")

        return HealthInfo(
            health=health,
            confidence=confidence,
            raw_text=raw_text,
        )

    def _match_health_region(
        self, crop: np.ndarray
    ) -> Optional[tuple[int, float, str]]:
        """
        Match templates against a health region to extract the health value.

        Handles 1-3 digit health values (0-150).

        Args:
            crop: Cropped health region

        Returns:
            Tuple of (health, confidence, raw_text) or None if matching fails
        """
        # Preprocess crop
        preprocessed = self._preprocess_crop(crop)

        # Find all digit matches in the region
        matches = self._find_all_digit_matches(preprocessed)

        if len(matches) == 0:
            return None

        # Filter matches that are too close together (likely duplicates)
        filtered_matches = self._filter_overlapping_matches(matches)

        if len(filtered_matches) == 0:
            return None

        # Sort by x-position (left to right) to get correct digit order
        filtered_matches.sort(key=lambda m: m["x"])

        # Limit to 3 digits max (health can't be more than 3 digits)
        if len(filtered_matches) > 3:
            log.warning(f"Too many digits detected ({len(filtered_matches)}), keeping best 3")
            # Keep the 3 highest confidence matches
            filtered_matches = sorted(filtered_matches, key=lambda m: m["confidence"], reverse=True)[:3]
            # Re-sort by position after keeping top 3
            filtered_matches.sort(key=lambda m: m["x"])

        # Construct health value from digits
        digits = [m["digit"] for m in filtered_matches]
        raw_text = "".join(digits)

        try:
            health = int(raw_text)
        except ValueError:
            log.warning(f"Failed to parse health from matched digits: {raw_text}")
            return None

        # Use minimum confidence across all matched digits
        confidence = min(m["confidence"] for m in filtered_matches)

        # Check confidence threshold
        if confidence < self.min_confidence:
            log.debug(
                f"Template match confidence {confidence:.2f} below threshold {self.min_confidence}"
            )
            return None

        return health, confidence, raw_text

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
        self, matches: list[dict], overlap_threshold: float = 0.5
    ) -> list[dict]:
        """
        Filter out overlapping matches, keeping only the best match per region.

        Args:
            matches: List of match dictionaries
            overlap_threshold: Fraction of overlap to consider matches as duplicates

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

                # Calculate overlap ratio
                min_width = min(match["w"], accepted["w"])
                overlap_ratio = overlap_width / min_width if min_width > 0 else 0

                if overlap_ratio >= overlap_threshold:
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(match)

        return filtered

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess health crop for template matching.

        Produces white text on black background to match templates.

        Args:
            crop: Cropped health region

        Returns:
            Preprocessed grayscale binary image (white text on black)
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Upscale for better matching (3x)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # Use Otsu's thresholding - handles varying brightness
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def detect_with_debug(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> tuple[Optional[HealthInfo], np.ndarray, dict]:
        """
        Detect health and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard

        Returns:
            Tuple of (HealthInfo or None, health_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            return None, np.array([]), {}

        player_crop_data = player_crops[player_index]

        if "health" not in player_crop_data:
            return None, np.array([]), {}

        health_crop = player_crop_data["health"]

        if health_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess using the same method as detection
        health_preprocessed = self._preprocess_crop(health_crop)

        # Find all matches for debug visualization
        matches = self._find_all_digit_matches(health_preprocessed)

        debug_info = {
            "all_matches": matches,
            "filtered_matches": self._filter_overlapping_matches(matches),
        }

        # Run detection
        health_info = self.detect(frame, player_index, side)

        return health_info, health_preprocessed, debug_info
