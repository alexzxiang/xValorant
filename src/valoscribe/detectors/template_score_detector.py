"""
Template-based score detector for Valorant HUD.

Uses template matching instead of OCR to detect team scores.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import ScoreInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class TemplateScoreDetector:
    """
    Template matching-based detector for extracting team scores.

    Uses pre-made templates of digits 0-9 to match against score regions
    instead of OCR, providing more robust detection especially for "0".
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        """
        Initialize template score detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_dir: Directory containing digit templates (0.png - 9.png)
                         If None, uses default: src/valoscribe/templates/score_digits/
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        # Set default template directory if not provided
        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "score_digits"

        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

        log.info(
            f"Template score detector initialized (min_confidence: {min_confidence}, "
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

    def detect(self, frame: np.ndarray) -> Optional[ScoreInfo]:
        """
        Detect team scores from a frame using template matching.

        Args:
            frame: Input frame (1080p)

        Returns:
            ScoreInfo if successfully detected, None otherwise
        """
        # Check if templates are loaded
        if len(self.templates) == 0:
            log.error("No templates loaded, cannot detect scores")
            return None

        # Crop both score regions
        team1_crop = self.cropper.crop_simple_region(frame, "team1_score")
        team2_crop = self.cropper.crop_simple_region(frame, "team2_score")

        if team1_crop.size == 0 or team2_crop.size == 0:
            log.warning("Score crop is empty")
            return None

        # Match templates for both teams
        team1_result = self._match_score_region(team1_crop)
        team2_result = self._match_score_region(team2_crop)

        if team1_result is None or team2_result is None:
            log.debug(
                f"Template matching failed: team1={team1_result}, team2={team2_result}"
            )
            return None

        team1_score, team1_conf, team1_text = team1_result
        team2_score, team2_conf, team2_text = team2_result

        # Basic sanity check - scores should be non-negative and reasonable
        # No upper limit since overtime can go arbitrarily high
        if team1_score < 0 or team2_score < 0 or team1_score > 99 or team2_score > 99:
            log.warning(
                f"Scores out of reasonable range: team1={team1_score}, team2={team2_score}"
            )
            return None

        # Use minimum confidence of the two scores
        confidence = min(team1_conf, team2_conf)

        log.debug(
            f"Template match: Team1={team1_score} ({team1_conf:.2f}), "
            f"Team2={team2_score} ({team2_conf:.2f})"
        )

        return ScoreInfo(
            team1_score=team1_score,
            team2_score=team2_score,
            confidence=confidence,
            team1_raw_text=team1_text,
            team2_raw_text=team2_text,
        )

    def _match_score_region(
        self, crop: np.ndarray
    ) -> Optional[tuple[int, float, str]]:
        """
        Match templates against a score region to extract the score.

        Handles both single-digit (0-9) and double-digit (10-13) scores.

        Args:
            crop: Cropped score region

        Returns:
            Tuple of (score, confidence, raw_text) or None if matching fails
        """
        # Preprocess crop
        preprocessed = self._preprocess_crop(crop)

        # Find all digit matches in the region
        matches = self._find_all_digit_matches(preprocessed)

        if len(matches) == 0:
            return None

        # Find all digit matches in the region
        # Filter matches that are too close together (likely duplicates)
        filtered_matches = self._filter_overlapping_matches(matches)

        if len(filtered_matches) == 0:
            return None

        # Sort by x-position (left to right) to get correct digit order
        filtered_matches.sort(key=lambda m: m["x"])

        # Construct score from digits
        digits = [m["digit"] for m in filtered_matches]
        raw_text = "".join(digits)

        try:
            score = int(raw_text)
        except ValueError:
            log.warning(f"Failed to parse score from matched digits: {raw_text}")
            return None

        # Use minimum confidence across all matched digits
        confidence = min(m["confidence"] for m in filtered_matches)

        # Check confidence threshold
        if confidence < self.min_confidence:
            log.debug(
                f"Template match confidence {confidence:.2f} below threshold {self.min_confidence}"
            )
            return None

        return score, confidence, raw_text

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
        Preprocess score crop for template matching.

        Produces white text on black background to match templates.

        Args:
            crop: Cropped score region

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

        # Use fixed threshold - results in white text on black background
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # No inversion - keep white text on black background for template matching

        return binary

    def detect_with_debug(
        self, frame: np.ndarray
    ) -> tuple[Optional[ScoreInfo], np.ndarray, np.ndarray, dict]:
        """
        Detect scores and return debug visualizations.

        Args:
            frame: Input frame (1080p)

        Returns:
            Tuple of (ScoreInfo or None, team1_preprocessed, team2_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Crop both score regions
        team1_crop = self.cropper.crop_simple_region(frame, "team1_score")
        team2_crop = self.cropper.crop_simple_region(frame, "team2_score")

        if team1_crop.size == 0 or team2_crop.size == 0:
            return None, np.array([]), np.array([]), {}

        # Preprocess using the same method as detection
        team1_preprocessed = self._preprocess_crop(team1_crop)
        team2_preprocessed = self._preprocess_crop(team2_crop)

        # Find all matches for debug visualization
        team1_matches = self._find_all_digit_matches(team1_preprocessed)
        team2_matches = self._find_all_digit_matches(team2_preprocessed)

        debug_info = {
            "team1_matches": team1_matches,
            "team2_matches": team2_matches,
        }

        # Run detection
        result = self.detect(frame)

        return result, team1_preprocessed, team2_preprocessed, debug_info
