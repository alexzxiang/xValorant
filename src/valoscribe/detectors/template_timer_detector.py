"""
Template-based timer detector for Valorant HUD.

Uses template matching instead of OCR to detect round timer.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import TimerInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class TemplateTimerDetector:
    """
    Template matching-based detector for extracting round timer.

    Uses pre-made templates of digits 0-9 to match against timer region.
    Handles both m:ss format (3 digits) and ss.ms format (4 digits) by
    detecting only digits and inferring format from count.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.6,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        """
        Initialize template timer detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_dir: Directory containing digit templates (0.png - 9.png)
                         If None, uses default: src/valoscribe/templates/timer_digits/
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        # Set default template directory if not provided
        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "timer_digits"

        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

        log.info(
            f"Template timer detector initialized (min_confidence: {min_confidence}, "
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

    def detect(self, frame: np.ndarray) -> Optional[TimerInfo]:
        """
        Detect round timer from a frame using template matching.

        Args:
            frame: Input frame (1080p)

        Returns:
            TimerInfo if successfully detected, None otherwise
        """
        # Check if templates are loaded
        if len(self.templates) == 0:
            log.error("No templates loaded, cannot detect timer")
            return None

        # Crop timer region
        timer_crop = self.cropper.crop_simple_region(frame, "round_timer")

        if timer_crop.size == 0:
            log.warning("Timer crop is empty")
            return None

        # Match templates
        result = self._match_timer_region(timer_crop)

        if result is None:
            log.debug("Template matching failed for timer")
            return None

        time_seconds, confidence, raw_text = result

        # Validate time range (0-100 seconds for Valorant rounds)
        if not (0.0 <= time_seconds <= 100.0):
            log.warning(f"Time out of range: {time_seconds}s")
            return None

        log.debug(f"Template match: Timer={time_seconds:.2f}s ({confidence:.2f})")

        return TimerInfo(
            time_seconds=time_seconds,
            confidence=confidence,
            raw_text=raw_text,
        )

    def _match_timer_region(
        self, crop: np.ndarray
    ) -> Optional[tuple[float, float, str]]:
        """
        Match templates against timer region to extract the time.

        Handles both m:ss format (3 digits) and ss.ms format (4 digits).

        Args:
            crop: Cropped timer region

        Returns:
            Tuple of (time_seconds, confidence, raw_text) or None if matching fails
        """
        # Preprocess crop
        preprocessed = self._preprocess_crop(crop)

        # Find all digit matches in the region
        matches = self._find_all_digit_matches(preprocessed)

        if len(matches) == 0:
            return None

        # Filter overlapping matches
        filtered_matches = self._filter_overlapping_matches(matches)

        if len(filtered_matches) == 0:
            return None

        # Sort by x-position (left to right) to get correct digit order
        filtered_matches.sort(key=lambda m: m["x"])

        # Construct digit string
        digits = [m["digit"] for m in filtered_matches]
        raw_text = "".join(digits)

        # Parse based on digit count
        try:
            if len(digits) == 3:
                # m:ss format (e.g., "145" = 1:45 = 105 seconds)
                minutes = int(digits[0])
                seconds = int(digits[1] + digits[2])
                time_seconds = minutes * 60.0 + seconds
            elif len(digits) == 4:
                # ss.ms format (e.g., "0967" = 09.67 seconds)
                seconds = int(digits[0] + digits[1])
                centiseconds = int(digits[2] + digits[3])
                time_seconds = seconds + centiseconds / 100.0
            else:
                log.warning(
                    f"Invalid digit count for timer: {len(digits)} (expected 3 or 4)"
                )
                return None
        except ValueError:
            log.warning(f"Failed to parse time from matched digits: {raw_text}")
            return None

        # Use minimum confidence across all matched digits
        confidence = min(m["confidence"] for m in filtered_matches)

        # Check confidence threshold
        if confidence < self.min_confidence:
            log.debug(
                f"Template match confidence {confidence:.2f} below threshold {self.min_confidence}"
            )
            return None

        return time_seconds, confidence, raw_text

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
        Preprocess timer crop for template matching.

        Produces white text on black background to match templates.
        Uses Otsu's thresholding to handle both white (normal) and red (low) timer text.

        Args:
            crop: Cropped timer region

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

        # Use Otsu's thresholding - handles both white and red text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def detect_with_debug(
        self, frame: np.ndarray
    ) -> tuple[Optional[TimerInfo], np.ndarray, dict]:
        """
        Detect timer and return debug visualizations.

        Args:
            frame: Input frame (1080p)

        Returns:
            Tuple of (TimerInfo or None, timer_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Crop timer region
        timer_crop = self.cropper.crop_simple_region(frame, "round_timer")

        if timer_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess using the same method as detection
        timer_preprocessed = self._preprocess_crop(timer_crop)

        # Find all matches for debug visualization
        timer_matches = self._find_all_digit_matches(timer_preprocessed)

        debug_info = {
            "timer_matches": timer_matches,
        }

        # Run detection
        result = self.detect(frame)

        return result, timer_preprocessed, debug_info
