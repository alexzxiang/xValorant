"""
Template-based spike detector for Valorant HUD.

Uses template matching to detect the spike icon that appears
in the timer region when the spike is planted.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import SpikeInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class TemplateSpikeDetector:
    """
    Template matching-based detector for spike icon.

    Detects the spike icon that appears in the timer region
    when the spike is planted, indicating a readable game frame
    during post-plant phase.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_path: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        """
        Initialize template spike detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_path: Path to spike template (spike.png)
                          If None, uses default: src/valoscribe/templates/spike/spike.png
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        # Set default template path if not provided
        if template_path is None:
            package_dir = Path(__file__).parent.parent
            template_path = package_dir / "templates" / "spike" / "spike.png"

        self.template_path = Path(template_path)
        self.template = self._load_template()

        log.info(
            f"Template spike detector initialized (min_confidence: {min_confidence}, "
            f"template_loaded: {self.template is not None}, method: {match_method})"
        )

    def _load_template(self) -> Optional[np.ndarray]:
        """
        Load spike template from template path.

        Returns:
            Template image if successful, None otherwise
        """
        if not self.template_path.exists():
            log.warning(f"Template file does not exist: {self.template_path}")
            return None

        # Load template in grayscale
        template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)

        if template is None:
            log.warning(f"Failed to load template: {self.template_path}")
            return None

        log.info(f"Loaded spike template: {template.shape}")
        return template

    def detect(self, frame: np.ndarray) -> Optional[SpikeInfo]:
        """
        Detect spike icon from a frame using template matching.

        Args:
            frame: Input frame (1080p)

        Returns:
            SpikeInfo if spike detected, None otherwise
        """
        # Check if template is loaded
        if self.template is None:
            log.error("No template loaded, cannot detect spike")
            return None

        # Crop timer region (spike appears where timer normally is)
        timer_crop = self.cropper.crop_simple_region(frame, "round_timer")

        if timer_crop.size == 0:
            log.warning("Timer crop is empty")
            return None

        # Preprocess crop
        preprocessed = self._preprocess_crop(timer_crop)

        # Match template
        result = cv2.matchTemplate(preprocessed, self.template, self.match_method)

        # Get best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # For TM_CCOEFF_NORMED, higher is better
        confidence = max_val

        # Check if confidence meets threshold
        if confidence >= self.min_confidence:
            log.debug(f"Spike detected with confidence: {confidence:.2f}")
            return SpikeInfo(
                spike_planted=True,
                confidence=float(confidence),
            )
        else:
            log.debug(f"Spike not detected (confidence: {confidence:.2f})")
            return None

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess timer crop for template matching.

        Uses same preprocessing as timer detector to ensure consistency.

        Args:
            crop: Cropped timer region

        Returns:
            Preprocessed grayscale binary image
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
        self, frame: np.ndarray
    ) -> tuple[Optional[SpikeInfo], np.ndarray, dict]:
        """
        Detect spike and return debug visualizations.

        Args:
            frame: Input frame (1080p)

        Returns:
            Tuple of (SpikeInfo or None, timer_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Crop timer region
        timer_crop = self.cropper.crop_simple_region(frame, "round_timer")

        if timer_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess using the same method as detection
        timer_preprocessed = self._preprocess_crop(timer_crop)

        # Get match result
        debug_info = {}
        if self.template is not None:
            result = cv2.matchTemplate(timer_preprocessed, self.template, self.match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            debug_info = {
                "max_confidence": float(max_val),
                "max_location": max_loc,
                "match_result_shape": result.shape,
            }

        # Run detection
        spike_info = self.detect(frame)

        return spike_info, timer_preprocessed, debug_info
