"""
Round number detector for Valorant HUD.

Extracts the current round number from the game HUD using OCR.
"""

from __future__ import annotations
from typing import Optional
import re

import numpy as np

import cv2

from valoscribe.detectors.cropper import Cropper
from valoscribe.utils.ocr import OCREngine
from valoscribe.types.detections import RoundInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class RoundDetector:
    """
    Detector for extracting round numbers from Valorant HUD.

    The round number appears in the format "ROUND X/24" or just "X/24"
    at the top center of the screen.
    """

    def __init__(
        self,
        cropper: Cropper,
        ocr_engine: Optional[OCREngine] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize round detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            ocr_engine: OCR engine instance (creates default if None)
            min_confidence: Minimum confidence threshold for valid detection (0-1)
        """
        self.cropper = cropper
        self.ocr_engine = ocr_engine or OCREngine()
        self.min_confidence = min_confidence
        log.info(f"Round detector initialized (min_confidence: {min_confidence})")

    def detect(self, frame: np.ndarray) -> Optional[RoundInfo]:
        """
        Detect round number from a frame.

        Args:
            frame: Input frame (1080p)

        Returns:
            RoundInfo if successfully detected, None otherwise
        """
        # Crop round number region
        round_crop = self.cropper.crop_simple_region(frame, "round_number")

        if round_crop.size == 0:
            log.warning("Round number crop is empty")
            return None

        # Preprocess the crop for better OCR
        # The round number is typically white/light text on dark background
        preprocessed = self._preprocess_round_crop(round_crop)

        # Apply OCR with appropriate whitelist (don't preprocess again)
        # Include ROUND, digits, slash, and space
        text, confidence = self.ocr_engine.read_single_line(
            preprocessed, whitelist="ROUND0123456789/ ", preprocess=False
        )

        log.debug(f"Round OCR: '{text}' (confidence: {confidence:.2f})")

        # Parse round number
        round_number = self._parse_round_number(text)

        if round_number is None:
            log.warning(f"Failed to parse round number from text: '{text}'")
            return None

        # Check confidence threshold
        if confidence < self.min_confidence:
            log.warning(
                f"Round detection confidence {confidence:.2f} below threshold {self.min_confidence}"
            )
            return None

        return RoundInfo(
            round_number=round_number,
            confidence=confidence,
            raw_text=text,
        )

    def _preprocess_round_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess round number crop for OCR.

        The round number is typically displayed as white/light text on a dark background.
        This preprocessing is optimized for that scenario.

        Args:
            crop: Cropped round number region

        Returns:
            Preprocessed binary image suitable for OCR
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Upscale for better OCR (3x)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # Use simple fixed threshold instead of Otsu
        # This is more conservative and less likely to create false connections
        # Threshold at 50% - anything brighter than mid-gray becomes white
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Check if we need to invert (OCR expects dark text on light background)
        # If most of the image is dark, we probably have light text on dark background
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return binary

    def _parse_round_number(self, text: str) -> Optional[int]:
        """
        Parse round number from OCR text.

        Expected formats:
        - "ROUND 5/24"
        - "5/24"
        - "ROUND 5"
        - "5"

        Args:
            text: OCR text output

        Returns:
            Parsed round number or None if parsing fails
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Try to extract number before slash (e.g., "5/24" -> 5)
        # Look for pattern at word boundary or after "ROUND"
        match = re.search(r"(?:ROUND\s+)?(\d+)\s*/\s*\d+", text, re.IGNORECASE)
        if match:
            round_num = int(match.group(1))
            if 1 <= round_num <= 24:
                return round_num
            # If first number is invalid, don't fall through to other patterns
            return None

        # Try to extract number after "ROUND" keyword
        match = re.search(r"ROUND\s+(\d+)", text, re.IGNORECASE)
        if match:
            round_num = int(match.group(1))
            if 1 <= round_num <= 24:
                return round_num

        # Try to extract just a number at start of string (fallback)
        match = re.match(r"^\s*(\d+)", text)
        if match:
            round_num = int(match.group(1))
            # Validate it's in reasonable range
            if 1 <= round_num <= 24:
                return round_num

        return None

    def detect_with_debug(self, frame: np.ndarray) -> tuple[Optional[RoundInfo], np.ndarray]:
        """
        Detect round number and return debug visualization.

        Args:
            frame: Input frame (1080p)

        Returns:
            Tuple of (RoundInfo or None, preprocessed crop image for debugging)
        """
        # Crop round number region
        round_crop = self.cropper.crop_simple_region(frame, "round_number")

        if round_crop.size == 0:
            return None, np.array([])

        # Preprocess using the same method as detection
        preprocessed = self._preprocess_round_crop(round_crop)

        # Run detection
        result = self.detect(frame)

        return result, preprocessed
