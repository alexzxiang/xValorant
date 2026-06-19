"""
Blob-based ultimate detector for Valorant HUD.

Uses a two-stage approach to detect ultimate charges:
1. Check if ultimate is full (solid ring) using pixel density
2. If not full, count segments using blob detection
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import UltimateInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class UltimateDetector:
    """
    Blob detection-based detector for ultimate charges.

    Detects ultimate segments in the ring around the ultimate icon.
    Uses pixel density to detect full ultimate (solid ring) and blob
    detection to count segments for partial ultimates.
    """

    def __init__(
        self,
        cropper: Cropper,
        brightness_threshold: int = 127,
        center_mask_radius: int = 12,
        ring_inner_radius: int = 13,
        ring_outer_radius: int = 23,
        fullness_threshold: float = 0.4,
        min_blob_area: int = 10,
        max_blob_area: int = 500,
        min_circularity: float = 0.2,
    ):
        """
        Initialize ultimate detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            brightness_threshold: Minimum brightness to consider a pixel as part of a blob (0-255)
            center_mask_radius: Radius of center circle to mask out icon (pixels)
            ring_inner_radius: Inner radius of ring region (pixels)
            ring_outer_radius: Outer radius of ring region (pixels)
            fullness_threshold: Minimum white pixel ratio (0-1) to consider ultimate full
            min_blob_area: Minimum area in pixels for a valid blob
            max_blob_area: Maximum area in pixels for a valid blob
            min_circularity: Minimum circularity (0-1) for blob validation
        """
        self.cropper = cropper
        self.brightness_threshold = brightness_threshold
        self.center_mask_radius = center_mask_radius
        self.ring_inner_radius = ring_inner_radius
        self.ring_outer_radius = ring_outer_radius
        self.fullness_threshold = fullness_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.min_circularity = min_circularity

        log.info(
            f"Ultimate detector initialized (brightness: {brightness_threshold}, "
            f"fullness_threshold: {fullness_threshold}, ring: {ring_inner_radius}-{ring_outer_radius}px)"
        )

    def detect_ultimate(
        self,
        frame: np.ndarray,
        player_index: int,
        side: str = "left",
    ) -> Optional[tuple[UltimateInfo, float]]:
        """
        Detect ultimate charges for a specific player.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard ("left" or "right")

        Returns:
            Tuple of (UltimateInfo, white_pixel_ratio) if successfully detected, None otherwise
        """
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

        # Get ultimate crop
        if "ultimate" not in player_crop_data:
            log.warning(f"Ultimate region not found in player crop data")
            return None

        ultimate_crop = player_crop_data["ultimate"]

        if ultimate_crop.size == 0:
            log.warning(f"Ultimate crop is empty for player {player_index}")
            return None

        # Preprocess crop
        preprocessed = self._preprocess_crop(ultimate_crop)

        # Mask out center icon
        h, w = preprocessed.shape
        center = (w // 2, h // 2)
        cv2.circle(preprocessed, center, self.center_mask_radius, 0, -1)

        # Create ring mask
        ring_mask = self._create_ring_mask(preprocessed.shape, center)

        # Calculate pixel density in ring
        ring_pixels = preprocessed[ring_mask > 0]
        if len(ring_pixels) == 0:
            log.warning(f"No pixels in ring region for player {player_index}")
            return None

        white_pixel_ratio = np.sum(ring_pixels == 255) / len(ring_pixels)

        # Check if full
        if white_pixel_ratio >= self.fullness_threshold:
            log.debug(
                f"Player {player_index} ultimate is FULL (white pixel ratio: {white_pixel_ratio:.2f})"
            )
            return (
                UltimateInfo(
                    # TODO: change this hardcoded 7 charges when full to the appropriate amount per agent
                    charges=7,  # Full ultimate = max charges (can be adjusted)
                    is_full=True,
                    total_blobs_detected=0,
                ),
                white_pixel_ratio,
            )

        # Count blobs for partial ultimate
        # Apply ring mask to only count blobs in ring region
        masked = cv2.bitwise_and(preprocessed, preprocessed, mask=ring_mask)
        charges, total_blobs = self._count_blobs(masked)

        log.debug(
            f"Player {player_index} ultimate has {charges} charges "
            f"(white ratio: {white_pixel_ratio:.2f}, total_blobs: {total_blobs})"
        )

        return (
            UltimateInfo(
                charges=charges,
                is_full=False,
                total_blobs_detected=total_blobs,
            ),
            white_pixel_ratio,
        )

    def _create_ring_mask(self, shape: tuple, center: tuple) -> np.ndarray:
        """
        Create annular (ring) mask for ultimate segments.

        Args:
            shape: Shape of image (h, w)
            center: Center point (x, y)

        Returns:
            Binary mask with ring region as white
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw outer circle
        cv2.circle(mask, center, self.ring_outer_radius, 255, -1)

        # Subtract inner circle
        cv2.circle(mask, center, self.ring_inner_radius, 0, -1)

        return mask

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess ultimate crop for blob detection.

        Args:
            crop: Cropped ultimate region

        Returns:
            Preprocessed grayscale binary image
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Apply brightness threshold
        _, binary = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        return binary

    def _count_blobs(self, binary_image: np.ndarray) -> tuple[int, int]:
        """
        Count blobs (ultimate segments) in binary image.

        Args:
            binary_image: Binary image with white blobs on black background

        Returns:
            Tuple of (valid_blob_count, total_blob_count)
        """
        # Find contours (blobs)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_blobs = len(contours)
        valid_blobs = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_blob_area or area > self.max_blob_area:
                continue

            # Filter by circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            valid_blobs += 1

        return valid_blobs, total_blobs

    def detect_with_debug(
        self,
        frame: np.ndarray,
        player_index: int,
        side: str = "left",
    ) -> tuple[Optional[UltimateInfo], np.ndarray, dict]:
        """
        Detect ultimate and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard

        Returns:
            Tuple of (UltimateInfo or None, preprocessed_image, debug_info)
            debug_info contains detection details for debugging
        """
        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            return None, np.array([]), {}

        player_crop_data = player_crops[player_index]

        if "ultimate" not in player_crop_data:
            return None, np.array([]), {}

        ultimate_crop = player_crop_data["ultimate"]

        if ultimate_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess
        preprocessed = self._preprocess_crop(ultimate_crop)

        # Mask center
        h, w = preprocessed.shape
        center = (w // 2, h // 2)
        preprocessed_with_mask = preprocessed.copy()
        cv2.circle(preprocessed_with_mask, center, self.center_mask_radius, 0, -1)

        # Create ring mask
        ring_mask = self._create_ring_mask(preprocessed.shape, center)

        # Calculate pixel density
        ring_pixels = preprocessed_with_mask[ring_mask > 0]
        white_pixel_ratio = np.sum(ring_pixels == 255) / len(ring_pixels) if len(ring_pixels) > 0 else 0

        # Count blobs
        masked = cv2.bitwise_and(preprocessed_with_mask, preprocessed_with_mask, mask=ring_mask)
        charges, total_blobs = self._count_blobs(masked)

        # Build debug info
        debug_info = {
            "white_pixel_ratio": float(white_pixel_ratio),
            "is_full": white_pixel_ratio >= self.fullness_threshold,
            "total_blobs": total_blobs,
            "valid_blobs": charges,
            "center": center,
            "ring_inner_radius": self.ring_inner_radius,
            "ring_outer_radius": self.ring_outer_radius,
        }

        # Run detection
        result = self.detect_ultimate(frame, player_index, side)

        if result is None:
            return None, preprocessed_with_mask, debug_info

        ultimate_info, white_pixel_ratio = result

        return ultimate_info, preprocessed_with_mask, debug_info
