"""
Pre-round ultimate detector for Valorant HUD.

Wrapper around UltimateDetector that uses pre-round player info regions.
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from valoscribe.detectors.ultimate_detector import UltimateDetector
from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import UltimateInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class PreroundUltimateDetector(UltimateDetector):
    """
    Pre-round ultimate detector using blob detection + pixel density.

    Inherits from UltimateDetector but uses pre-round player info regions.
    Uses two-stage approach:
    1. Check if ultimate is full (solid ring) using pixel density
    2. If not full, count segments using blob detection
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
        Initialize pre-round ultimate detector.

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
        super().__init__(
            cropper=cropper,
            brightness_threshold=brightness_threshold,
            center_mask_radius=center_mask_radius,
            ring_inner_radius=ring_inner_radius,
            ring_outer_radius=ring_outer_radius,
            fullness_threshold=fullness_threshold,
            min_blob_area=min_blob_area,
            max_blob_area=max_blob_area,
            min_circularity=min_circularity,
        )
        log.info("Pre-round ultimate detector initialized")

    def detect_ultimate(
        self,
        frame: np.ndarray,
        player_index: int,
        side: str = "left",
    ) -> Optional[tuple[UltimateInfo, float]]:
        """
        Detect ultimate charges for a specific player in pre-round HUD.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard ("left" or "right")

        Returns:
            Tuple of (UltimateInfo, white_pixel_ratio) if successfully detected, None otherwise
        """
        # Get PRE-ROUND player info crops (this is the key difference)
        player_crops = self.cropper.crop_player_info_preround(frame)

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

        # Preprocess crop (uses parent's method)
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
                f"Player {player_index} ultimate is FULL (pre-round, white pixel ratio: {white_pixel_ratio:.2f})"
            )
            return (
                UltimateInfo(
                    # TODO: change this hardcoded 7 charges when full to the appropriate amount per agent
                    charges=7,  # Full ultimate = max charges
                    is_full=True,
                    total_blobs_detected=0,
                ),
                white_pixel_ratio,
            )

        # Count blobs for partial ultimate (uses parent's method)
        # Apply ring mask to only count blobs in ring region
        masked = cv2.bitwise_and(preprocessed, preprocessed, mask=ring_mask)
        charges, total_blobs = self._count_blobs(masked)

        log.debug(
            f"Player {player_index} ultimate has {charges} charges (pre-round) "
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

    def detect_with_debug(
        self,
        frame: np.ndarray,
        player_index: int,
        side: str = "left",
    ) -> tuple[Optional[UltimateInfo], np.ndarray, dict]:
        """
        Detect pre-round ultimate and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard

        Returns:
            Tuple of (UltimateInfo or None, preprocessed_image, debug_info)
            debug_info contains detection details for debugging
        """
        # Get PRE-ROUND player info crops
        player_crops = self.cropper.crop_player_info_preround(frame)

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
