"""
Pre-round ability detector for Valorant HUD.

Wrapper around AbilityDetector that uses pre-round player info regions.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from valoscribe.detectors.ability_detector import AbilityDetector
from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import AbilityInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class PreroundAbilityDetector(AbilityDetector):
    """
    Pre-round ability detector using blob detection.

    Inherits from AbilityDetector but uses pre-round player info regions.
    """

    def __init__(
        self,
        cropper: Cropper,
        brightness_threshold: int = 127,
        min_blob_area: int = 5,
        max_blob_area: int = 100,
    ):
        """
        Initialize pre-round ability detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            brightness_threshold: Minimum brightness to consider a pixel as part of a blob (0-255)
            min_blob_area: Minimum area in pixels for a valid blob
            max_blob_area: Maximum area in pixels for a valid blob
        """
        super().__init__(
            cropper=cropper,
            brightness_threshold=brightness_threshold,
            min_blob_area=min_blob_area,
            max_blob_area=max_blob_area,
        )
        log.info("Pre-round ability detector initialized")

    def detect_ability(
        self,
        frame: np.ndarray,
        player_index: int,
        ability_name: str,
        side: str = "left",
    ) -> Optional[AbilityInfo]:
        """
        Detect charges for a specific ability in pre-round HUD.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            ability_name: Name of ability region ("ability_1", "ability_2", or "ability_3")
            side: Which side of scoreboard ("left" or "right")

        Returns:
            AbilityInfo if successfully detected, None otherwise
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

        # Get specific ability crop
        if ability_name not in player_crop_data:
            log.warning(f"Ability '{ability_name}' not found in player crop data")
            return None

        ability_crop = player_crop_data[ability_name]

        if ability_crop.size == 0:
            log.warning(f"Ability crop is empty for player {player_index}, ability {ability_name}")
            return None

        # Detect blobs (uses parent's method which handles preprocessing internally)
        charges, total_blobs = self._count_blobs(ability_crop)

        log.debug(
            f"Player {player_index} {ability_name} (pre-round): {charges} charges "
            f"({total_blobs} total blobs detected)"
        )

        return AbilityInfo(
            charges=charges,
            total_blobs_detected=total_blobs,
        )
