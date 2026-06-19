"""
Pre-round template-based credits detector for Valorant HUD.

Uses template matching to detect the credits icon in pre-round player info regions.
The presence of credits icons can be used to determine if a frame is in the pre-round state.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import numpy as np

from valoscribe.detectors.template_credits_detector import TemplateCreditsDetector
from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import CreditsInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class PreroundCreditsDetector(TemplateCreditsDetector):
    """
    Template matching-based detector for pre-round credits icon.

    Inherits from TemplateCreditsDetector but uses pre-round player info regions.
    Detects the credits icon in pre-round HUD to determine if frame is pre-round.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_path: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = 5,  # cv2.TM_CCOEFF_NORMED
    ):
        """
        Initialize pre-round template credits detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_path: Path to pre-round credits icon template
                          If None, uses default: src/valoscribe/templates/credits/credits_icon_preround.png
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
        """
        # Set default pre-round template path if not provided
        if template_path is None:
            package_dir = Path(__file__).parent.parent
            template_path = package_dir / "templates" / "credits" / "credits_icon_preround.png"

        # Call parent constructor with pre-round template
        super().__init__(
            cropper=cropper,
            template_path=template_path,
            min_confidence=min_confidence,
            match_method=match_method,
        )

        log.info("Pre-round credits detector initialized")

    def detect(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> Optional[CreditsInfo]:
        """
        Detect pre-round credits icon for a specific player using template matching.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard ("left" or "right")

        Returns:
            CreditsInfo if successfully checked, None if error
        """
        # Check if template is loaded
        if self.template is None:
            log.error("No template loaded, cannot detect pre-round credits")
            return None

        # Get PRE-ROUND player info crops
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

        # Get credits crop
        if "credits" not in player_crop_data:
            log.warning(f"Credits region not found in player crop data")
            return None

        credits_crop = player_crop_data["credits"]

        if credits_crop.size == 0:
            log.warning(f"Credits crop is empty for player {player_index}")
            return None

        # Preprocess crop (uses parent's method)
        preprocessed = self._preprocess_crop(credits_crop)

        # Match template (uses parent's logic)
        import cv2
        result = cv2.matchTemplate(preprocessed, self.template, self.match_method)

        # Get best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # For TM_CCOEFF_NORMED, higher is better
        # Clamp confidence to [0.0, 1.0] range
        confidence = max(0.0, min(1.0, float(max_val)))

        # Check if confidence meets threshold
        if confidence >= self.min_confidence:
            log.debug(f"Player {player_index} pre-round credits visible with confidence: {confidence:.2f}")
            return CreditsInfo(
                credits_visible=True,
                confidence=confidence,
            )
        else:
            log.debug(f"Player {player_index} pre-round credits not visible (confidence: {confidence:.2f})")
            return CreditsInfo(
                credits_visible=False,
                confidence=confidence,
            )

    def detect_with_debug(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> tuple[Optional[CreditsInfo], np.ndarray, dict]:
        """
        Detect pre-round credits and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard

        Returns:
            Tuple of (CreditsInfo or None, credits_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Get PRE-ROUND player info crops
        player_crops = self.cropper.crop_player_info_preround(frame)

        if player_index >= len(player_crops):
            return None, np.array([]), {}

        player_crop_data = player_crops[player_index]

        if "credits" not in player_crop_data:
            return None, np.array([]), {}

        credits_crop = player_crop_data["credits"]

        if credits_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess using the same method as detection (from parent)
        credits_preprocessed = self._preprocess_crop(credits_crop)

        # Get match result
        debug_info = {}
        if self.template is not None:
            import cv2
            result = cv2.matchTemplate(credits_preprocessed, self.template, self.match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Clamp confidence to [0.0, 1.0] range
            clamped_confidence = max(0.0, min(1.0, float(max_val)))

            debug_info = {
                "max_confidence": clamped_confidence,
                "max_location": max_loc,
                "match_result_shape": result.shape,
            }

        # Run detection
        credits_info = self.detect(frame, player_index, side)

        return credits_info, credits_preprocessed, debug_info

    def is_preround_frame(self, frame: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Determine if a frame is in pre-round state by checking all players.

        A frame is considered pre-round if more than `threshold` fraction of
        players have visible pre-round credits icons.

        Args:
            frame: Input frame (1080p)
            threshold: Minimum fraction of players with visible credits (0-1, default 0.5)

        Returns:
            True if frame appears to be pre-round, False otherwise
        """
        visible_count = 0
        total_checked = 0

        # Check all 10 players
        for player_idx in range(10):
            # Determine side (0-4 left, 5-9 right)
            side = "left" if player_idx < 5 else "right"

            # Detect credits
            result = self.detect(frame, player_idx, side)

            if result is not None:
                total_checked += 1
                if result.credits_visible:
                    visible_count += 1

        # If we couldn't check any players, return False
        if total_checked == 0:
            log.warning("Could not check any players for pre-round status")
            return False

        # Calculate fraction of visible credits
        visible_fraction = visible_count / total_checked

        is_preround = visible_fraction >= threshold

        log.debug(
            f"Pre-round check: {visible_count}/{total_checked} players "
            f"({visible_fraction:.1%}) have visible credits. "
            f"Is pre-round: {is_preround} (threshold: {threshold:.1%})"
        )

        return is_preround
