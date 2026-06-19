"""
In-round credits detector for Valorant HUD.

OCRs the credits region in each player's active-round info panel to extract
the remaining credit amount (0-9000).
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.utils.ocr import OCREngine
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class InRoundCreditsDetector:
    """
    OCR-based detector for player credits during active round.

    The credits region (70×15 px) lives in the individual_player_info box at
    x=240, y=70. Right-side players are extracted from a horizontally-mirrored
    frame by the cropper, so the crop is flipped back here before OCR.
    """

    MAX_CREDITS = 9000

    def __init__(
        self,
        cropper: Cropper,
        ocr_engine: Optional[OCREngine] = None,
        min_confidence: float = 0.45,
    ):
        self.cropper = cropper
        self.ocr_engine = ocr_engine or OCREngine()
        self.min_confidence = min_confidence
        log.info("InRoundCreditsDetector initialized")

    def detect(self, frame: np.ndarray, player_index: int) -> Optional[int]:
        """
        Detect credits for one player during active round.

        Args:
            frame: Full 1080p broadcast frame.
            player_index: 0-9 (0-4 = left side, 5-9 = right side).

        Returns:
            Credits amount (0-9000), or None if unreadable.
        """
        player_crops = self.cropper.crop_player_info(frame)
        if player_index >= len(player_crops):
            return None

        credits_crop = player_crops[player_index].get("credits")
        if credits_crop is None or credits_crop.size == 0:
            return None

        # Right-side crops come from a mirrored frame; flip text back to be readable.
        if player_index >= 5:
            credits_crop = cv2.flip(credits_crop, 1)

        # Upscale 3× — the region is only 70×15, too small for reliable OCR.
        credits_crop = cv2.resize(
            credits_crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC
        )

        text, confidence = self.ocr_engine.read_single_line(
            credits_crop, whitelist="0123456789"
        )

        if confidence < self.min_confidence:
            log.debug(f"Player {player_index} credits low confidence: {confidence:.2f}")
            return None

        text = text.strip()
        if not text:
            return None

        try:
            amount = int(text)
        except ValueError:
            log.debug(f"Player {player_index} credits OCR bad value: '{text}'")
            return None

        if not (0 <= amount <= self.MAX_CREDITS):
            log.debug(f"Player {player_index} credits out of range: {amount}")
            return None

        log.debug(f"Player {player_index} credits={amount} (conf={confidence:.2f})")
        return amount
