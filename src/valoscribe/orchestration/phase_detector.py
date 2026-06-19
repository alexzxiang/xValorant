"""Phase detection for game state orchestration."""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional

import numpy as np

from valoscribe.detectors.template_timer_detector import TemplateTimerDetector
from valoscribe.detectors.template_spike_detector import TemplateSpikeDetector
from valoscribe.detectors.template_score_detector import TemplateScoreDetector
from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector
from valoscribe.types.detections import TimerInfo, SpikeInfo, ScoreInfo, CreditsInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class Phase(Enum):
    """Game phases for routing detection logic."""

    NON_GAME = auto()  # Not in game (timer not parseable)
    PREROUND = auto()  # Pre-round buy phase
    ACTIVE_ROUND = auto()  # Active gameplay
    POST_ROUND = auto()  # After round ends, before next pre-round


class PhaseDetector:
    """
    Detects the current game phase.

    Phase detection logic:
    1. Check spike planted indicator
       - If spike present -> NOT PREROUND
    2. Check timer
       - If timer NOT parseable AND spike NOT present -> NON_GAME
    3. Check preround UI (only if spike NOT present)
       - If preround credits detected -> PREROUND
    4. Check score change
       - If in ACTIVE_ROUND and score changed -> POST_ROUND
       - If already in POST_ROUND -> stay in POST_ROUND
       - Otherwise -> ACTIVE_ROUND (requires score detection)
    5. Score requirement for ACTIVE_ROUND
       - Score must be detected to classify as ACTIVE_ROUND
       - If no score detected -> NON_GAME

    Returns: (Phase, detections_dict)
    """

    def __init__(
        self,
        timer_detector: TemplateTimerDetector,
        spike_detector: TemplateSpikeDetector,
        score_detector: TemplateScoreDetector,
        credits_detector: PreroundCreditsDetector,
    ):
        """
        Initialize phase detector.

        Args:
            timer_detector: Timer detector instance
            spike_detector: Spike detector instance
            score_detector: Score detector instance
            credits_detector: Preround credits detector instance
        """
        self.timer_detector = timer_detector
        self.spike_detector = spike_detector
        self.score_detector = score_detector
        self.credits_detector = credits_detector

        # Track previous score for change detection
        self.previous_score: Optional[tuple[int, int]] = None

        log.info("PhaseDetector initialized")

    def detect_phase(
        self,
        frame: np.ndarray,
        current_phase: Phase,
    ) -> tuple[Phase, dict]:
        """
        Detect the current game phase and run associated detectors.

        Args:
            frame: Input frame (1080p)
            current_phase: Current phase (for state transitions)

        Returns:
            Tuple of (detected_phase, detections_dict)
            detections_dict contains: {
                "timer": Optional[TimerInfo],
                "spike": Optional[SpikeInfo],
                "score": Optional[ScoreInfo],
                "preround_credits": Optional[bool],  # True if any credits detected
            }
        """
        detections = {
            "timer": None,
            "spike": None,
            "score": None,
            "preround_credits": None,
        }

        # Step 1: Check spike planted indicator
        spike_info = self.spike_detector.detect(frame)
        detections["spike"] = spike_info
        spike_planted = spike_info.spike_planted if spike_info else False

        # Step 2: Check timer
        timer_info = self.timer_detector.detect(frame)
        detections["timer"] = timer_info

        # If timer NOT parseable AND spike NOT present -> NON_GAME
        if timer_info is None and not spike_planted:
            log.debug("Phase: NON_GAME (no timer, no spike)")
            return Phase.NON_GAME, detections

        # Step 3: Check preround UI (only if spike NOT present)
        if not spike_planted:
            # Check if any player has credits visible (preround indicator)
            # We'll check player 0 on left side as a heuristic
            # TODO: Could check multiple players for robustness
            credits_info = self.credits_detector.detect(frame, player_index=0, side="left")
            detections["preround_credits"] = (
                credits_info.credits_visible if credits_info else False
            )

            if credits_info and credits_info.credits_visible:
                log.debug("Phase: PREROUND (credits visible)")
                return Phase.PREROUND, detections

        # Step 4: Check score for phase transitions
        score_info = self.score_detector.detect(frame)
        detections["score"] = score_info

        if score_info:
            log.debug(f"Score detected: {score_info.team1_score}-{score_info.team2_score}")
        else:
            log.debug("Score not detected")

        if score_info:
            current_score_tuple = (score_info.team1_score, score_info.team2_score)

            # Check for score change
            score_changed = (
                self.previous_score is not None
                and current_score_tuple != self.previous_score
            )

            # Update previous score
            self.previous_score = current_score_tuple

            # If in ACTIVE_ROUND and score changed -> POST_ROUND
            if current_phase == Phase.ACTIVE_ROUND and score_changed:
                log.debug(
                    f"Phase: POST_ROUND (score changed from {self.previous_score} to {current_score_tuple})"
                )
                return Phase.POST_ROUND, detections

        # If already in POST_ROUND, check if we should continue
        if current_phase == Phase.POST_ROUND:
            # POST_ROUND should only last for the 7-8 second cooldown
            # If timer is high (>70s), we've transitioned to a new active round
            if timer_info and timer_info.time_seconds > 70.0:
                # Require score detection for ACTIVE_ROUND
                if score_info is None:
                    log.debug("Phase: NON_GAME (no score detected)")
                    return Phase.NON_GAME, detections

                log.debug(
                    f"Phase: ACTIVE_ROUND (timer={timer_info.time_seconds:.1f}s indicates new round started)"
                )
                return Phase.ACTIVE_ROUND, detections

            log.debug("Phase: POST_ROUND (continuing)")
            return Phase.POST_ROUND, detections

        # Default to ACTIVE_ROUND (but require score detection)
        if score_info is None:
            log.debug("Phase: NON_GAME (no score detected, cannot be ACTIVE_ROUND)")
            return Phase.NON_GAME, detections

        log.debug("Phase: ACTIVE_ROUND (default)")
        return Phase.ACTIVE_ROUND, detections
