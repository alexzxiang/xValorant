"""Timer manager for tracking game time across different contexts."""

from __future__ import annotations
from typing import Optional

from valoscribe.orchestration.phase_detector import Phase


class TimerManager:
    """
    Manages game timing across different contexts.

    Tracks three types of timers:
    1. game_timer: Visible round timer (100s countdown) - from detector
    2. spike_timer: Time since spike plant - calculated
    3. post_round_timer: Time since round end - calculated
    """

    def __init__(self):
        """Initialize timer manager."""
        self.spike_planted_at: Optional[float] = None
        self.round_ended_at: Optional[float] = None

    def on_spike_planted(self, timestamp: float) -> None:
        """
        Called when spike plant event occurs.

        Args:
            timestamp: Video timestamp when spike was planted
        """
        self.spike_planted_at = timestamp

    def on_round_ended(self, timestamp: float) -> None:
        """
        Called when round ends.

        Args:
            timestamp: Video timestamp when round ended
        """
        self.round_ended_at = timestamp

    def reset_for_new_round(self) -> None:
        """Reset timers for new round."""
        self.spike_planted_at = None
        self.round_ended_at = None

    def get_timers(
        self,
        timestamp: float,
        phase: Phase,
        game_timer: Optional[float],
    ) -> dict[str, Optional[float]]:
        """
        Calculate all timer values for current frame.

        Args:
            timestamp: Current video timestamp
            phase: Current game phase
            game_timer: Visible game timer from detector (or None if not visible)

        Returns:
            Dictionary with keys:
            - game_timer: Visible round timer (None if not visible)
            - spike_timer: Seconds since spike plant (None if not planted)
            - post_round_timer: Seconds since round end (None if not in post-round)
        """
        timers = {
            "game_timer": None,
            "spike_timer": None,
            "post_round_timer": None,
        }

        # Game timer (from detector, only visible pre-plant in active round)
        if phase == Phase.ACTIVE_ROUND and self.spike_planted_at is None:
            timers["game_timer"] = game_timer

        # Spike timer (time elapsed since plant, stops when post-round starts)
        if self.spike_planted_at is not None and phase != Phase.POST_ROUND:
            timers["spike_timer"] = timestamp - self.spike_planted_at

        # Post-round timer (time elapsed since round end)
        if phase == Phase.POST_ROUND and self.round_ended_at is not None:
            timers["post_round_timer"] = timestamp - self.round_ended_at

        return timers

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TimerManager("
            f"spike_planted_at={self.spike_planted_at}, "
            f"round_ended_at={self.round_ended_at})"
        )
