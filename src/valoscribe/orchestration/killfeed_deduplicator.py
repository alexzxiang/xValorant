"""Killfeed deduplicator for preventing duplicate event logging."""

from __future__ import annotations
from typing import Optional
from collections import deque

from valoscribe.types.detections import KillfeedAgentDetection
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class KillfeedDeduplicator:
    """
    Prevents duplicate killfeed events from being logged.

    Killfeed entries persist on screen for ~5 seconds, appearing across
    multiple frames. This deduplicator tracks recent kills and prevents
    the same kill from being logged multiple times.

    Kill signature: (killer_agent, killer_side, victim_agent, victim_side)
    """

    def __init__(self, dedup_window_sec: float = 5.0):
        """
        Initialize killfeed deduplicator.

        Args:
            dedup_window_sec: Time window for deduplication (default: 5.0 seconds)
        """
        self.dedup_window_sec = dedup_window_sec
        # Store (timestamp, kill_signature) tuples
        self.recent_kills: deque[tuple[float, tuple[str, str, str, str]]] = deque()

        log.debug(f"KillfeedDeduplicator initialized with {dedup_window_sec}s window")

    def is_duplicate(
        self,
        timestamp: float,
        detection: KillfeedAgentDetection,
    ) -> bool:
        """
        Check if a killfeed detection is a duplicate of a recent kill.

        Args:
            timestamp: Current frame timestamp in seconds
            detection: Killfeed detection to check

        Returns:
            True if this is a duplicate (already logged), False if it's new
        """
        # Create kill signature
        kill_signature = (
            detection.killer_agent,
            detection.killer_side,
            detection.victim_agent,
            detection.victim_side,
        )

        # Remove old kills outside the deduplication window
        self._cleanup_old_kills(timestamp)

        # Check if this kill signature exists in recent kills
        for _, existing_signature in self.recent_kills:
            if existing_signature == kill_signature:
                log.debug(
                    f"Duplicate kill detected: {kill_signature[0]} ({kill_signature[1]}) "
                    f"-> {kill_signature[2]} ({kill_signature[3]})"
                )
                return True

        # Not a duplicate - add to recent kills
        self.recent_kills.append((timestamp, kill_signature))
        log.debug(
            f"New kill logged: {kill_signature[0]} ({kill_signature[1]}) "
            f"-> {kill_signature[2]} ({kill_signature[3]})"
        )
        return False

    def _cleanup_old_kills(self, current_timestamp: float) -> None:
        """
        Remove kills older than the deduplication window.

        Args:
            current_timestamp: Current timestamp in seconds
        """
        cutoff_time = current_timestamp - self.dedup_window_sec

        # Remove from front of deque while timestamps are old
        while self.recent_kills and self.recent_kills[0][0] < cutoff_time:
            removed = self.recent_kills.popleft()
            log.debug(f"Removed old kill from dedup cache: {removed[1]}")

    def reset(self) -> None:
        """Reset the deduplicator, clearing all recent kills."""
        self.recent_kills.clear()
        log.debug("KillfeedDeduplicator reset")

    def get_recent_kill_count(self) -> int:
        """
        Get the number of kills currently in the deduplication cache.

        Returns:
            Number of recent kills being tracked
        """
        return len(self.recent_kills)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KillfeedDeduplicator("
            f"window={self.dedup_window_sec}s, "
            f"recent_kills={len(self.recent_kills)})"
        )
