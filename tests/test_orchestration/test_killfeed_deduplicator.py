"""Unit tests for KillfeedDeduplicator."""

from __future__ import annotations
import pytest

from valoscribe.orchestration.killfeed_deduplicator import KillfeedDeduplicator
from valoscribe.types.detections import KillfeedAgentDetection


class TestKillfeedDeduplicator:
    """Tests for KillfeedDeduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        """Create KillfeedDeduplicator instance."""
        return KillfeedDeduplicator()

    @pytest.fixture
    def sample_kill(self):
        """Create sample killfeed detection."""
        return KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )

    def test_init(self, deduplicator):
        """Test KillfeedDeduplicator initialization."""
        assert deduplicator.dedup_window_sec == 5.0
        assert len(deduplicator.recent_kills) == 0

    def test_init_custom_window(self):
        """Test initialization with custom dedup window."""
        deduplicator = KillfeedDeduplicator(dedup_window_sec=10.0)
        assert deduplicator.dedup_window_sec == 10.0

    def test_first_kill_not_duplicate(self, deduplicator, sample_kill):
        """Test that first kill is not detected as duplicate."""
        is_dup = deduplicator.is_duplicate(timestamp=0.0, detection=sample_kill)
        assert not is_dup
        assert deduplicator.get_recent_kill_count() == 1

    def test_same_kill_is_duplicate(self, deduplicator, sample_kill):
        """Test that same kill within window is detected as duplicate."""
        # First occurrence
        deduplicator.is_duplicate(timestamp=0.0, detection=sample_kill)

        # Same kill 1 second later - should be duplicate
        is_dup = deduplicator.is_duplicate(timestamp=1.0, detection=sample_kill)
        assert is_dup
        assert deduplicator.get_recent_kill_count() == 1  # Still only one kill tracked

    def test_different_kills_not_duplicate(self, deduplicator):
        """Test that different kills are not detected as duplicates."""
        kill1 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )
        kill2 = KillfeedAgentDetection(
            killer_agent="raze",
            killer_side="attack",
            victim_agent="cypher",
            victim_side="defense",
            confidence=0.95,
        )

        # First kill
        is_dup1 = deduplicator.is_duplicate(timestamp=0.0, detection=kill1)
        assert not is_dup1

        # Different kill - not a duplicate
        is_dup2 = deduplicator.is_duplicate(timestamp=1.0, detection=kill2)
        assert not is_dup2

        assert deduplicator.get_recent_kill_count() == 2

    def test_same_agents_different_sides_not_duplicate(self, deduplicator):
        """Test that same agents on different sides are not duplicates."""
        kill1 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )
        kill2 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="defense",  # Different side
            victim_agent="sova",
            victim_side="attack",  # Different side
            confidence=0.95,
        )

        is_dup1 = deduplicator.is_duplicate(timestamp=0.0, detection=kill1)
        is_dup2 = deduplicator.is_duplicate(timestamp=1.0, detection=kill2)

        assert not is_dup1
        assert not is_dup2
        assert deduplicator.get_recent_kill_count() == 2

    def test_kill_expires_after_window(self, deduplicator, sample_kill):
        """Test that kills expire after deduplication window."""
        # First occurrence at t=0
        deduplicator.is_duplicate(timestamp=0.0, detection=sample_kill)
        assert deduplicator.get_recent_kill_count() == 1

        # Same kill at t=6 (beyond 5 second window) - should not be duplicate
        is_dup = deduplicator.is_duplicate(timestamp=6.0, detection=sample_kill)
        assert not is_dup
        assert deduplicator.get_recent_kill_count() == 1  # Old kill removed, new one added

    def test_kill_within_window_is_duplicate(self, deduplicator, sample_kill):
        """Test that kill within window is still duplicate."""
        # First occurrence at t=0
        deduplicator.is_duplicate(timestamp=0.0, detection=sample_kill)

        # Same kill at t=4.9 (within 5 second window) - should be duplicate
        is_dup = deduplicator.is_duplicate(timestamp=4.9, detection=sample_kill)
        assert is_dup
        assert deduplicator.get_recent_kill_count() == 1

    def test_cleanup_removes_old_kills(self, deduplicator):
        """Test that cleanup removes old kills."""
        kill1 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )
        kill2 = KillfeedAgentDetection(
            killer_agent="raze",
            killer_side="attack",
            victim_agent="cypher",
            victim_side="defense",
            confidence=0.95,
        )

        # Add two kills
        deduplicator.is_duplicate(timestamp=0.0, detection=kill1)
        deduplicator.is_duplicate(timestamp=3.0, detection=kill2)
        assert deduplicator.get_recent_kill_count() == 2

        # At t=7, first kill should be removed (0 + 5 < 7)
        deduplicator._cleanup_old_kills(current_timestamp=7.0)
        assert deduplicator.get_recent_kill_count() == 1

        # At t=10, second kill should also be removed (3 + 5 < 10)
        deduplicator._cleanup_old_kills(current_timestamp=10.0)
        assert deduplicator.get_recent_kill_count() == 0

    def test_multiple_kills_same_frame(self, deduplicator):
        """Test handling multiple different kills at same timestamp."""
        kill1 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )
        kill2 = KillfeedAgentDetection(
            killer_agent="raze",
            killer_side="attack",
            victim_agent="cypher",
            victim_side="defense",
            confidence=0.95,
        )

        # Both kills at same timestamp
        is_dup1 = deduplicator.is_duplicate(timestamp=0.0, detection=kill1)
        is_dup2 = deduplicator.is_duplicate(timestamp=0.0, detection=kill2)

        assert not is_dup1
        assert not is_dup2
        assert deduplicator.get_recent_kill_count() == 2

    def test_reset(self, deduplicator, sample_kill):
        """Test reset clears all recent kills."""
        # Add some kills
        deduplicator.is_duplicate(timestamp=0.0, detection=sample_kill)
        assert deduplicator.get_recent_kill_count() == 1

        # Reset
        deduplicator.reset()
        assert deduplicator.get_recent_kill_count() == 0

        # Same kill after reset should not be duplicate
        is_dup = deduplicator.is_duplicate(timestamp=1.0, detection=sample_kill)
        assert not is_dup

    def test_repr(self, deduplicator):
        """Test string representation."""
        repr_str = repr(deduplicator)
        assert "KillfeedDeduplicator" in repr_str
        assert "window=5.0s" in repr_str
        assert "recent_kills=0" in repr_str

    def test_realistic_scenario(self, deduplicator):
        """Test realistic scenario with multiple kills over time."""
        # Kill 1: jett kills sova at t=0
        kill1 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )
        assert not deduplicator.is_duplicate(timestamp=0.0, detection=kill1)

        # Same kill detected on next 10 frames (0.25s apart)
        for i in range(1, 11):
            t = i * 0.25
            assert deduplicator.is_duplicate(timestamp=t, detection=kill1)

        # Kill 2: raze kills cypher at t=3
        kill2 = KillfeedAgentDetection(
            killer_agent="raze",
            killer_side="attack",
            victim_agent="cypher",
            victim_side="defense",
            confidence=0.95,
        )
        assert not deduplicator.is_duplicate(timestamp=3.0, detection=kill2)

        # Both kills still in cache
        assert deduplicator.get_recent_kill_count() == 2

        # At t=6, kill1 should expire, kill2 still active
        kill3 = KillfeedAgentDetection(
            killer_agent="omen",
            killer_side="defense",
            victim_agent="breach",
            victim_side="attack",
            confidence=0.95,
        )
        assert not deduplicator.is_duplicate(timestamp=6.0, detection=kill3)
        assert deduplicator.get_recent_kill_count() == 2  # kill2 and kill3

        # Kill1 reappears at t=7 - should not be duplicate (expired)
        assert not deduplicator.is_duplicate(timestamp=7.0, detection=kill1)

    def test_custom_window_size(self):
        """Test deduplicator with custom window size."""
        # 10 second window
        deduplicator = KillfeedDeduplicator(dedup_window_sec=10.0)

        kill = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )

        # First occurrence
        deduplicator.is_duplicate(timestamp=0.0, detection=kill)

        # At t=9 (within 10s window) - should be duplicate
        assert deduplicator.is_duplicate(timestamp=9.0, detection=kill)

        # At t=11 (beyond 10s window) - should not be duplicate
        assert not deduplicator.is_duplicate(timestamp=11.0, detection=kill)
