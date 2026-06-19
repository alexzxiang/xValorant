"""Unit tests for PhaseDetector."""

from __future__ import annotations
from unittest.mock import Mock
import pytest
import numpy as np

from valoscribe.orchestration.phase_detector import Phase, PhaseDetector
from valoscribe.types.detections import TimerInfo, SpikeInfo, ScoreInfo, CreditsInfo


class TestPhaseDetector:
    """Tests for PhaseDetector class."""

    @pytest.fixture
    def mock_detectors(self):
        """Create mock detector instances."""
        return {
            "timer": Mock(),
            "spike": Mock(),
            "score": Mock(),
            "credits": Mock(),
        }

    @pytest.fixture
    def phase_detector(self, mock_detectors):
        """Create PhaseDetector with mocked detectors."""
        return PhaseDetector(
            timer_detector=mock_detectors["timer"],
            spike_detector=mock_detectors["spike"],
            score_detector=mock_detectors["score"],
            credits_detector=mock_detectors["credits"],
        )

    @pytest.fixture
    def dummy_frame(self):
        """Create dummy frame for testing."""
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_non_game_detection(self, phase_detector, mock_detectors, dummy_frame):
        """Test NON_GAME phase detection when timer not parseable."""
        # Setup: No timer, no spike
        mock_detectors["timer"].detect.return_value = None
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=False, confidence=0.9
        )

        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.ACTIVE_ROUND)

        assert phase == Phase.NON_GAME
        assert detections["timer"] is None
        assert detections["spike"].spike_planted is False

    def test_preround_detection(self, phase_detector, mock_detectors, dummy_frame):
        """Test PREROUND phase detection when credits visible."""
        # Setup: Timer present, no spike, credits visible
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=45.0, confidence=0.9, raw_text="045"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=False, confidence=0.9
        )
        mock_detectors["credits"].detect.return_value = CreditsInfo(
            credits_visible=True, confidence=0.9
        )

        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.ACTIVE_ROUND)

        assert phase == Phase.PREROUND
        assert detections["preround_credits"] is True

    def test_active_round_default(self, phase_detector, mock_detectors, dummy_frame):
        """Test ACTIVE_ROUND as default when no special conditions met."""
        # Setup: Timer present, no spike, no credits
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=80.0, confidence=0.9, raw_text="120"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=False, confidence=0.9
        )
        mock_detectors["credits"].detect.return_value = CreditsInfo(
            credits_visible=False, confidence=0.9
        )
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=3, team2_score=2, confidence=0.9
        )

        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.ACTIVE_ROUND)

        assert phase == Phase.ACTIVE_ROUND

    def test_post_round_on_score_change(
        self, phase_detector, mock_detectors, dummy_frame
    ):
        """Test POST_ROUND detection when score changes during ACTIVE_ROUND."""
        # Setup: Timer present, score detected
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=80.0, confidence=0.9, raw_text="120"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=True, confidence=0.9
        )
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=3, team2_score=2, confidence=0.9
        )

        # First call to establish previous score
        phase_detector.detect_phase(dummy_frame, Phase.ACTIVE_ROUND)

        # Second call with changed score
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=4, team2_score=2, confidence=0.9
        )

        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.ACTIVE_ROUND)

        assert phase == Phase.POST_ROUND

    def test_post_round_continues(self, phase_detector, mock_detectors, dummy_frame):
        """Test that POST_ROUND continues until PREROUND detected."""
        # Setup: Timer present, no spike, no credits
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=7.0, confidence=0.9, raw_text="007"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=False, confidence=0.9
        )
        mock_detectors["credits"].detect.return_value = CreditsInfo(
            credits_visible=False, confidence=0.9
        )
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=4, team2_score=2, confidence=0.9
        )

        # Call with POST_ROUND as current phase
        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.POST_ROUND)

        assert phase == Phase.POST_ROUND

    def test_spike_prevents_preround(self, phase_detector, mock_detectors, dummy_frame):
        """Test that spike planted prevents PREROUND detection."""
        # Setup: Timer present, spike planted, credits visible (shouldn't matter)
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=45.0, confidence=0.9, raw_text="045"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=True, confidence=0.9
        )
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=3, team2_score=2, confidence=0.9
        )

        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.ACTIVE_ROUND)

        # Should not be PREROUND since spike is planted
        assert phase != Phase.PREROUND
        # Credits detector should not be called when spike is planted
        mock_detectors["credits"].detect.assert_not_called()

    def test_post_round_transitions_on_high_timer(
        self, phase_detector, mock_detectors, dummy_frame
    ):
        """Test that POST_ROUND transitions to ACTIVE_ROUND when timer indicates new round."""
        # Setup: High timer (80s = new round started), no spike, no credits
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=80.0, confidence=0.9, raw_text="120"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=False, confidence=0.9
        )
        mock_detectors["credits"].detect.return_value = CreditsInfo(
            credits_visible=False, confidence=0.9
        )
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=4, team2_score=2, confidence=0.9
        )

        # Call with POST_ROUND as current phase (simulating missed PREROUND)
        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.POST_ROUND)

        # Should transition to ACTIVE_ROUND because timer > 70s
        assert phase == Phase.ACTIVE_ROUND

    def test_post_round_continues_with_low_timer(
        self, phase_detector, mock_detectors, dummy_frame
    ):
        """Test that POST_ROUND continues when timer shows cooldown period."""
        # Setup: Low timer (5s = still in post-round cooldown)
        mock_detectors["timer"].detect.return_value = TimerInfo(
            time_seconds=5.0, confidence=0.9, raw_text="005"
        )
        mock_detectors["spike"].detect.return_value = SpikeInfo(
            spike_planted=False, confidence=0.9
        )
        mock_detectors["credits"].detect.return_value = CreditsInfo(
            credits_visible=False, confidence=0.9
        )
        mock_detectors["score"].detect.return_value = ScoreInfo(
            team1_score=4, team2_score=2, confidence=0.9
        )

        # Call with POST_ROUND as current phase
        phase, detections = phase_detector.detect_phase(dummy_frame, Phase.POST_ROUND)

        # Should stay in POST_ROUND because timer < 70s
        assert phase == Phase.POST_ROUND
