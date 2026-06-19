"""
Unit tests for the minimap position parser.

Covers:
  - _match_blobs_to_trackers: Hungarian assignment, EMA velocity, identity-swap
    prevention, max-distance rejection, miss decay, pass-2 seeding.
  - MinimapDetector construction (no real video needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from valoscribe.orchestration.game_state_manager import GameStateManager
from valoscribe.detectors.minimap_detector import MinimapDetector
from valoscribe.types.detections import PlayerPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(x: float, y: float) -> PlayerPosition:
    return PlayerPosition(x=x, y=y, confidence=1.0)


class _Tracker:
    """Minimal tracker stub matching the interface used by _match_blobs_to_trackers."""

    def __init__(self, pos_x=None, pos_y=None, vx=0.0, vy=0.0):
        self.current_state: dict = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_vx": vx,
            "pos_vy": vy,
        }


MAX = MinimapDetector.TRACK_MAX_DIST   # 0.20


# ---------------------------------------------------------------------------
# Basic assignment
# ---------------------------------------------------------------------------

class TestBasicAssignment:
    def test_single_tracker_single_blob_assigned(self):
        tracker = _Tracker(pos_x=0.5, pos_y=0.5)
        blobs = [_pos(0.5, 0.5)]
        GameStateManager._match_blobs_to_trackers(blobs, [tracker], MAX)
        assert tracker.current_state["pos_x"] == 0.5
        assert tracker.current_state["pos_y"] == 0.5

    def test_two_trackers_two_blobs_matched_correctly(self):
        t1 = _Tracker(pos_x=0.1, pos_y=0.1)
        t2 = _Tracker(pos_x=0.9, pos_y=0.9)
        blobs = [_pos(0.9, 0.9), _pos(0.1, 0.1)]  # reversed order
        GameStateManager._match_blobs_to_trackers(blobs, [t1, t2], MAX)
        # Hungarian should match t1→(0.1,0.1) and t2→(0.9,0.9), not the naive order
        assert t1.current_state["pos_x"] == pytest.approx(0.1)
        assert t2.current_state["pos_x"] == pytest.approx(0.9)

    def test_no_blobs_clears_all_trackers(self):
        t = _Tracker(pos_x=0.5, pos_y=0.5)
        GameStateManager._match_blobs_to_trackers([], [t], MAX)
        assert t.current_state["pos_x"] is None

    def test_no_trackers_no_crash(self):
        # Should not raise
        GameStateManager._match_blobs_to_trackers([_pos(0.5, 0.5)], [], MAX)

    def test_more_blobs_than_trackers(self):
        t = _Tracker(pos_x=0.3, pos_y=0.3)
        blobs = [_pos(0.3, 0.3), _pos(0.7, 0.7)]
        GameStateManager._match_blobs_to_trackers(blobs, [t], MAX)
        assert t.current_state["pos_x"] == pytest.approx(0.3)

    def test_more_trackers_than_blobs(self):
        t1 = _Tracker(pos_x=0.2, pos_y=0.2)
        t2 = _Tracker(pos_x=0.8, pos_y=0.8)
        blobs = [_pos(0.2, 0.2)]  # only one blob
        GameStateManager._match_blobs_to_trackers(blobs, [t1, t2], MAX)
        assert t1.current_state["pos_x"] == pytest.approx(0.2)
        assert t2.current_state["pos_x"] is None


# ---------------------------------------------------------------------------
# Max-distance rejection
# ---------------------------------------------------------------------------

class TestMaxDistance:
    def test_blob_within_max_dist_assigned(self):
        t = _Tracker(pos_x=0.5, pos_y=0.5)
        blobs = [_pos(0.5 + MAX * 0.9, 0.5)]  # just inside threshold
        GameStateManager._match_blobs_to_trackers(blobs, [t], MAX)
        assert t.current_state["pos_x"] is not None

    def test_blob_beyond_max_dist_rejected(self):
        t = _Tracker(pos_x=0.5, pos_y=0.5)
        blobs = [_pos(0.5 + MAX * 1.1, 0.5)]  # just outside threshold
        GameStateManager._match_blobs_to_trackers(blobs, [t], MAX)
        assert t.current_state["pos_x"] is None

    def test_only_closer_blob_wins_when_far_one_rejected(self):
        t = _Tracker(pos_x=0.5, pos_y=0.5)
        close_blob = _pos(0.5 + MAX * 0.5, 0.5)
        far_blob = _pos(0.5 + MAX * 2.0, 0.5)
        GameStateManager._match_blobs_to_trackers([far_blob, close_blob], [t], MAX)
        assert t.current_state["pos_x"] == pytest.approx(close_blob.x)


# ---------------------------------------------------------------------------
# Identity-swap prevention
# ---------------------------------------------------------------------------

class TestIdentitySwap:
    def test_swap_prevented_when_players_cross(self):
        """
        Two trackers at (0.1, 0.5) and (0.9, 0.5) receive blobs that are
        very close to the *other* tracker — greedy NN would swap them;
        Hungarian keeps them correct.
        """
        # Trackers at left and right
        t_left  = _Tracker(pos_x=0.1, pos_y=0.5)
        t_right = _Tracker(pos_x=0.9, pos_y=0.5)
        # Blobs have moved slightly toward each other but are still closer to their own tracker
        blobs = [_pos(0.15, 0.5), _pos(0.85, 0.5)]
        GameStateManager._match_blobs_to_trackers(blobs, [t_left, t_right], MAX)
        # t_left should get (0.15,0.5) and t_right should get (0.85,0.5)
        assert t_left.current_state["pos_x"] == pytest.approx(0.15)
        assert t_right.current_state["pos_x"] == pytest.approx(0.85)

    def test_hungarian_chooses_globally_optimal(self):
        """
        Greedy NN on t1 would steal blob B (dist 0.05) even though t2 needs B.
        Hungarian assigns t1→A and t2→B (total cost 0.20) rather than
        t1→B and t2→A (total cost 0.30).
        """
        t1 = _Tracker(pos_x=0.5, pos_y=0.5)
        t2 = _Tracker(pos_x=0.6, pos_y=0.5)
        blob_A = _pos(0.65, 0.5)  # dist to t1=0.15, dist to t2=0.05
        blob_B = _pos(0.55, 0.5)  # dist to t1=0.05, dist to t2=0.05
        GameStateManager._match_blobs_to_trackers([blob_A, blob_B], [t1, t2], MAX)
        # t2 is closer to blob_A; Hungarian should assign t2→blob_A, t1→blob_B
        assert t1.current_state["pos_x"] == pytest.approx(0.55)
        assert t2.current_state["pos_x"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Velocity EMA
# ---------------------------------------------------------------------------

class TestVelocityEMA:
    def test_velocity_initialised_to_zero_for_unknown_tracker(self):
        t = _Tracker()  # pos_x=None → pass-2 slot
        GameStateManager._match_blobs_to_trackers([_pos(0.3, 0.4)], [t], MAX)
        assert t.current_state["pos_vx"] == pytest.approx(0.0)
        assert t.current_state["pos_vy"] == pytest.approx(0.0)

    def test_velocity_updated_on_match(self):
        t = _Tracker(pos_x=0.4, pos_y=0.4, vx=0.0, vy=0.0)
        # Move +0.05 in x, +0.05 in y
        GameStateManager._match_blobs_to_trackers([_pos(0.45, 0.45)], [t], MAX)
        assert t.current_state["pos_vx"] == pytest.approx(0.4 * 0.05, abs=1e-6)
        assert t.current_state["pos_vy"] == pytest.approx(0.4 * 0.05, abs=1e-6)

    def test_velocity_ema_accumulates_over_frames(self):
        """After a steady 0.05/frame displacement, velocity should converge toward 0.05."""
        t = _Tracker(pos_x=0.0, pos_y=0.0, vx=0.0, vy=0.0)
        for step in range(1, 20):
            x = step * 0.05
            blobs = [_pos(min(x, 1.0), 0.0)]
            GameStateManager._match_blobs_to_trackers(blobs, [t], max_dist=1.0)
        # After many frames of constant 0.05/step, EMA → 0.05
        assert t.current_state["pos_vx"] == pytest.approx(0.05, abs=0.005)

    def test_velocity_prediction_enables_fast_mover_match(self):
        """
        A tracker moving at +0.15/frame would miss the blob without velocity
        prediction (distance 0.15 > threshold may still be < MAX with prediction).
        With prediction, the predicted position is at the blob.
        """
        # Tracker at 0.3, moving +0.15/frame consistently
        t = _Tracker(pos_x=0.3, pos_y=0.5, vx=0.15, vy=0.0)
        # Blob lands exactly at predicted position: 0.3 + 0.15 = 0.45
        blobs = [_pos(0.45, 0.5)]
        GameStateManager._match_blobs_to_trackers(blobs, [t], MAX)
        assert t.current_state["pos_x"] == pytest.approx(0.45)

    def test_velocity_decays_on_miss(self):
        t = _Tracker(pos_x=0.5, pos_y=0.5, vx=0.10, vy=0.08)
        # No blobs → tracker misses, velocity should halve
        GameStateManager._match_blobs_to_trackers([], [t], MAX)
        assert t.current_state["pos_vx"] == pytest.approx(0.05)
        assert t.current_state["pos_vy"] == pytest.approx(0.04)
        assert t.current_state["pos_x"] is None  # cleared on miss


# ---------------------------------------------------------------------------
# Pass-2: unknown trackers seeded from leftover blobs
# ---------------------------------------------------------------------------

class TestPass2Seeding:
    def test_unknown_tracker_gets_leftover_blob(self):
        known = _Tracker(pos_x=0.5, pos_y=0.5)
        unknown = _Tracker()  # pos_x=None
        blobs = [_pos(0.5, 0.5), _pos(0.2, 0.3)]
        GameStateManager._match_blobs_to_trackers(blobs, [known, unknown], MAX)
        assert known.current_state["pos_x"] == pytest.approx(0.5)
        assert unknown.current_state["pos_x"] == pytest.approx(0.2)
        assert unknown.current_state["pos_vx"] == pytest.approx(0.0)

    def test_unknown_tracker_stays_none_when_no_leftover(self):
        known = _Tracker(pos_x=0.5, pos_y=0.5)
        unknown = _Tracker()
        blobs = [_pos(0.5, 0.5)]  # only one blob → consumed by known
        GameStateManager._match_blobs_to_trackers(blobs, [known, unknown], MAX)
        assert unknown.current_state["pos_x"] is None

    def test_multiple_unknown_trackers_seeded_in_order(self):
        u1 = _Tracker()
        u2 = _Tracker()
        blobs = [_pos(0.1, 0.1), _pos(0.9, 0.9)]
        GameStateManager._match_blobs_to_trackers(blobs, [u1, u2], MAX)
        # Both should be seeded (order matches blob list order)
        assert u1.current_state["pos_x"] is not None
        assert u2.current_state["pos_x"] is not None

    def test_all_unknown_no_known(self):
        """When no trackers have prior positions, all are unknown and seeded from blobs."""
        trackers = [_Tracker() for _ in range(5)]
        blobs = [_pos(i * 0.2, 0.5) for i in range(5)]
        GameStateManager._match_blobs_to_trackers(blobs, trackers, MAX)
        positions = {t.current_state["pos_x"] for t in trackers}
        assert None not in positions
        assert len(positions) == 5


# ---------------------------------------------------------------------------
# TRACK_MAX_DIST constant sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_track_max_dist_is_float(self):
        assert isinstance(MinimapDetector.TRACK_MAX_DIST, float)

    def test_track_max_dist_reasonable_range(self):
        # Should be between 0.05 (too tight) and 0.50 (too loose)
        assert 0.05 < MinimapDetector.TRACK_MAX_DIST < 0.50
