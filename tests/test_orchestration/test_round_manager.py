"""Unit tests for RoundManager."""

from __future__ import annotations
import pytest

from valoscribe.orchestration.round_manager import RoundManager


class TestRoundManager:
    """Tests for RoundManager class."""

    @pytest.fixture
    def vlr_metadata(self):
        """Create sample VLR metadata."""
        return {
            "map_number": 1,
            "map_name": "Corrode",
            "vod_url": "https://youtu.be/example",
            "teams": [
                {
                    "name": "NRG",
                    "starting_side": "defense",
                    "players": [
                        {"name": "brawk", "agent": "sova"},
                        {"name": "mada", "agent": "waylay"},
                        {"name": "Ethan", "agent": "kayo"},
                        {"name": "s0m", "agent": "omen"},
                        {"name": "skuba", "agent": "viper"},
                    ],
                },
                {
                    "name": "FNATIC",
                    "starting_side": "attack",
                    "players": [
                        {"name": "kaajak", "agent": "yoru"},
                        {"name": "crashies", "agent": "fade"},
                        {"name": "Chronicle", "agent": "viper"},
                        {"name": "Boaster", "agent": "astra"},
                        {"name": "Alfajer", "agent": "vyse"},
                    ],
                },
            ],
        }

    @pytest.fixture
    def round_manager(self, vlr_metadata):
        """Create RoundManager instance."""
        return RoundManager(vlr_metadata)

    def test_init(self, round_manager):
        """Test RoundManager initialization."""
        assert round_manager.team_names == ["NRG", "FNATIC"]
        assert round_manager.starting_sides == {"team1": "defense", "team2": "attack"}
        assert round_manager.current_round == 0
        assert round_manager.current_score == {"team1": 0, "team2": 0}
        assert round_manager.round_history == []

    def test_init_invalid_side(self, vlr_metadata):
        """Test initialization with invalid starting side."""
        vlr_metadata["teams"][0]["starting_side"] = "invalid"
        with pytest.raises(ValueError, match="Invalid starting side"):
            RoundManager(vlr_metadata)

    def test_init_same_sides(self, vlr_metadata):
        """Test initialization with both teams on same side."""
        vlr_metadata["teams"][0]["starting_side"] = "attack"
        vlr_metadata["teams"][1]["starting_side"] = "attack"
        with pytest.raises(ValueError, match="cannot start on the same side"):
            RoundManager(vlr_metadata)

    def test_start_new_round(self, round_manager):
        """Test starting a new round."""
        round_num = round_manager.start_new_round(timestamp=0.0)

        assert round_num == 1
        assert round_manager.current_round == 1
        assert len(round_manager.round_history) == 1
        assert round_manager.round_history[0] == {
            "round_number": 1,
            "start_time": 0.0,
            "end_time": None,
            "winner": None,
        }

    def test_start_multiple_rounds(self, round_manager):
        """Test starting multiple rounds."""
        round_manager.start_new_round(timestamp=0.0)
        round_manager.start_new_round(timestamp=120.0)
        round_manager.start_new_round(timestamp=240.0)

        assert round_manager.current_round == 3
        assert len(round_manager.round_history) == 3

    def test_infer_round_end_team1_wins(self, round_manager):
        """Test inferring round end when team1 wins."""
        round_manager.start_new_round(timestamp=0.0)

        winner = round_manager.infer_round_end(
            new_score={"team1": 1, "team2": 0}, timestamp=90.0
        )

        assert winner == "team1"
        assert round_manager.current_score == {"team1": 1, "team2": 0}
        assert round_manager.round_history[0]["end_time"] == 90.0
        assert round_manager.round_history[0]["winner"] == "team1"

    def test_infer_round_end_team2_wins(self, round_manager):
        """Test inferring round end when team2 wins."""
        round_manager.start_new_round(timestamp=0.0)

        winner = round_manager.infer_round_end(
            new_score={"team1": 0, "team2": 1}, timestamp=85.0
        )

        assert winner == "team2"
        assert round_manager.current_score == {"team1": 0, "team2": 1}
        assert round_manager.round_history[0]["winner"] == "team2"

    def test_infer_round_end_no_change(self, round_manager):
        """Test inferring round end with no score change."""
        round_manager.start_new_round(timestamp=0.0)

        winner = round_manager.infer_round_end(
            new_score={"team1": 0, "team2": 0}, timestamp=50.0
        )

        assert winner is None
        assert round_manager.current_score == {"team1": 0, "team2": 0}
        assert round_manager.round_history[0]["end_time"] is None
        assert round_manager.round_history[0]["winner"] is None

    def test_get_current_sides_first_half(self, round_manager):
        """Test getting sides during first half (rounds 1-12)."""
        # Round 1
        round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "defense", "team2": "attack"}

        # Round 6
        for _ in range(5):
            round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "defense", "team2": "attack"}

        # Round 12
        for _ in range(6):
            round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "defense", "team2": "attack"}

    def test_get_current_sides_second_half(self, round_manager):
        """Test getting sides during second half (rounds 13-24) - sides swap."""
        # Advance to round 13
        for _ in range(13):
            round_manager.start_new_round(timestamp=0.0)

        sides = round_manager.get_current_sides()
        # Sides should be swapped
        assert sides == {"team1": "attack", "team2": "defense"}

        # Round 24
        for _ in range(11):
            round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "attack", "team2": "defense"}

    def test_get_current_sides_overtime(self, round_manager):
        """Test getting sides during overtime (rounds 25+) - swap every round."""
        # Advance to overtime
        for _ in range(25):
            round_manager.start_new_round(timestamp=0.0)

        # Round 25 (odd overtime round): original sides
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "defense", "team2": "attack"}

        # Round 26 (even overtime round): swapped sides
        round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "attack", "team2": "defense"}

        # Round 27 (odd overtime round): original sides
        round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "defense", "team2": "attack"}

        # Round 28 (even overtime round): swapped sides
        round_manager.start_new_round(timestamp=0.0)
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "attack", "team2": "defense"}

    def test_get_current_sides_before_start(self, round_manager):
        """Test getting sides before game starts (round 0)."""
        sides = round_manager.get_current_sides()
        assert sides == {"team1": "defense", "team2": "attack"}

    def test_get_player_side_team1(self, round_manager):
        """Test getting side for team1 players (indices 0-4)."""
        round_manager.start_new_round(timestamp=0.0)

        for player_idx in range(5):
            side = round_manager.get_player_side(player_idx)
            assert side == "defense"  # team1 starts on defense

    def test_get_player_side_team2(self, round_manager):
        """Test getting side for team2 players (indices 5-9)."""
        round_manager.start_new_round(timestamp=0.0)

        for player_idx in range(5, 10):
            side = round_manager.get_player_side(player_idx)
            assert side == "attack"  # team2 starts on attack

    def test_get_player_side_after_halftime(self, round_manager):
        """Test getting player side after halftime (sides swap)."""
        # Advance to round 13
        for _ in range(13):
            round_manager.start_new_round(timestamp=0.0)

        # Team1 players should now be on attack
        for player_idx in range(5):
            side = round_manager.get_player_side(player_idx)
            assert side == "attack"

        # Team2 players should now be on defense
        for player_idx in range(5, 10):
            side = round_manager.get_player_side(player_idx)
            assert side == "defense"

    def test_get_player_side_invalid_index(self, round_manager):
        """Test getting player side with invalid index."""
        with pytest.raises(ValueError, match="Invalid player index"):
            round_manager.get_player_side(-1)

        with pytest.raises(ValueError, match="Invalid player index"):
            round_manager.get_player_side(10)

    def test_get_team_for_player(self, round_manager):
        """Test getting team name for players."""
        for player_idx in range(5):
            team = round_manager.get_team_for_player(player_idx)
            assert team == "NRG"

        for player_idx in range(5, 10):
            team = round_manager.get_team_for_player(player_idx)
            assert team == "FNATIC"

    def test_get_team_for_player_invalid_index(self, round_manager):
        """Test getting team with invalid index."""
        with pytest.raises(ValueError, match="Invalid player index"):
            round_manager.get_team_for_player(10)

    def test_repr(self, round_manager):
        """Test string representation."""
        round_manager.start_new_round(timestamp=0.0)
        round_manager.infer_round_end(
            new_score={"team1": 1, "team2": 0}, timestamp=90.0
        )

        repr_str = repr(round_manager)
        assert "round=1" in repr_str
        assert "score=1-0" in repr_str
        assert "NRG vs FNATIC" in repr_str
