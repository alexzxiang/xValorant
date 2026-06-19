"""Round and score tracking for game state orchestration."""

from __future__ import annotations
from typing import Optional

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class RoundManager:
    """
    Tracks rounds, scores, and side swaps.

    Responsibilities:
    - Track current round number (1-24+)
    - Track team scores
    - Handle side swaps (halftime at round 13, overtime swaps)
    - Maintain round history (start/end times, winners)
    - Determine current sides based on round number
    """

    def __init__(self, vlr_metadata: dict):
        """
        Initialize round manager with VLR metadata.

        Args:
            vlr_metadata: VLR scraped map data with structure:
                {
                    "map_number": 1,
                    "map_name": "Corrode",
                    "vod_url": "...",
                    "teams": [
                        {
                            "name": "NRG",
                            "starting_side": "defense",
                            "players": [...]
                        },
                        {
                            "name": "FNATIC",
                            "starting_side": "attack",
                            "players": [...]
                        }
                    ]
                }
        """
        # Extract team names (team1 = index 0, team2 = index 1)
        self.team_names = [team["name"] for team in vlr_metadata["teams"]]

        # Extract starting sides
        self.starting_sides = {
            "team1": vlr_metadata["teams"][0]["starting_side"],
            "team2": vlr_metadata["teams"][1]["starting_side"],
        }

        # Validate starting sides
        if self.starting_sides["team1"] not in ["attack", "defense"]:
            raise ValueError(
                f"Invalid starting side for team1: {self.starting_sides['team1']}"
            )
        if self.starting_sides["team2"] not in ["attack", "defense"]:
            raise ValueError(
                f"Invalid starting side for team2: {self.starting_sides['team2']}"
            )

        # Ensure teams start on opposite sides
        if self.starting_sides["team1"] == self.starting_sides["team2"]:
            raise ValueError(
                f"Teams cannot start on the same side: "
                f"team1={self.starting_sides['team1']}, team2={self.starting_sides['team2']}"
            )

        # Initialize round tracking
        self.current_round = 0  # Will be incremented to 1 on first round start
        self.current_score = {"team1": 0, "team2": 0}

        # Round history: list of dicts with round_number, start_time, end_time, winner
        self.round_history: list[dict] = []

        log.info(
            f"RoundManager initialized: {self.team_names[0]} vs {self.team_names[1]}, "
            f"starting sides: {self.starting_sides}"
        )

    def start_new_round(self, timestamp: float) -> int:
        """
        Start a new round.

        Args:
            timestamp: Timestamp when round started (seconds)

        Returns:
            New round number
        """
        self.current_round += 1

        # Add to history
        self.round_history.append({
            "round_number": self.current_round,
            "start_time": timestamp,
            "end_time": None,
            "winner": None,
        })

        log.info(f"Round {self.current_round} started at {timestamp:.2f}s")
        return self.current_round

    def infer_round_end(
        self, new_score: dict[str, int], timestamp: float
    ) -> Optional[str]:
        """
        Infer round end from score change and update history.

        Args:
            new_score: New score dict {"team1": int, "team2": int}
            timestamp: Timestamp when score changed (seconds)

        Returns:
            Winner team key ("team1" or "team2") or None if no change
        """
        # Determine which team's score increased
        winner = None

        if new_score["team1"] > self.current_score["team1"]:
            winner = "team1"
        elif new_score["team2"] > self.current_score["team2"]:
            winner = "team2"

        if winner:
            # Update score
            self.current_score = new_score.copy()

            # Update round history
            if self.round_history and self.round_history[-1]["end_time"] is None:
                self.round_history[-1]["end_time"] = timestamp
                self.round_history[-1]["winner"] = winner

                log.info(
                    f"Round {self.current_round} ended at {timestamp:.2f}s, "
                    f"winner: {self.team_names[0 if winner == 'team1' else 1]} ({winner})"
                )

        return winner

    def get_current_sides(self) -> dict[str, str]:
        """
        Get current team sides based on round number.

        Side swap logic:
        - Rounds 1-12: original sides
        - Rounds 13-24: swapped sides
        - Overtime (25+): swap every round, start from original

        Returns:
            Dictionary: {"team1": "attack"|"defense", "team2": "attack"|"defense"}
        """
        if self.current_round == 0:
            # Game hasn't started yet, return starting sides
            return self.starting_sides.copy()

        if self.current_round <= 12:
            # First half: original sides
            return self.starting_sides.copy()
        elif self.current_round <= 24:
            # Second half: swapped sides
            return {
                "team1": self._swap_side(self.starting_sides["team1"]),
                "team2": self._swap_side(self.starting_sides["team2"]),
            }
        else:
            # Overtime: swap every round, start from original sides
            # Round 25 = original, 26 = swapped, 27 = original, etc.
            overtime_round = self.current_round - 24  # 1, 2, 3, ...

            if overtime_round % 2 == 1:
                # Odd overtime rounds: original sides
                return self.starting_sides.copy()
            else:
                # Even overtime rounds: swapped sides
                return {
                    "team1": self._swap_side(self.starting_sides["team1"]),
                    "team2": self._swap_side(self.starting_sides["team2"]),
                }

    def get_player_side(self, player_index: int) -> str:
        """
        Get current side for a player.

        Args:
            player_index: Player index (0-9, where 0-4 is team1, 5-9 is team2)

        Returns:
            Current side: "attack" or "defense"
        """
        if not 0 <= player_index <= 9:
            raise ValueError(f"Invalid player index: {player_index}, must be 0-9")

        # Determine which team the player is on
        team_key = "team1" if player_index < 5 else "team2"

        # Get current sides
        current_sides = self.get_current_sides()

        return current_sides[team_key]

    def get_team_for_player(self, player_index: int) -> str:
        """
        Get team name for a player.

        Args:
            player_index: Player index (0-9)

        Returns:
            Team name
        """
        if not 0 <= player_index <= 9:
            raise ValueError(f"Invalid player index: {player_index}, must be 0-9")

        team_idx = 0 if player_index < 5 else 1
        return self.team_names[team_idx]

    @staticmethod
    def _swap_side(side: str) -> str:
        """Swap attack <-> defense."""
        return "defense" if side == "attack" else "attack"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RoundManager(round={self.current_round}, "
            f"score={self.current_score['team1']}-{self.current_score['team2']}, "
            f"teams={self.team_names[0]} vs {self.team_names[1]})"
        )
