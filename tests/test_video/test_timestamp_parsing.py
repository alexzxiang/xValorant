"""Tests for YouTube timestamp parsing."""

import pytest
from valoscribe.video.youtube import _parse_timestamp


class TestTimestampParsing:
    """Test _parse_timestamp helper function."""

    def test_simple_seconds(self):
        """Test parsing simple second values."""
        assert _parse_timestamp("1234") == 1234.0
        assert _parse_timestamp("0") == 0.0
        assert _parse_timestamp("9999") == 9999.0

    def test_seconds_with_suffix(self):
        """Test parsing seconds with 's' suffix."""
        assert _parse_timestamp("1234s") == 1234.0
        assert _parse_timestamp("0s") == 0.0
        assert _parse_timestamp("9999s") == 9999.0

    def test_minutes_seconds_format(self):
        """Test parsing 'MMmSSs' format."""
        assert _parse_timestamp("20m34s") == 1234.0
        assert _parse_timestamp("1m0s") == 60.0
        assert _parse_timestamp("0m30s") == 30.0
        assert _parse_timestamp("59m59s") == 3599.0

    def test_hours_minutes_seconds_format(self):
        """Test parsing 'HHhMMmSSs' format."""
        assert _parse_timestamp("1h20m34s") == 4834.0
        assert _parse_timestamp("2h0m0s") == 7200.0
        assert _parse_timestamp("0h5m30s") == 330.0
        assert _parse_timestamp("10h30m45s") == 37845.0

    def test_colon_format_mm_ss(self):
        """Test parsing 'MM:SS' format."""
        assert _parse_timestamp("20:34") == 1234.0
        assert _parse_timestamp("1:00") == 60.0
        assert _parse_timestamp("0:30") == 30.0
        assert _parse_timestamp("59:59") == 3599.0

    def test_colon_format_hh_mm_ss(self):
        """Test parsing 'HH:MM:SS' format."""
        assert _parse_timestamp("1:20:34") == 4834.0
        assert _parse_timestamp("2:00:00") == 7200.0
        assert _parse_timestamp("0:05:30") == 330.0
        assert _parse_timestamp("10:30:45") == 37845.0

    def test_partial_youtube_format(self):
        """Test parsing partial YouTube format (missing components)."""
        # Just minutes
        assert _parse_timestamp("20m") == 1200.0
        # Just hours
        assert _parse_timestamp("2h") == 7200.0
        # Hours and minutes
        assert _parse_timestamp("1h30m") == 5400.0
        # Minutes and seconds
        assert _parse_timestamp("5m30s") == 330.0

    def test_invalid_formats(self):
        """Test that invalid formats return None."""
        assert _parse_timestamp("invalid") is None
        assert _parse_timestamp("") is None
        assert _parse_timestamp("abc123") is None
        assert _parse_timestamp("12.34.56") is None

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        assert _parse_timestamp("  1234  ") == 1234.0
        assert _parse_timestamp("  1234s  ") == 1234.0
        assert _parse_timestamp("  1h30m  ") == 5400.0

    def test_real_world_examples(self):
        """Test real YouTube URL timestamp values."""
        # Common YouTube timestamp formats
        assert _parse_timestamp("3600") == 3600.0  # 1 hour
        assert _parse_timestamp("3600s") == 3600.0
        assert _parse_timestamp("1h") == 3600.0
        assert _parse_timestamp("1h0m0s") == 3600.0
        assert _parse_timestamp("1:00:00") == 3600.0

        # Realistic match timestamps
        assert _parse_timestamp("5400") == 5400.0  # 1:30:00
        assert _parse_timestamp("1h30m") == 5400.0
        assert _parse_timestamp("90m") == 5400.0
        assert _parse_timestamp("1:30:00") == 5400.0
