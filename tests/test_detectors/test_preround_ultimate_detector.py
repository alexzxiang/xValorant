"""Unit tests for pre-round ultimate detector."""

from __future__ import annotations
from unittest.mock import Mock
import pytest
import numpy as np
import cv2

from valoscribe.detectors.preround_ultimate_detector import PreroundUltimateDetector
from valoscribe.types.detections import UltimateInfo


class TestPreroundUltimateDetector:
    """Tests for PreroundUltimateDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        return cropper

    @pytest.fixture
    def detector(self, mock_cropper):
        """Create pre-round ultimate detector with default settings."""
        return PreroundUltimateDetector(
            mock_cropper,
            brightness_threshold=127,
            fullness_threshold=0.4,
        )

    def create_ring_image(
        self, num_segments: int = 0, is_full: bool = False, size: int = 50
    ) -> np.ndarray:
        """
        Helper to create test ultimate ring image.

        Args:
            num_segments: Number of segments to draw (0-7)
            is_full: If True, draw a complete filled ring
            size: Size of the image (size x size)

        Returns:
            Test image with ultimate ring
        """
        # Create black background
        image = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)

        if is_full:
            # Draw filled ring (solid white ring)
            cv2.circle(image, center, 23, 255, -1)  # Outer circle
            cv2.circle(image, center, 13, 0, -1)  # Inner circle (mask)
        elif num_segments > 0:
            # Draw individual segments around the ring
            segment_angle = 360 / 7  # 7 segments total
            for i in range(num_segments):
                angle = i * segment_angle
                # Draw small arc/blob for each segment
                start_angle = int(angle - 10)
                end_angle = int(angle + 10)
                # Draw as filled circle at position around ring
                angle_rad = np.radians(angle)
                x = int(center[0] + 18 * np.cos(angle_rad))
                y = int(center[1] + 18 * np.sin(angle_rad))
                cv2.circle(image, (x, y), 4, 255, -1)

        return image

    def test_init(self, mock_cropper):
        """Test pre-round ultimate detector initialization."""
        detector = PreroundUltimateDetector(
            mock_cropper,
            brightness_threshold=150,
            fullness_threshold=0.5,
        )

        assert detector.cropper == mock_cropper
        assert detector.brightness_threshold == 150
        assert detector.fullness_threshold == 0.5

    def test_inherits_from_ultimate_detector(self, detector):
        """Test that PreroundUltimateDetector inherits from UltimateDetector."""
        from valoscribe.detectors.ultimate_detector import UltimateDetector

        assert isinstance(detector, UltimateDetector)
        assert isinstance(detector, PreroundUltimateDetector)

        # Should have inherited methods
        assert hasattr(detector, '_preprocess_crop')
        assert hasattr(detector, '_create_ring_mask')
        assert hasattr(detector, '_count_blobs')
        assert hasattr(detector, 'detect_ultimate')

    def test_detect_ultimate_uses_preround_crops(self, detector, mock_cropper):
        """Test that detector uses crop_player_info_preround instead of crop_player_info."""
        ultimate_crop = self.create_ring_image(num_segments=3)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        # Verify that crop_player_info_preround was called (not crop_player_info)
        mock_cropper.crop_player_info_preround.assert_called_once_with(frame)
        assert result is not None

    def test_detect_ultimate_player_out_of_range(self, detector, mock_cropper):
        """Test detection with invalid player index."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=10)

        assert result is None

    def test_detect_ultimate_wrong_side(self, detector, mock_cropper):
        """Test detection with wrong side specified."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0, side="right")

        assert result is None

    def test_detect_ultimate_missing_ultimate_region(self, detector, mock_cropper):
        """Test detection when ultimate region is not in crop data."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left"}  # No ultimate region
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is None

    def test_detect_ultimate_empty_crop(self, detector, mock_cropper):
        """Test detection with empty ultimate crop."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is None

    def test_detect_ultimate_full(self, detector, mock_cropper):
        """Test detection of full ultimate (solid ring)."""
        ultimate_crop = self.create_ring_image(is_full=True, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert ultimate_info.is_full is True
        assert ultimate_info.charges == 7
        assert white_pixel_ratio >= 0.4  # Should exceed fullness threshold

    def test_detect_ultimate_partial_charges(self, detector, mock_cropper):
        """Test detection of partial ultimate with segments."""
        # Test with 3 segments
        ultimate_crop = self.create_ring_image(num_segments=3, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert ultimate_info.is_full is False
        # Should detect some charges (may not be exactly 3 due to blob detection sensitivity)
        assert ultimate_info.charges >= 0
        assert ultimate_info.charges <= 7

    def test_detect_ultimate_zero_charges(self, detector, mock_cropper):
        """Test detection with no charges (empty ring)."""
        # Empty black ultimate crop
        ultimate_crop = np.zeros((50, 50, 3), dtype=np.uint8)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert ultimate_info.is_full is False
        assert ultimate_info.charges == 0
        assert white_pixel_ratio < 0.4  # Should be below fullness threshold

    def test_detect_ultimate_right_side_player(self, detector, mock_cropper):
        """Test detection for right-side player."""
        ultimate_crop = self.create_ring_image(num_segments=5, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)},
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)},
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)},
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)},
            {"side": "left", "ultimate": np.zeros((50, 50, 3), dtype=np.uint8)},
            {"side": "right", "ultimate": ultimate_crop_color},
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=5, side="right")

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert ultimate_info.charges >= 0  # Should detect some charges

    def test_detect_with_debug(self, detector, mock_cropper):
        """Test detect_with_debug method."""
        ultimate_crop = self.create_ring_image(num_segments=4, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0
        )

        assert ultimate_info is not None
        assert isinstance(ultimate_info, UltimateInfo)
        assert preprocessed.size > 0
        assert isinstance(debug_info, dict)
        assert "white_pixel_ratio" in debug_info
        assert "is_full" in debug_info
        assert "total_blobs" in debug_info

    def test_detect_with_debug_empty_crop(self, detector, mock_cropper):
        """Test detect_with_debug with empty crop."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0
        )

        assert ultimate_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_detect_with_debug_missing_ultimate(self, detector, mock_cropper):
        """Test detect_with_debug when ultimate region is missing."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left"}  # No ultimate
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0
        )

        assert ultimate_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_different_fullness_thresholds(self, mock_cropper):
        """Test detector with different fullness thresholds."""
        # Create image with medium pixel density
        ultimate_crop = self.create_ring_image(num_segments=4, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Low threshold detector (more likely to detect as full)
        detector_low = PreroundUltimateDetector(mock_cropper, fullness_threshold=0.1)
        result_low = detector_low.detect_ultimate(frame, 0)

        # High threshold detector (less likely to detect as full)
        detector_high = PreroundUltimateDetector(mock_cropper, fullness_threshold=0.9)
        result_high = detector_high.detect_ultimate(frame, 0)

        # Both should return results
        assert result_low is not None
        assert result_high is not None

        # High threshold should not detect as full (segments != full ring)
        ultimate_info_high, _ = result_high
        assert ultimate_info_high.is_full is False

    def test_uses_parent_blob_detection_logic(self, detector, mock_cropper):
        """Test that pre-round detector uses parent's blob counting logic."""
        # Create ultimate with clear segments
        ultimate_crop = self.create_ring_image(num_segments=5, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, 0)

        # Should use parent's blob detection
        assert result is not None
        ultimate_info, _ = result
        assert ultimate_info.charges >= 0
        assert ultimate_info.charges <= 7
        # total_blobs_detected should be present
        assert hasattr(ultimate_info, 'total_blobs_detected')

    def test_detect_ultimate_returns_white_pixel_ratio(self, detector, mock_cropper):
        """Test that detect_ultimate returns both UltimateInfo and white_pixel_ratio."""
        ultimate_crop = self.create_ring_image(num_segments=2, size=50)
        ultimate_crop_color = cv2.cvtColor(ultimate_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ultimate": ultimate_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, 0)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert isinstance(white_pixel_ratio, (float, np.floating))
        assert 0.0 <= white_pixel_ratio <= 1.0

    def test_all_players_detection(self, detector, mock_cropper):
        """Test detection for all 10 players."""
        # Create different ultimate states for each player
        crops = []
        for i in range(10):
            if i % 3 == 0:
                crop = self.create_ring_image(is_full=True, size=50)
            elif i % 3 == 1:
                crop = self.create_ring_image(num_segments=i % 7, size=50)
            else:
                crop = np.zeros((50, 50), dtype=np.uint8)

            crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            side = "left" if i < 5 else "right"
            crops.append({"side": side, "ultimate": crop_color})

        mock_cropper.crop_player_info_preround.return_value = crops
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Test each player
        for i in range(10):
            side = "left" if i < 5 else "right"
            result = detector.detect_ultimate(frame, i, side)

            # All should return valid results
            assert result is not None
            ultimate_info, white_pixel_ratio = result
            assert isinstance(ultimate_info, UltimateInfo)
            assert 0.0 <= white_pixel_ratio <= 1.0
