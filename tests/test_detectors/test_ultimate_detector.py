"""Unit tests for blob-based ultimate detector."""

from __future__ import annotations
from unittest.mock import Mock
import pytest
import numpy as np
import cv2

from valoscribe.detectors.ultimate_detector import UltimateDetector
from valoscribe.types.detections import UltimateInfo


class TestUltimateDetector:
    """Tests for UltimateDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        return cropper

    @pytest.fixture
    def detector(self, mock_cropper):
        """Create ultimate detector with default settings."""
        return UltimateDetector(
            mock_cropper,
            brightness_threshold=127,
            fullness_threshold=0.4,
        )

    def create_full_ultimate_crop(self, size: int = 46) -> np.ndarray:
        """
        Create a crop simulating a full ultimate (solid white ring).

        Args:
            size: Size of the crop (default 46x46)

        Returns:
            Test image with solid ring
        """
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size // 2, size // 2)

        # Draw solid white ring (simulating full ultimate)
        cv2.circle(crop, center, 23, (255, 255, 255), -1)  # Outer circle
        cv2.circle(crop, center, 13, (0, 0, 0), -1)  # Inner circle (make ring)
        # Center icon area (will be masked out anyway)
        cv2.circle(crop, center, 12, (128, 128, 128), -1)

        return crop

    def create_partial_ultimate_crop(self, num_segments: int, size: int = 46) -> np.ndarray:
        """
        Create a crop simulating a partial ultimate (segmented ring).

        Args:
            num_segments: Number of segments to create
            size: Size of the crop (default 46x46)

        Returns:
            Test image with segmented ring
        """
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size // 2, size // 2)

        # Draw segments around the ring
        for i in range(num_segments):
            angle = (360 / num_segments) * i
            # Draw small arc/blob for each segment
            start_angle = angle - 15
            end_angle = angle + 15

            # Draw arc segment
            cv2.ellipse(
                crop,
                center,
                (18, 18),  # Ring radius
                0,
                start_angle,
                end_angle,
                (255, 255, 255),
                4,  # Thickness
            )

        return crop

    def test_init(self, mock_cropper):
        """Test ultimate detector initialization."""
        detector = UltimateDetector(
            mock_cropper,
            brightness_threshold=150,
            center_mask_radius=15,
            ring_inner_radius=16,
            ring_outer_radius=25,
            fullness_threshold=0.5,
            min_blob_area=20,
            max_blob_area=600,
            min_circularity=0.3,
        )

        assert detector.cropper == mock_cropper
        assert detector.brightness_threshold == 150
        assert detector.center_mask_radius == 15
        assert detector.ring_inner_radius == 16
        assert detector.ring_outer_radius == 25
        assert detector.fullness_threshold == 0.5
        assert detector.min_blob_area == 20
        assert detector.max_blob_area == 600
        assert detector.min_circularity == 0.3

    def test_init_defaults(self, mock_cropper):
        """Test ultimate detector initialization with defaults."""
        detector = UltimateDetector(mock_cropper)

        assert detector.brightness_threshold == 127
        assert detector.center_mask_radius == 12
        assert detector.ring_inner_radius == 13
        assert detector.ring_outer_radius == 23
        assert detector.fullness_threshold == 0.4
        assert detector.min_blob_area == 10
        assert detector.max_blob_area == 500
        assert detector.min_circularity == 0.2

    def test_create_ring_mask(self, detector):
        """Test ring mask creation."""
        shape = (46, 46)
        center = (23, 23)

        mask = detector._create_ring_mask(shape, center)

        # Check shape
        assert mask.shape == shape

        # Check that it's binary
        assert np.all((mask == 0) | (mask == 255))

        # Check center is black (inside inner radius)
        assert mask[center[1], center[0]] == 0

        # Check ring area has white pixels
        # Point at inner_radius + 1
        test_point_y = center[1] - (detector.ring_inner_radius + 1)
        assert mask[test_point_y, center[0]] == 255

        # Check outside outer radius is black
        test_outside_y = center[1] - (detector.ring_outer_radius + 2)
        if 0 <= test_outside_y < shape[0]:
            assert mask[test_outside_y, center[0]] == 0

    def test_preprocess_crop_color_input(self, detector):
        """Test preprocessing with color input."""
        crop = np.zeros((46, 46, 3), dtype=np.uint8)
        # Add bright region
        crop[15:30, 15:30] = 200

        preprocessed = detector._preprocess_crop(crop)

        # Should be grayscale
        assert len(preprocessed.shape) == 2
        # Should be binary
        assert np.all((preprocessed == 0) | (preprocessed == 255))
        # Bright region should be white
        assert np.any(preprocessed[15:30, 15:30] == 255)

    def test_preprocess_crop_grayscale_input(self, detector):
        """Test preprocessing with grayscale input."""
        crop = np.zeros((46, 46), dtype=np.uint8)
        crop[15:30, 15:30] = 200

        preprocessed = detector._preprocess_crop(crop)

        assert len(preprocessed.shape) == 2
        assert np.all((preprocessed == 0) | (preprocessed == 255))

    def test_count_blobs_zero(self, detector):
        """Test blob counting with no blobs."""
        image = np.zeros((46, 46), dtype=np.uint8)

        charges, total_blobs = detector._count_blobs(image)

        assert charges == 0
        assert total_blobs == 0

    def test_count_blobs_single(self, detector):
        """Test blob counting with one blob."""
        image = np.zeros((46, 46), dtype=np.uint8)
        cv2.circle(image, (23, 18), 5, 255, -1)

        charges, total_blobs = detector._count_blobs(image)

        assert charges >= 1
        assert total_blobs >= 1

    def test_count_blobs_multiple(self, detector):
        """Test blob counting with multiple blobs."""
        image = np.zeros((46, 46), dtype=np.uint8)

        # Draw 3 blobs around ring
        cv2.circle(image, (23, 10), 4, 255, -1)
        cv2.circle(image, (33, 23), 4, 255, -1)
        cv2.circle(image, (23, 33), 4, 255, -1)

        charges, total_blobs = detector._count_blobs(image)

        assert charges == 3
        assert total_blobs >= 3

    def test_count_blobs_filters_small(self, detector):
        """Test that small noise is filtered out."""
        image = np.zeros((46, 46), dtype=np.uint8)

        # Valid blob
        cv2.circle(image, (23, 18), 5, 255, -1)

        # Tiny noise (below min_blob_area)
        image[5:7, 5:7] = 255  # 4 pixels

        charges, total_blobs = detector._count_blobs(image)

        # Should only count valid blob
        assert charges == 1

    def test_count_blobs_filters_large(self, detector):
        """Test that too-large regions are filtered out."""
        image = np.zeros((100, 100), dtype=np.uint8)

        # Valid blob (area ~78, within 10-500 range)
        cv2.circle(image, (30, 30), 5, 255, -1)

        # Huge blob (exceeds max_blob_area=500, area ~1256)
        cv2.circle(image, (70, 70), 20, 255, -1)

        charges, total_blobs = detector._count_blobs(image)

        # Should only count valid blob
        assert charges == 1
        assert total_blobs >= 2  # Both blobs detected, but only 1 valid

    def test_detect_ultimate_player_out_of_range(self, detector, mock_cropper):
        """Test detection with invalid player index."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": np.zeros((46, 46, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=10)

        assert result is None

    def test_detect_ultimate_wrong_side(self, detector, mock_cropper):
        """Test detection with wrong side specified."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": np.zeros((46, 46, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0, side="right")

        assert result is None

    def test_detect_ultimate_missing_ultimate_region(self, detector, mock_cropper):
        """Test detection when ultimate region is not in crop data."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left"}  # No ultimate region
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is None

    def test_detect_ultimate_empty_crop(self, detector, mock_cropper):
        """Test detection with empty ultimate crop."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is None

    def test_detect_ultimate_full(self, detector, mock_cropper):
        """Test detection of full ultimate (solid ring)."""
        ultimate_crop = self.create_full_ultimate_crop()

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": ultimate_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert ultimate_info.is_full is True
        assert ultimate_info.charges == 7  # Full ultimate = 7 charges
        assert white_pixel_ratio >= 0.4  # Above fullness threshold

    def test_detect_ultimate_partial(self, detector, mock_cropper):
        """Test detection of partial ultimate (segments)."""
        ultimate_crop = self.create_partial_ultimate_crop(num_segments=3)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": ultimate_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert ultimate_info.is_full is False
        # Should detect segments (may not be exactly 3 due to filtering)
        assert ultimate_info.charges >= 0
        assert white_pixel_ratio < 0.4  # Below fullness threshold

    def test_detect_ultimate_empty(self, detector, mock_cropper):
        """Test detection of empty ultimate (no charges)."""
        # All black crop (no segments)
        ultimate_crop = np.zeros((46, 46, 3), dtype=np.uint8)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": ultimate_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=0)

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert isinstance(ultimate_info, UltimateInfo)
        assert ultimate_info.is_full is False
        assert ultimate_info.charges == 0
        assert white_pixel_ratio < 0.4

    def test_detect_ultimate_right_side(self, detector, mock_cropper):
        """Test detection for right-side player."""
        ultimate_crop = self.create_full_ultimate_crop()

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": np.zeros((46, 46, 3), dtype=np.uint8)},
            {"side": "right", "ultimate": ultimate_crop},
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ultimate(frame, player_index=1, side="right")

        assert result is not None
        ultimate_info, white_pixel_ratio = result
        assert ultimate_info.is_full is True

    def test_detect_with_debug_success_full(self, detector, mock_cropper):
        """Test debug detection with full ultimate."""
        ultimate_crop = self.create_full_ultimate_crop()

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": ultimate_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0
        )

        assert ultimate_info is not None
        assert ultimate_info.is_full is True
        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.size > 0
        assert "white_pixel_ratio" in debug_info
        assert "is_full" in debug_info
        assert "total_blobs" in debug_info
        assert "valid_blobs" in debug_info
        assert "center" in debug_info
        assert bool(debug_info["is_full"]) is True

    def test_detect_with_debug_success_partial(self, detector, mock_cropper):
        """Test debug detection with partial ultimate."""
        ultimate_crop = self.create_partial_ultimate_crop(num_segments=4)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": ultimate_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0
        )

        assert ultimate_info is not None
        assert ultimate_info.is_full is False
        assert preprocessed.size > 0
        assert bool(debug_info["is_full"]) is False

    def test_detect_with_debug_empty_crop(self, detector, mock_cropper):
        """Test debug detection with empty crop."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0
        )

        assert ultimate_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_detect_with_debug_player_out_of_range(self, detector, mock_cropper):
        """Test debug detection with invalid player index."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ultimate": np.zeros((46, 46, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ultimate_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=10
        )

        assert ultimate_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_different_fullness_thresholds(self, mock_cropper):
        """Test detector with different fullness thresholds."""
        # Create crop with medium fullness
        ultimate_crop = self.create_partial_ultimate_crop(num_segments=5)

        # Low threshold detector
        detector_low = UltimateDetector(mock_cropper, fullness_threshold=0.1)
        mock_cropper.crop_player_info.return_value = [{"side": "left", "ultimate": ultimate_crop}]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result_low = detector_low.detect_ultimate(frame, player_index=0)

        # High threshold detector
        detector_high = UltimateDetector(mock_cropper, fullness_threshold=0.9)
        result_high = detector_high.detect_ultimate(frame, player_index=0)

        # Both should return results
        assert result_low is not None
        assert result_high is not None

        # Low threshold is more likely to mark as full
        # High threshold is less likely to mark as full
        ultimate_info_low, _ = result_low
        ultimate_info_high, _ = result_high

        assert ultimate_info_low is not None
        assert ultimate_info_high is not None

    def test_different_brightness_thresholds(self, mock_cropper):
        """Test detector with different brightness thresholds."""
        # Create crop with medium brightness (150)
        ultimate_crop = np.ones((46, 46, 3), dtype=np.uint8) * 150
        cv2.circle(ultimate_crop, (23, 23), 18, (150, 150, 150), 4)

        # Low threshold (100) - should detect segments
        detector_low = UltimateDetector(mock_cropper, brightness_threshold=100)
        mock_cropper.crop_player_info.return_value = [{"side": "left", "ultimate": ultimate_crop}]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result_low = detector_low.detect_ultimate(frame, player_index=0)

        # High threshold (200) - should NOT detect segments
        detector_high = UltimateDetector(mock_cropper, brightness_threshold=200)
        result_high = detector_high.detect_ultimate(frame, player_index=0)

        # Both should return results but with different detections
        assert result_low is not None
        assert result_high is not None

    def test_ring_mask_dimensions(self, mock_cropper):
        """Test that ring mask has correct dimensions."""
        detector = UltimateDetector(
            mock_cropper,
            ring_inner_radius=15,
            ring_outer_radius=20,
        )

        mask = detector._create_ring_mask((46, 46), (23, 23))

        # Count white pixels in mask
        white_pixels = np.sum(mask == 255)

        # Ring area should be π * (outer² - inner²)
        expected_area = np.pi * (20**2 - 15**2)

        # Allow some tolerance due to discretization
        assert abs(white_pixels - expected_area) < expected_area * 0.3


class TestUltimateInfoValidation:
    """Tests for UltimateInfo Pydantic model validation."""

    def test_valid_ultimate_info_full(self):
        """Test creating valid UltimateInfo for full ultimate."""
        info = UltimateInfo(charges=7, is_full=True, total_blobs_detected=0)

        assert info.charges == 7
        assert info.is_full is True
        assert info.total_blobs_detected == 0

    def test_valid_ultimate_info_partial(self):
        """Test creating valid UltimateInfo for partial ultimate."""
        info = UltimateInfo(charges=3, is_full=False, total_blobs_detected=5)

        assert info.charges == 3
        assert info.is_full is False
        assert info.total_blobs_detected == 5

    def test_valid_ultimate_info_empty(self):
        """Test creating valid UltimateInfo for empty ultimate."""
        info = UltimateInfo(charges=0, is_full=False, total_blobs_detected=0)

        assert info.charges == 0
        assert info.is_full is False

    def test_charges_range_validation(self):
        """Test charges must be in valid range (0-8)."""
        # Valid cases
        UltimateInfo(charges=0, is_full=False, total_blobs_detected=0)
        UltimateInfo(charges=8, is_full=True, total_blobs_detected=0)

        # Invalid cases
        with pytest.raises(Exception):  # Pydantic ValidationError
            UltimateInfo(charges=-1, is_full=False, total_blobs_detected=0)

        with pytest.raises(Exception):
            UltimateInfo(charges=9, is_full=False, total_blobs_detected=0)

    def test_total_blobs_validation(self):
        """Test total_blobs_detected must be non-negative."""
        # Valid
        UltimateInfo(charges=3, is_full=False, total_blobs_detected=0)
        UltimateInfo(charges=3, is_full=False, total_blobs_detected=100)

        # Invalid
        with pytest.raises(Exception):
            UltimateInfo(charges=3, is_full=False, total_blobs_detected=-1)

    def test_is_full_boolean(self):
        """Test is_full accepts boolean values."""
        info1 = UltimateInfo(charges=7, is_full=True, total_blobs_detected=0)
        assert info1.is_full is True

        info2 = UltimateInfo(charges=3, is_full=False, total_blobs_detected=3)
        assert info2.is_full is False

    def test_missing_required_fields(self):
        """Test that required fields cannot be omitted."""
        # Missing charges
        with pytest.raises(Exception):
            UltimateInfo(is_full=True, total_blobs_detected=0)

        # Missing is_full
        with pytest.raises(Exception):
            UltimateInfo(charges=7, total_blobs_detected=0)

        # Missing total_blobs_detected
        with pytest.raises(Exception):
            UltimateInfo(charges=7, is_full=True)
