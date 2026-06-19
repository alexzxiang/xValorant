"""Unit tests for blob-based ability detector."""

from __future__ import annotations
from unittest.mock import Mock
import pytest
import numpy as np
import cv2

from valoscribe.detectors.ability_detector import AbilityDetector
from valoscribe.types.detections import AbilityInfo


class TestAbilityDetector:
    """Tests for AbilityDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        return cropper

    @pytest.fixture
    def detector(self, mock_cropper):
        """Create ability detector with default settings."""
        return AbilityDetector(
            mock_cropper,
            brightness_threshold=127,
            min_blob_area=10,
            max_blob_area=500,
            min_circularity=0.3,
        )

    def create_blob_image(self, num_blobs: int, blob_size: int = 20) -> np.ndarray:
        """
        Helper to create test image with specified number of bright blobs.

        Args:
            num_blobs: Number of blobs to create
            blob_size: Size of each blob in pixels

        Returns:
            Test image with blobs
        """
        # Create black background
        image = np.zeros((50, 200), dtype=np.uint8)

        # Add bright circular blobs
        for i in range(num_blobs):
            center_x = 30 + i * 40
            center_y = 25
            cv2.circle(image, (center_x, center_y), blob_size // 2, 255, -1)

        return image

    def test_init(self, mock_cropper):
        """Test ability detector initialization."""
        detector = AbilityDetector(
            mock_cropper,
            brightness_threshold=150,
            min_blob_area=20,
            max_blob_area=400,
            min_circularity=0.5,
        )

        assert detector.cropper == mock_cropper
        assert detector.brightness_threshold == 150
        assert detector.min_blob_area == 20
        assert detector.max_blob_area == 400
        assert detector.min_circularity == 0.5

    def test_count_blobs_zero_charges(self, detector):
        """Test blob counting with no blobs (ability on cooldown)."""
        # Empty black image
        image = np.zeros((50, 100), dtype=np.uint8)

        charges, total_blobs = detector._count_blobs(image)

        assert charges == 0
        assert total_blobs == 0

    def test_count_blobs_single_charge(self, detector):
        """Test blob counting with one blob."""
        image = self.create_blob_image(num_blobs=1)

        charges, total_blobs = detector._count_blobs(image)

        assert charges == 1
        assert total_blobs >= 1

    def test_count_blobs_two_charges(self, detector):
        """Test blob counting with two blobs (e.g., Jett dash)."""
        image = self.create_blob_image(num_blobs=2)

        charges, total_blobs = detector._count_blobs(image)

        assert charges == 2
        assert total_blobs >= 2

    def test_count_blobs_three_charges(self, detector):
        """Test blob counting with three blobs."""
        image = self.create_blob_image(num_blobs=3)

        charges, total_blobs = detector._count_blobs(image)

        assert charges == 3
        assert total_blobs >= 3

    def test_count_blobs_filters_small_noise(self, detector):
        """Test that small noise pixels are filtered out."""
        image = np.zeros((50, 100), dtype=np.uint8)

        # Add valid blob
        cv2.circle(image, (30, 25), 10, 255, -1)

        # Add tiny noise (below min_blob_area)
        image[10:12, 70:72] = 255  # 4 pixels

        charges, total_blobs = detector._count_blobs(image)

        # Should only count the valid blob, not the noise
        assert charges == 1

    def test_count_blobs_filters_large_regions(self, detector):
        """Test that too-large regions are filtered out."""
        image = np.zeros((100, 100), dtype=np.uint8)

        # Add valid blob
        cv2.circle(image, (30, 30), 10, 255, -1)

        # Add huge blob (exceeds max_blob_area=500)
        cv2.circle(image, (70, 70), 40, 255, -1)  # Area ~5000

        charges, total_blobs = detector._count_blobs(image)

        # Should only count valid blob
        assert charges == 1

    def test_count_blobs_filters_non_circular(self, detector):
        """Test that non-circular blobs are filtered by circularity."""
        image = np.zeros((50, 150), dtype=np.uint8)

        # Add circular blob
        cv2.circle(image, (30, 25), 10, 255, -1)

        # Add very elongated rectangle (very low circularity)
        # A 3x60 rectangle has circularity ≈ 0.15, well below min_circularity=0.3
        image[22:25, 80:140] = 255  # 3x60 rectangle

        charges, total_blobs = detector._count_blobs(image)

        # Rectangle should be filtered out due to low circularity
        assert charges == 1

    def test_preprocess_crop_color_input(self, detector):
        """Test preprocessing with color input."""
        # Create color image with bright region
        crop = np.zeros((30, 40, 3), dtype=np.uint8)
        crop[10:20, 15:25] = 200  # Bright region

        preprocessed = detector._preprocess_crop(crop)

        # Should be grayscale
        assert len(preprocessed.shape) == 2
        # Should be binary
        assert np.all((preprocessed == 0) | (preprocessed == 255))
        # Bright region should be white
        assert np.any(preprocessed[10:20, 15:25] == 255)

    def test_preprocess_crop_grayscale_input(self, detector):
        """Test preprocessing with grayscale input."""
        crop = np.zeros((30, 40), dtype=np.uint8)
        crop[10:20, 15:25] = 200

        preprocessed = detector._preprocess_crop(crop)

        assert len(preprocessed.shape) == 2
        assert np.all((preprocessed == 0) | (preprocessed == 255))

    def test_detect_ability_player_out_of_range(self, detector, mock_cropper):
        """Test detection with invalid player index."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=10, ability_name="ability_1")

        assert result is None

    def test_detect_ability_wrong_side(self, detector, mock_cropper):
        """Test detection with wrong side specified."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(
            frame, player_index=0, ability_name="ability_1", side="right"
        )

        assert result is None

    def test_detect_ability_missing_ability_region(self, detector, mock_cropper):
        """Test detection when ability region is not in crop data."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left"}  # No ability regions
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is None

    def test_detect_ability_empty_crop(self, detector, mock_cropper):
        """Test detection with empty ability crop."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ability_1": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is None

    def test_detect_ability_success(self, detector, mock_cropper):
        """Test successful ability detection."""
        # Create ability crop with 2 blobs
        ability_crop = self.create_blob_image(num_blobs=2, blob_size=15)
        # Convert to color for consistency
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ability_1": ability_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is not None
        assert isinstance(result, AbilityInfo)
        assert result.charges == 2
        assert result.total_blobs_detected >= 2

    def test_detect_player_abilities_all_three(self, detector, mock_cropper):
        """Test detecting all three abilities for a player."""
        # Create different number of blobs for each ability
        ability1_crop = cv2.cvtColor(self.create_blob_image(1), cv2.COLOR_GRAY2BGR)
        ability2_crop = cv2.cvtColor(self.create_blob_image(2), cv2.COLOR_GRAY2BGR)
        ability3_crop = cv2.cvtColor(self.create_blob_image(0), cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info.return_value = [
            {
                "side": "left",
                "ability_1": ability1_crop,
                "ability_2": ability2_crop,
                "ability_3": ability3_crop,
            }
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = detector.detect_player_abilities(frame, player_index=0)

        assert "ability_1" in results
        assert "ability_2" in results
        assert "ability_3" in results

        assert results["ability_1"].charges == 1
        assert results["ability_2"].charges == 2
        assert results["ability_3"].charges == 0

    def test_detect_with_debug_success(self, detector, mock_cropper):
        """Test debug detection with successful blob detection."""
        ability_crop = cv2.cvtColor(self.create_blob_image(2), cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ability_1": ability_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ability_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0, ability_name="ability_1"
        )

        assert ability_info is not None
        assert ability_info.charges == 2
        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.size > 0
        assert "total_blobs" in debug_info
        assert "blob_stats" in debug_info
        assert len(debug_info["blob_stats"]) >= 2

    def test_detect_with_debug_empty_crop(self, detector, mock_cropper):
        """Test debug detection with empty crop."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "ability_1": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ability_info, preprocessed, debug_info = detector.detect_with_debug(
            frame, player_index=0, ability_name="ability_1"
        )

        assert ability_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_different_brightness_thresholds(self, mock_cropper):
        """Test detector with different brightness thresholds."""
        # Create image with medium brightness (150)
        image = np.ones((50, 100), dtype=np.uint8) * 150

        # Low threshold (100) - should detect
        detector_low = AbilityDetector(mock_cropper, brightness_threshold=100)
        preprocessed_low = detector_low._preprocess_crop(image)
        assert np.any(preprocessed_low == 255)

        # High threshold (200) - should NOT detect
        detector_high = AbilityDetector(mock_cropper, brightness_threshold=200)
        preprocessed_high = detector_high._preprocess_crop(image)
        assert np.all(preprocessed_high == 0)

    def test_blob_area_constraints(self, mock_cropper):
        """Test that min/max blob area constraints work."""
        # Create image with blobs of different sizes
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (20, 20), 3, 255, -1)   # Small: ~28 px
        cv2.circle(image, (50, 50), 10, 255, -1)  # Medium: ~314 px
        cv2.circle(image, (80, 80), 20, 255, -1)  # Large: ~1256 px

        # Detector that only accepts medium-sized blobs
        detector = AbilityDetector(
            mock_cropper,
            min_blob_area=100,
            max_blob_area=500,
        )

        charges, _ = detector._count_blobs(image)
        assert charges == 1  # Only medium blob


class TestAbilityInfoValidation:
    """Tests for AbilityInfo Pydantic model validation."""

    def test_valid_ability_info(self):
        """Test creating valid AbilityInfo."""
        info = AbilityInfo(charges=2, total_blobs_detected=3)

        assert info.charges == 2
        assert info.total_blobs_detected == 3

    def test_charges_range_validation(self):
        """Test charges must be in valid range (0-10)."""
        # Valid cases
        AbilityInfo(charges=0, total_blobs_detected=0)
        AbilityInfo(charges=10, total_blobs_detected=10)

        # Invalid cases
        with pytest.raises(Exception):  # Pydantic ValidationError
            AbilityInfo(charges=-1, total_blobs_detected=0)

        with pytest.raises(Exception):
            AbilityInfo(charges=11, total_blobs_detected=11)

    def test_total_blobs_validation(self):
        """Test total_blobs_detected must be non-negative."""
        # Valid
        AbilityInfo(charges=2, total_blobs_detected=0)
        AbilityInfo(charges=2, total_blobs_detected=100)

        # Invalid
        with pytest.raises(Exception):
            AbilityInfo(charges=2, total_blobs_detected=-1)

    def test_charges_can_be_less_than_total_blobs(self):
        """Test that charges can be less than total_blobs (due to filtering)."""
        info = AbilityInfo(charges=1, total_blobs_detected=5)
        assert info.charges == 1
        assert info.total_blobs_detected == 5
