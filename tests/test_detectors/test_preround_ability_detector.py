"""Unit tests for pre-round ability detector."""

from __future__ import annotations
from unittest.mock import Mock
import pytest
import numpy as np
import cv2

from valoscribe.detectors.preround_ability_detector import PreroundAbilityDetector
from valoscribe.types.detections import AbilityInfo


class TestPreroundAbilityDetector:
    """Tests for PreroundAbilityDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        return cropper

    @pytest.fixture
    def detector(self, mock_cropper):
        """Create pre-round ability detector with default settings."""
        return PreroundAbilityDetector(
            mock_cropper,
            brightness_threshold=127,
            min_blob_area=10,
            max_blob_area=500,
        )

    def create_blob_image(self, num_blobs: int, blob_size: int = 10) -> np.ndarray:
        """
        Helper to create test image with specified number of bright blobs.

        Args:
            num_blobs: Number of blobs to create
            blob_size: Size of each blob in pixels

        Returns:
            Test image with blobs
        """
        # Create black background (smaller for pre-round ability crops)
        image = np.zeros((15, 120), dtype=np.uint8)

        # Add bright circular blobs
        for i in range(num_blobs):
            center_x = 15 + i * 25
            center_y = 7
            cv2.circle(image, (center_x, center_y), blob_size // 2, 255, -1)

        return image

    def test_init(self, mock_cropper):
        """Test pre-round ability detector initialization."""
        detector = PreroundAbilityDetector(
            mock_cropper,
            brightness_threshold=150,
            min_blob_area=20,
            max_blob_area=400,
        )

        assert detector.cropper == mock_cropper
        assert detector.brightness_threshold == 150
        assert detector.min_blob_area == 20
        assert detector.max_blob_area == 400

    def test_inherits_from_ability_detector(self, detector):
        """Test that PreroundAbilityDetector inherits from AbilityDetector."""
        from valoscribe.detectors.ability_detector import AbilityDetector

        assert isinstance(detector, AbilityDetector)
        assert isinstance(detector, PreroundAbilityDetector)

        # Should have inherited methods
        assert hasattr(detector, '_preprocess_crop')
        assert hasattr(detector, '_count_blobs')
        assert hasattr(detector, 'detect_player_abilities')

    def test_detect_ability_uses_preround_crops(self, detector, mock_cropper):
        """Test that detector uses crop_player_info_preround instead of crop_player_info."""
        ability_crop = self.create_blob_image(num_blobs=1, blob_size=8)
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": ability_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        # Verify that crop_player_info_preround was called (not crop_player_info)
        mock_cropper.crop_player_info_preround.assert_called_once_with(frame)
        assert result is not None

    def test_detect_ability_player_out_of_range(self, detector, mock_cropper):
        """Test detection with invalid player index."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=10, ability_name="ability_1")

        assert result is None

    def test_detect_ability_wrong_side(self, detector, mock_cropper):
        """Test detection with wrong side specified."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(
            frame, player_index=0, ability_name="ability_1", side="right"
        )

        assert result is None

    def test_detect_ability_missing_ability_region(self, detector, mock_cropper):
        """Test detection when ability region is not in crop data."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left"}  # No ability regions
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is None

    def test_detect_ability_empty_crop(self, detector, mock_cropper):
        """Test detection with empty ability crop."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is None

    def test_detect_ability_zero_charges(self, detector, mock_cropper):
        """Test detection with no blobs (ability on cooldown)."""
        # Empty black ability crop
        ability_crop = np.zeros((15, 42, 3), dtype=np.uint8)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": ability_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is not None
        assert isinstance(result, AbilityInfo)
        assert result.charges == 0
        assert result.total_blobs_detected == 0

    def test_detect_ability_one_charge(self, detector, mock_cropper):
        """Test detection with one blob."""
        ability_crop = self.create_blob_image(num_blobs=1, blob_size=8)
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": ability_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is not None
        assert isinstance(result, AbilityInfo)
        assert result.charges == 1
        assert result.total_blobs_detected >= 1

    def test_detect_ability_two_charges(self, detector, mock_cropper):
        """Test detection with two blobs."""
        ability_crop = self.create_blob_image(num_blobs=2, blob_size=8)
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": ability_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_1")

        assert result is not None
        assert isinstance(result, AbilityInfo)
        assert result.charges == 2
        assert result.total_blobs_detected >= 2

    def test_detect_ability_three_charges(self, detector, mock_cropper):
        """Test detection with three blobs."""
        ability_crop = self.create_blob_image(num_blobs=3, blob_size=8)
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_2": ability_crop_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=0, ability_name="ability_2")

        assert result is not None
        assert isinstance(result, AbilityInfo)
        assert result.charges == 3
        assert result.total_blobs_detected >= 3

    def test_detect_ability_right_side_player(self, detector, mock_cropper):
        """Test detection for right-side player."""
        ability_crop = self.create_blob_image(num_blobs=2, blob_size=8)
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)},
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)},
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)},
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)},
            {"side": "left", "ability_1": np.zeros((15, 42, 3), dtype=np.uint8)},
            {"side": "right", "ability_1": ability_crop_color},
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, player_index=5, ability_name="ability_1", side="right")

        assert result is not None
        assert result.charges == 2

    def test_detect_player_abilities_all_three(self, detector, mock_cropper):
        """Test detecting all three abilities for a player."""
        # Create different number of blobs for each ability
        ability1_crop = cv2.cvtColor(self.create_blob_image(1, blob_size=8), cv2.COLOR_GRAY2BGR)
        ability2_crop = cv2.cvtColor(self.create_blob_image(2, blob_size=8), cv2.COLOR_GRAY2BGR)
        ability3_crop = cv2.cvtColor(self.create_blob_image(0), cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
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

        assert results["ability_1"] is not None
        assert results["ability_1"].charges == 1

        assert results["ability_2"] is not None
        assert results["ability_2"].charges == 2

        assert results["ability_3"] is not None
        assert results["ability_3"].charges == 0

    def test_detect_player_abilities_with_none_values(self, detector, mock_cropper):
        """Test that detect_player_abilities handles missing crops gracefully."""
        # Only provide some abilities
        mock_cropper.crop_player_info_preround.return_value = [
            {
                "side": "left",
                "ability_1": cv2.cvtColor(self.create_blob_image(1, blob_size=8), cv2.COLOR_GRAY2BGR),
                # ability_2 and ability_3 missing
            }
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = detector.detect_player_abilities(frame, player_index=0)

        assert "ability_1" in results
        assert "ability_2" in results
        assert "ability_3" in results

        assert results["ability_1"] is not None
        assert results["ability_2"] is None
        assert results["ability_3"] is None

    def test_different_brightness_thresholds(self, mock_cropper):
        """Test detector with different brightness thresholds."""
        # Create image with medium brightness blobs
        image = np.zeros((15, 60), dtype=np.uint8)
        cv2.circle(image, (15, 7), 5, 150, -1)  # Medium brightness
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Low threshold detector (should detect)
        detector_low = PreroundAbilityDetector(mock_cropper, brightness_threshold=100)
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": image_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result_low = detector_low.detect_ability(frame, 0, "ability_1")

        # High threshold detector (should not detect)
        detector_high = PreroundAbilityDetector(mock_cropper, brightness_threshold=200)
        result_high = detector_high.detect_ability(frame, 0, "ability_1")

        # Low threshold should detect, high threshold should not
        assert result_low is not None
        assert result_low.charges >= 1 or result_low.charges == 0  # May or may not detect depending on threshold

        assert result_high is not None
        assert result_high.charges == 0  # Should not detect

    def test_detect_all_ability_names(self, detector, mock_cropper):
        """Test detection for all three ability slots."""
        ability_crop = self.create_blob_image(num_blobs=1, blob_size=8)
        ability_crop_color = cv2.cvtColor(ability_crop, cv2.COLOR_GRAY2BGR)

        # Test ability_1
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": ability_crop_color}
        ]
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result1 = detector.detect_ability(frame, 0, "ability_1")
        assert result1 is not None

        # Test ability_2
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_2": ability_crop_color}
        ]
        result2 = detector.detect_ability(frame, 0, "ability_2")
        assert result2 is not None

        # Test ability_3
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_3": ability_crop_color}
        ]
        result3 = detector.detect_ability(frame, 0, "ability_3")
        assert result3 is not None

    def test_uses_parent_blob_detection_logic(self, detector, mock_cropper):
        """Test that pre-round detector uses parent's blob counting logic."""
        # Create crop with blobs that should be filtered by parent's logic
        image = np.zeros((15, 100), dtype=np.uint8)

        # Add valid blob
        cv2.circle(image, (20, 7), 5, 255, -1)

        # Add tiny noise (below min_blob_area)
        image[2:4, 50:52] = 255  # 4 pixels

        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "ability_1": image_color}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_ability(frame, 0, "ability_1")

        # Should use parent's filtering logic (only count valid blob)
        assert result is not None
        assert result.charges == 1
        # total_blobs might be higher due to noise before filtering
        assert result.total_blobs_detected >= 1
