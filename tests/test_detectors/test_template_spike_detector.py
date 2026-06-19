"""Unit tests for template-based spike detector."""

from __future__ import annotations
from unittest.mock import Mock, patch
from pathlib import Path
import pytest
import numpy as np
import cv2

from valoscribe.detectors.template_spike_detector import TemplateSpikeDetector
from valoscribe.types.detections import SpikeInfo


class TestTemplateSpikeDetector:
    """Tests for TemplateSpikeDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        # Default: return non-empty crops
        cropper.crop_simple_region.return_value = np.zeros((40, 80, 3), dtype=np.uint8)
        return cropper

    @pytest.fixture
    def mock_template(self):
        """Create mock spike template."""
        # Create simple template (108x191 white spike on black background)
        template = np.zeros((108, 191), dtype=np.uint8)
        # Add some white pixels to simulate spike icon
        template[40:70, 80:110] = 255
        return template

    @pytest.fixture
    def detector(self, mock_cropper, mock_template, tmp_path):
        """Create template spike detector with mocked template."""
        # Create temporary template file
        template_path = tmp_path / "spike.png"
        cv2.imwrite(str(template_path), mock_template)

        detector = TemplateSpikeDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.7
        )

        return detector

    def test_init(self, mock_cropper, tmp_path):
        """Test template spike detector initialization."""
        template_path = tmp_path / "spike.png"
        template = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(template_path), template)

        detector = TemplateSpikeDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.8
        )

        assert detector.cropper == mock_cropper
        assert detector.min_confidence == 0.8
        assert detector.match_method == cv2.TM_CCOEFF_NORMED

    def test_load_template_success(self, mock_cropper, tmp_path):
        """Test successful template loading."""
        template_path = tmp_path / "spike.png"
        template = np.zeros((108, 191), dtype=np.uint8)
        cv2.imwrite(str(template_path), template)

        detector = TemplateSpikeDetector(mock_cropper, template_path=template_path)

        assert detector.template is not None
        assert detector.template.shape == (108, 191)

    def test_load_template_missing_file(self, mock_cropper, tmp_path):
        """Test loading template from non-existent file."""
        template_path = tmp_path / "nonexistent.png"

        detector = TemplateSpikeDetector(mock_cropper, template_path=template_path)

        assert detector.template is None

    def test_load_template_invalid_file(self, mock_cropper, tmp_path):
        """Test loading invalid template file."""
        template_path = tmp_path / "invalid.png"
        # Create invalid PNG file
        template_path.write_text("not a valid image")

        detector = TemplateSpikeDetector(mock_cropper, template_path=template_path)

        assert detector.template is None

    def test_detect_no_template(self, mock_cropper, tmp_path):
        """Test detection fails when no template is loaded."""
        template_path = tmp_path / "nonexistent.png"

        detector = TemplateSpikeDetector(mock_cropper, template_path=template_path)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_empty_crop(self, detector, mock_cropper):
        """Test detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_spike_planted_high_confidence(self, detector, mock_cropper):
        """Test successful spike detection with high confidence."""
        # Mock matchTemplate to return high confidence match
        with patch('cv2.matchTemplate') as mock_match:
            # Create result with high confidence at a location
            result = np.zeros((1, 1))
            result[0, 0] = 0.95
            mock_match.return_value = result

            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            spike_info = detector.detect(frame)

            assert spike_info is not None
            assert spike_info.spike_planted is True
            assert spike_info.confidence >= 0.7

    def test_detect_spike_not_planted_low_confidence(self, detector, mock_cropper):
        """Test spike not detected with low confidence."""
        # Mock matchTemplate to return low confidence
        with patch('cv2.matchTemplate') as mock_match:
            result = np.zeros((1, 1))
            result[0, 0] = 0.3  # Below threshold of 0.7
            mock_match.return_value = result

            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            spike_info = detector.detect(frame)

            # Should return None for low confidence
            assert spike_info is None

    def test_detect_spike_threshold_boundary(self, detector, mock_cropper):
        """Test spike detection at confidence threshold boundary."""
        # Mock matchTemplate to return confidence exactly at threshold
        with patch('cv2.matchTemplate') as mock_match:
            result = np.zeros((1, 1))
            result[0, 0] = 0.7  # Exactly at threshold
            mock_match.return_value = result

            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            spike_info = detector.detect(frame)

            assert spike_info is not None
            assert spike_info.spike_planted is True
            assert spike_info.confidence == 0.7

    def test_detect_calls_cropper_with_timer_region(self, detector, mock_cropper):
        """Test that detect calls cropper with 'round_timer' region."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detector.detect(frame)

        # Verify cropper was called with correct region name
        mock_cropper.crop_simple_region.assert_called_once_with(frame, "round_timer")

    def test_preprocess_crop(self, detector):
        """Test crop preprocessing."""
        # Create color crop
        crop = np.ones((12, 23, 3), dtype=np.uint8) * 200

        preprocessed = detector._preprocess_crop(crop)

        # Should be grayscale
        assert len(preprocessed.shape) == 2

        # Should be upscaled 3x
        assert preprocessed.shape[0] == 12 * 3
        assert preprocessed.shape[1] == 23 * 3

        # Should be binary (Otsu's threshold)
        assert np.all((preprocessed == 0) | (preprocessed == 255))

    def test_preprocess_crop_grayscale_input(self, detector):
        """Test preprocessing with grayscale input."""
        # Create grayscale crop
        crop = np.ones((20, 40), dtype=np.uint8) * 150

        preprocessed = detector._preprocess_crop(crop)

        # Should still be grayscale
        assert len(preprocessed.shape) == 2

        # Should be upscaled 3x
        assert preprocessed.shape[0] == 20 * 3
        assert preprocessed.shape[1] == 40 * 3

    def test_detect_with_debug_spike_planted(self, detector, mock_cropper):
        """Test debug detection with spike planted."""
        with patch('cv2.matchTemplate') as mock_match:
            result = np.zeros((1, 1))
            result[0, 0] = 0.85
            mock_match.return_value = result

            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            spike_info, timer_preprocessed, debug_info = detector.detect_with_debug(frame)

            assert spike_info is not None
            assert spike_info.spike_planted is True
            assert isinstance(timer_preprocessed, np.ndarray)
            assert timer_preprocessed.size > 0
            assert "max_confidence" in debug_info
            assert debug_info["max_confidence"] == 0.85
            assert "max_location" in debug_info
            assert "match_result_shape" in debug_info

    def test_detect_with_debug_spike_not_planted(self, detector, mock_cropper):
        """Test debug detection with spike not planted."""
        with patch('cv2.matchTemplate') as mock_match:
            result = np.zeros((1, 1))
            result[0, 0] = 0.4  # Low confidence
            mock_match.return_value = result

            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            spike_info, timer_preprocessed, debug_info = detector.detect_with_debug(frame)

            assert spike_info is None
            assert isinstance(timer_preprocessed, np.ndarray)
            assert timer_preprocessed.size > 0
            assert "max_confidence" in debug_info
            assert debug_info["max_confidence"] == 0.4

    def test_detect_with_debug_empty_crop(self, detector, mock_cropper):
        """Test debug detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        spike_info, timer_preprocessed, debug_info = detector.detect_with_debug(frame)

        assert spike_info is None
        assert timer_preprocessed.size == 0
        assert debug_info == {}

    def test_detect_with_debug_no_template(self, mock_cropper, tmp_path):
        """Test debug detection when no template is loaded."""
        template_path = tmp_path / "nonexistent.png"
        detector = TemplateSpikeDetector(mock_cropper, template_path=template_path)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        spike_info, timer_preprocessed, debug_info = detector.detect_with_debug(frame)

        assert spike_info is None
        # Preprocessed should still be created
        assert isinstance(timer_preprocessed, np.ndarray)
        # Debug info should be empty since template is None
        assert debug_info == {}

    def test_different_confidence_thresholds(self, mock_cropper, tmp_path):
        """Test detector with different confidence thresholds."""
        template_path = tmp_path / "spike.png"
        template = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(template_path), template)

        # Test with low threshold
        detector_low = TemplateSpikeDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.5
        )
        assert detector_low.min_confidence == 0.5

        # Test with high threshold
        detector_high = TemplateSpikeDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.9
        )
        assert detector_high.min_confidence == 0.9

    def test_template_matching_method(self, mock_cropper, tmp_path):
        """Test that detector uses correct template matching method."""
        template_path = tmp_path / "spike.png"
        template = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(template_path), template)

        # Default method
        detector_default = TemplateSpikeDetector(
            mock_cropper,
            template_path=template_path
        )
        assert detector_default.match_method == cv2.TM_CCOEFF_NORMED

        # Custom method
        detector_custom = TemplateSpikeDetector(
            mock_cropper,
            template_path=template_path,
            match_method=cv2.TM_CCORR_NORMED
        )
        assert detector_custom.match_method == cv2.TM_CCORR_NORMED


class TestSpikeInfoValidation:
    """Tests for SpikeInfo Pydantic model validation."""

    def test_valid_spike_info_planted(self):
        """Test creating valid SpikeInfo for planted spike."""
        info = SpikeInfo(
            spike_planted=True,
            confidence=0.88,
        )

        assert info.spike_planted is True
        assert info.confidence == 0.88

    def test_valid_spike_info_not_planted(self):
        """Test creating valid SpikeInfo for not planted."""
        info = SpikeInfo(
            spike_planted=False,
            confidence=0.45,
        )

        assert info.spike_planted is False
        assert info.confidence == 0.45

    def test_confidence_range_validation(self):
        """Test confidence must be in valid range (0-1)."""
        # Valid cases
        SpikeInfo(spike_planted=True, confidence=0.0)
        SpikeInfo(spike_planted=True, confidence=1.0)

        # Invalid cases
        with pytest.raises(Exception):  # Pydantic ValidationError
            SpikeInfo(spike_planted=True, confidence=-0.1)

        with pytest.raises(Exception):
            SpikeInfo(spike_planted=True, confidence=1.5)

    def test_spike_planted_type_validation(self):
        """Test spike_planted must be boolean."""
        # Valid cases
        SpikeInfo(spike_planted=True, confidence=0.8)
        SpikeInfo(spike_planted=False, confidence=0.8)

        # These should work due to Pydantic's coercion
        info1 = SpikeInfo(spike_planted=1, confidence=0.8)
        assert info1.spike_planted is True

        info2 = SpikeInfo(spike_planted=0, confidence=0.8)
        assert info2.spike_planted is False
