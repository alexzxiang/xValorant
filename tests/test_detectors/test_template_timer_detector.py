"""Unit tests for template-based timer detector."""

from __future__ import annotations
from unittest.mock import Mock, patch
from pathlib import Path
import pytest
import numpy as np
import cv2

from valoscribe.detectors.template_timer_detector import TemplateTimerDetector
from valoscribe.types.detections import TimerInfo


class TestTemplateTimerDetector:
    """Tests for TemplateTimerDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        # Default: return non-empty crops
        cropper.crop_simple_region.return_value = np.zeros((40, 80, 3), dtype=np.uint8)
        return cropper

    @pytest.fixture
    def mock_templates(self):
        """Create mock templates for digits 0-9."""
        templates = {}
        for digit in range(10):
            # Create simple template (30x20 white digit on black background)
            template = np.zeros((30, 20), dtype=np.uint8)
            templates[str(digit)] = template
        return templates

    @pytest.fixture
    def detector(self, mock_cropper, mock_templates, tmp_path):
        """Create template timer detector with mocked templates."""
        # Create temporary template directory
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Save mock templates
        for digit, template in mock_templates.items():
            cv2.imwrite(str(template_dir / f"{digit}.png"), template)

        detector = TemplateTimerDetector(
            mock_cropper,
            template_dir=template_dir,
            min_confidence=0.6
        )

        return detector

    def test_init(self, mock_cropper, tmp_path):
        """Test template timer detector initialization."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        detector = TemplateTimerDetector(
            mock_cropper,
            template_dir=template_dir,
            min_confidence=0.7
        )

        assert detector.cropper == mock_cropper
        assert detector.min_confidence == 0.7
        assert detector.match_method == cv2.TM_CCOEFF_NORMED

    def test_load_templates_success(self, mock_cropper, tmp_path):
        """Test successful template loading."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Create templates for 0-9
        for digit in range(10):
            template = np.zeros((30, 20), dtype=np.uint8)
            cv2.imwrite(str(template_dir / f"{digit}.png"), template)

        detector = TemplateTimerDetector(mock_cropper, template_dir=template_dir)

        assert len(detector.templates) == 10
        assert all(str(i) in detector.templates for i in range(10))

    def test_load_templates_missing_directory(self, mock_cropper, tmp_path):
        """Test loading templates from non-existent directory."""
        template_dir = tmp_path / "nonexistent"

        detector = TemplateTimerDetector(mock_cropper, template_dir=template_dir)

        assert len(detector.templates) == 0

    def test_load_templates_partial(self, mock_cropper, tmp_path):
        """Test loading templates when only some exist."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Only create templates for 0, 1, 2
        for digit in [0, 1, 2]:
            template = np.zeros((30, 20), dtype=np.uint8)
            cv2.imwrite(str(template_dir / f"{digit}.png"), template)

        detector = TemplateTimerDetector(mock_cropper, template_dir=template_dir)

        assert len(detector.templates) == 3
        assert "0" in detector.templates
        assert "1" in detector.templates
        assert "2" in detector.templates
        assert "3" not in detector.templates

    def test_detect_no_templates(self, mock_cropper, tmp_path):
        """Test detection fails when no templates are loaded."""
        template_dir = tmp_path / "empty"
        template_dir.mkdir()

        detector = TemplateTimerDetector(mock_cropper, template_dir=template_dir)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_empty_crop(self, detector, mock_cropper):
        """Test detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_mss_format_3_digits(self, detector, mock_cropper):
        """Test detection of m:ss format (3 digits)."""
        # Mock _match_timer_region to return 3 digits: "130" = 1:30 = 90s
        detector._match_timer_region = Mock(return_value=(90.0, 0.85, "130"))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.time_seconds == 90.0
        assert result.confidence == 0.85
        assert result.raw_text == "130"

    def test_detect_ssms_format_4_digits(self, detector, mock_cropper):
        """Test detection of ss.ms format (4 digits)."""
        # Mock _match_timer_region to return 4 digits: "0967" = 09.67s
        detector._match_timer_region = Mock(return_value=(9.67, 0.82, "0967"))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.time_seconds == 9.67
        assert result.confidence == 0.82
        assert result.raw_text == "0967"

    def test_detect_out_of_range_time(self, detector, mock_cropper):
        """Test detection rejects time out of range."""
        # Return time > 100s
        detector._match_timer_region = Mock(return_value=(150.0, 0.85, "250"))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_match_fails(self, detector, mock_cropper):
        """Test detection when template matching fails."""
        detector._match_timer_region = Mock(return_value=None)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_match_timer_region_3_digits_mss(self, detector):
        """Test matching 3 digits for m:ss format."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Mock _find_all_digit_matches to return 3 digits: "1", "4", "5"
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "1", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "4", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "5", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_timer_region(crop)

        assert result is not None
        time_seconds, confidence, raw_text = result
        assert time_seconds == 105.0  # 1:45 = 105 seconds
        assert confidence == 0.85  # minimum
        assert raw_text == "145"

    def test_match_timer_region_4_digits_ssms(self, detector):
        """Test matching 4 digits for ss.ms format."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Mock _find_all_digit_matches to return 4 digits: "0", "9", "6", "7"
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "0", "confidence": 0.92, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "9", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "6", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
            {"digit": "7", "confidence": 0.83, "x": 80, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_timer_region(crop)

        assert result is not None
        time_seconds, confidence, raw_text = result
        assert time_seconds == 9.67  # 09.67 seconds
        assert confidence == 0.83  # minimum
        assert raw_text == "0967"

    def test_match_timer_region_invalid_digit_count(self, detector):
        """Test matching with invalid digit count (not 3 or 4)."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Mock _find_all_digit_matches to return 2 digits (invalid)
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "1", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "2", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_timer_region(crop)

        # Should return None for invalid digit count
        assert result is None

    def test_match_timer_region_low_confidence(self, detector):
        """Test matching rejects low confidence."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "1", "confidence": 0.40, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "4", "confidence": 0.38, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "5", "confidence": 0.35, "x": 55, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_timer_region(crop)

        # Should return None because 0.35 < 0.6 (min_confidence)
        assert result is None

    def test_match_timer_region_no_matches(self, detector):
        """Test matching when no digits are found."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        detector._find_all_digit_matches = Mock(return_value=[])

        result = detector._match_timer_region(crop)

        assert result is None

    def test_find_all_digit_matches(self, detector):
        """Test finding all digit matches in an image."""
        # Create test image
        image = np.zeros((100, 100), dtype=np.uint8)

        # Mock matchTemplate to return controlled results
        with patch('cv2.matchTemplate') as mock_match:
            # Simulate finding a digit at x=10, y=10 with high confidence
            result = np.zeros((80, 70))  # Result is smaller than input
            result[10, 10] = 0.95
            mock_match.return_value = result

            matches = detector._find_all_digit_matches(image)

            # Should find one match per template (10 templates)
            assert len(matches) > 0
            # Each match should have required fields
            for match in matches:
                assert "digit" in match
                assert "confidence" in match
                assert "x" in match
                assert "y" in match
                assert "w" in match
                assert "h" in match

    def test_filter_overlapping_matches_no_overlap(self, detector):
        """Test filtering when matches don't overlap."""
        matches = [
            {"digit": "1", "confidence": 0.95, "x": 0, "y": 0, "w": 20, "h": 30},
            {"digit": "4", "confidence": 0.90, "x": 25, "y": 0, "w": 20, "h": 30},
        ]

        filtered = detector._filter_overlapping_matches(matches)

        # Both should remain since they don't overlap
        assert len(filtered) == 2

    def test_filter_overlapping_matches_with_overlap(self, detector):
        """Test filtering when matches overlap significantly."""
        matches = [
            {"digit": "1", "confidence": 0.95, "x": 0, "y": 0, "w": 20, "h": 30},
            {"digit": "7", "confidence": 0.85, "x": 5, "y": 0, "w": 20, "h": 30},  # Overlaps with "1"
        ]

        filtered = detector._filter_overlapping_matches(matches)

        # Only highest confidence should remain
        assert len(filtered) == 1
        assert filtered[0]["digit"] == "1"
        assert filtered[0]["confidence"] == 0.95

    def test_filter_overlapping_matches_preserves_order(self, detector):
        """Test that filtering and re-sorting preserves left-to-right order."""
        matches = [
            {"digit": "4", "confidence": 0.95, "x": 25, "y": 0, "w": 20, "h": 30},  # Right, higher conf
            {"digit": "1", "confidence": 0.90, "x": 0, "y": 0, "w": 20, "h": 30},   # Left, lower conf
        ]

        filtered = detector._filter_overlapping_matches(matches)

        # Both should remain
        assert len(filtered) == 2

        # After sorting by x (done in _match_timer_region), should get "14" not "41"
        filtered.sort(key=lambda m: m["x"])
        digits = [m["digit"] for m in filtered]
        assert digits == ["1", "4"]

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

        # Should be binary
        assert np.all((preprocessed == 0) | (preprocessed == 255))

    def test_detect_with_debug(self, detector, mock_cropper):
        """Test debug detection returns additional info."""
        # Mock successful detection
        detector._match_timer_region = Mock(return_value=(45.0, 0.88, "045"))
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "0", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "4", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "5", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, timer_preprocessed, debug_info = detector.detect_with_debug(frame)

        assert result is not None
        assert result.time_seconds == 45.0
        assert isinstance(timer_preprocessed, np.ndarray)
        assert "timer_matches" in debug_info

    def test_detect_with_debug_empty_crop(self, detector, mock_cropper):
        """Test debug detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, timer_preprocessed, debug_info = detector.detect_with_debug(frame)

        assert result is None
        assert timer_preprocessed.size == 0
        assert debug_info == {}

    def test_time_parsing_mss_format(self, detector):
        """Test various m:ss format time parsing."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Test 0:00
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "0", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
        ])
        result = detector._match_timer_region(crop)
        assert result[0] == 0.0

        # Test 1:30
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "1", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "3", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
        ])
        result = detector._match_timer_region(crop)
        assert result[0] == 90.0  # 1:30 = 90 seconds

    def test_time_parsing_ssms_format(self, detector):
        """Test various ss.ms format time parsing."""
        crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Test 00.00
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "0", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.82, "x": 80, "y": 5, "w": 20, "h": 30},
        ])
        result = detector._match_timer_region(crop)
        assert result[0] == 0.0

        # Test 45.23
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "4", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "5", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
            {"digit": "2", "confidence": 0.85, "x": 55, "y": 5, "w": 20, "h": 30},
            {"digit": "3", "confidence": 0.82, "x": 80, "y": 5, "w": 20, "h": 30},
        ])
        result = detector._match_timer_region(crop)
        assert result[0] == 45.23


class TestTimerInfoValidation:
    """Tests for TimerInfo Pydantic model validation."""

    def test_valid_timer_info(self):
        """Test creating valid TimerInfo."""
        info = TimerInfo(
            time_seconds=45.67,
            confidence=0.88,
            raw_text="4567",
        )

        assert info.time_seconds == 45.67
        assert info.confidence == 0.88
        assert info.raw_text == "4567"

    def test_time_range_validation(self):
        """Test time must be in valid range (0-200)."""
        # Valid cases
        TimerInfo(time_seconds=0.0, confidence=0.9)
        TimerInfo(time_seconds=100.0, confidence=0.9)
        TimerInfo(time_seconds=200.0, confidence=0.9)

        # Invalid cases
        with pytest.raises(Exception):  # Pydantic ValidationError
            TimerInfo(time_seconds=-1.0, confidence=0.9)

        with pytest.raises(Exception):
            TimerInfo(time_seconds=201.0, confidence=0.9)

    def test_confidence_range_validation(self):
        """Test confidence must be in valid range (0-1)."""
        # Valid cases
        TimerInfo(time_seconds=45.0, confidence=0.0)
        TimerInfo(time_seconds=45.0, confidence=1.0)

        # Invalid cases
        with pytest.raises(Exception):
            TimerInfo(time_seconds=45.0, confidence=-0.1)

        with pytest.raises(Exception):
            TimerInfo(time_seconds=45.0, confidence=1.5)
