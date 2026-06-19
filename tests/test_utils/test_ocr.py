"""Unit tests for OCR utility."""

from __future__ import annotations
from unittest.mock import patch, Mock
import pytest
import numpy as np

from valoscribe.utils.ocr import OCREngine, PSM


class TestOCREngine:
    """Tests for OCREngine class."""

    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance."""
        return OCREngine(lang="eng")

    @pytest.fixture
    def mock_pytesseract(self):
        """Mock pytesseract.image_to_data."""
        with patch("valoscribe.utils.ocr.pytesseract.image_to_data") as mock:
            yield mock

    def test_init(self, ocr_engine):
        """Test OCR engine initialization."""
        assert ocr_engine.lang == "eng"

    def test_read_single_line_success(self, ocr_engine, mock_pytesseract):
        """Test reading single line of text."""
        # Mock OCR output
        mock_pytesseract.return_value = {
            "text": ["Round", "5"],
            "conf": [95, 98],
        }

        image = np.zeros((20, 100, 3), dtype=np.uint8)
        text, confidence = ocr_engine.read_single_line(image, preprocess=False)

        assert text == "Round 5"
        assert 0.9 < confidence < 1.0  # Average of 95 and 98

    def test_read_single_line_with_whitelist(self, ocr_engine, mock_pytesseract):
        """Test reading with character whitelist."""
        mock_pytesseract.return_value = {
            "text": ["123"],
            "conf": [90],
        }

        image = np.zeros((20, 100, 3), dtype=np.uint8)
        text, confidence = ocr_engine.read_single_line(
            image, whitelist="0123456789", preprocess=False
        )

        assert text == "123"
        # Verify whitelist was passed in config
        config = mock_pytesseract.call_args[1]["config"]
        assert "tessedit_char_whitelist" in config
        assert "0123456789" in config

    def test_read_single_line_empty_result(self, ocr_engine, mock_pytesseract):
        """Test handling empty OCR result."""
        mock_pytesseract.return_value = {
            "text": ["", ""],
            "conf": [-1, -1],
        }

        image = np.zeros((20, 100, 3), dtype=np.uint8)
        text, confidence = ocr_engine.read_single_line(image, preprocess=False)

        assert text == ""
        assert confidence == 0.0

    def test_read_digits_success(self, ocr_engine, mock_pytesseract):
        """Test reading digits from image."""
        mock_pytesseract.return_value = {
            "text": ["12"],
            "conf": [95],
        }

        image = np.zeros((20, 50, 3), dtype=np.uint8)
        text, confidence = ocr_engine.read_digits(image, max_digits=2, preprocess=False)

        assert text == "12"
        assert confidence > 0.9

    def test_read_digits_exceeds_max(self, ocr_engine, mock_pytesseract):
        """Test reading digits when result exceeds max."""
        mock_pytesseract.return_value = {
            "text": ["12345"],
            "conf": [95],
        }

        image = np.zeros((20, 50, 3), dtype=np.uint8)
        text, confidence = ocr_engine.read_digits(image, max_digits=2, preprocess=False)

        # Should still return the result but log warning
        assert text == "12345"

    def test_read_single_char_success(self, ocr_engine, mock_pytesseract):
        """Test reading single character."""
        mock_pytesseract.return_value = {
            "text": ["A"],
            "conf": [98],
        }

        image = np.zeros((20, 20, 3), dtype=np.uint8)
        char, confidence = ocr_engine.read_single_char(image, preprocess=False)

        assert char == "A"
        assert confidence == 0.98

    def test_read_single_char_empty(self, ocr_engine, mock_pytesseract):
        """Test reading single character with empty result."""
        mock_pytesseract.return_value = {
            "text": [""],
            "conf": [-1],
        }

        image = np.zeros((20, 20, 3), dtype=np.uint8)
        char, confidence = ocr_engine.read_single_char(image, preprocess=False)

        assert char == ""
        assert confidence == 0.0

    def test_read_multi_line_success(self, ocr_engine, mock_pytesseract):
        """Test reading multiple lines of text."""
        mock_pytesseract.return_value = {
            "text": ["Line", "One", "Line", "Two"],
            "conf": [95, 96, 94, 93],
            "line_num": [0, 0, 1, 1],
        }

        image = np.zeros((50, 100, 3), dtype=np.uint8)
        lines, confidence = ocr_engine.read_multi_line(image, preprocess=False)

        assert len(lines) == 2
        assert lines[0] == "Line One"
        assert lines[1] == "Line Two"
        assert 0.9 < confidence < 1.0

    def test_build_config_basic(self, ocr_engine):
        """Test building basic Tesseract config."""
        config = ocr_engine._build_config(PSM.SINGLE_LINE, whitelist=None)

        assert "--psm 7" in config
        assert "--oem 1" in config

    def test_build_config_with_whitelist(self, ocr_engine):
        """Test building config with character whitelist."""
        config = ocr_engine._build_config(PSM.SINGLE_LINE, whitelist="0123456789")

        assert "--psm 7" in config
        assert "tessedit_char_whitelist" in config
        assert "0123456789" in config


class TestPreprocessing:
    """Tests for preprocessing functions."""

    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance."""
        return OCREngine()

    def test_preprocess_for_digits(self, ocr_engine):
        """Test preprocessing for digit recognition."""
        # Create test image with some content
        image = np.random.randint(0, 255, (20, 100, 3), dtype=np.uint8)

        result = ocr_engine.preprocess_for_digits(image, scale=2)

        # Should be upscaled and binary
        assert result.shape[0] == 40  # 20 * 2
        assert result.shape[1] == 200  # 100 * 2
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8
        # Binary image should only have 0 and 255
        unique_values = np.unique(result)
        assert len(unique_values) <= 2

    def test_preprocess_for_digits_grayscale_input(self, ocr_engine):
        """Test preprocessing with grayscale input."""
        image = np.random.randint(0, 255, (20, 100), dtype=np.uint8)

        result = ocr_engine.preprocess_for_digits(image, scale=1)

        assert len(result.shape) == 2
        assert result.shape == (20, 100)

    def test_preprocess_for_text(self, ocr_engine):
        """Test preprocessing for general text."""
        image = np.random.randint(0, 255, (20, 100, 3), dtype=np.uint8)

        result = ocr_engine.preprocess_for_text(image, scale=2)

        # Should be upscaled and binary
        assert result.shape[0] == 40
        assert result.shape[1] == 200
        assert len(result.shape) == 2  # Grayscale

    def test_preprocess_invert(self, ocr_engine):
        """Test image inversion."""
        image = np.array([[0, 128, 255]], dtype=np.uint8)

        result = ocr_engine.preprocess_invert(image)

        assert result[0, 0] == 255
        assert result[0, 1] == 127
        assert result[0, 2] == 0

    def test_preprocess_denoise_color(self, ocr_engine):
        """Test denoising on color image."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        result = ocr_engine.preprocess_denoise(image, strength=5)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_preprocess_denoise_grayscale(self, ocr_engine):
        """Test denoising on grayscale image."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        result = ocr_engine.preprocess_denoise(image, strength=5)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
