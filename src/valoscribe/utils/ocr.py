"""
OCR utility for text extraction from images.

This module provides a wrapper around pytesseract with various configurations
and preprocessing utilities optimized for different text extraction scenarios.
"""

from __future__ import annotations
from typing import Optional
from enum import Enum

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class PSM(Enum):
    """Tesseract Page Segmentation Modes."""

    OSD_ONLY = 0  # Orientation and script detection only
    AUTO_OSD = 1  # Automatic page segmentation with OSD
    AUTO = 3  # Fully automatic page segmentation (default)
    SINGLE_COLUMN = 4  # Assume a single column of text
    SINGLE_BLOCK_VERT = 5  # Assume a single uniform block of vertically aligned text
    SINGLE_BLOCK = 6  # Assume a single uniform block of text
    SINGLE_LINE = 7  # Treat the image as a single text line
    SINGLE_WORD = 8  # Treat the image as a single word
    CIRCLE_WORD = 9  # Treat the image as a single word in a circle
    SINGLE_CHAR = 10  # Treat the image as a single character
    SPARSE_TEXT = 11  # Sparse text, find as much text as possible
    SPARSE_TEXT_OSD = 12  # Sparse text with OSD
    RAW_LINE = 13  # Raw line, bypass Tesseract-specific algorithms


class OCREngine:
    """
    OCR engine wrapper for pytesseract with various configurations.

    This class provides convenient methods for different OCR scenarios
    with appropriate preprocessing and configuration.
    """

    def __init__(self, lang: str = "eng"):
        """
        Initialize OCR engine.

        Args:
            lang: Tesseract language (default: "eng")
        """
        self.lang = lang
        log.info(f"OCR engine initialized with language: {lang}")

    def read_single_line(
        self,
        image: np.ndarray,
        whitelist: Optional[str] = None,
        preprocess: bool = True,
    ) -> tuple[str, float]:
        """
        Read a single line of text from an image.

        Args:
            image: Input image (grayscale or color)
            whitelist: Characters to whitelist (e.g., "0123456789" for digits only)
            preprocess: Apply preprocessing for better accuracy

        Returns:
            Tuple of (text, confidence) where confidence is 0-1
        """
        if preprocess:
            image = self.preprocess_for_text(image)

        config = self._build_config(PSM.SINGLE_LINE, whitelist)
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=config, output_type=Output.DICT
        )

        # Extract text and confidence
        text = " ".join([t for t in data["text"] if t.strip()])
        confidences = [
            conf for conf, t in zip(data["conf"], data["text"]) if t.strip() and conf != -1
        ]
        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

        log.debug(f"OCR single line: '{text}' (confidence: {avg_confidence:.2f})")
        return text.strip(), float(avg_confidence)

    def read_digits(
        self,
        image: np.ndarray,
        max_digits: Optional[int] = None,
        preprocess: bool = True,
    ) -> tuple[str, float]:
        """
        Read digits from an image.

        Args:
            image: Input image (grayscale or color)
            max_digits: Maximum number of expected digits
            preprocess: Apply preprocessing optimized for digits

        Returns:
            Tuple of (text, confidence) where confidence is 0-1
        """
        if preprocess:
            image = self.preprocess_for_digits(image)

        # Use single line mode and digit whitelist
        config = self._build_config(PSM.SINGLE_LINE, "0123456789")
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=config, output_type=Output.DICT
        )

        # Extract text and confidence
        text = " ".join([t for t in data["text"] if t.strip()])
        confidences = [
            conf for conf, t in zip(data["conf"], data["text"]) if t.strip() and conf != -1
        ]
        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

        log.debug(f"OCR digits: '{text}' (confidence: {avg_confidence:.2f})")

        # Validate result
        if max_digits and len(text) > max_digits:
            log.warning(f"OCR returned {len(text)} digits, expected max {max_digits}: '{text}'")

        return text.strip(), float(avg_confidence)

    def read_single_char(
        self,
        image: np.ndarray,
        whitelist: Optional[str] = None,
        preprocess: bool = True,
    ) -> tuple[str, float]:
        """
        Read a single character from an image.

        Args:
            image: Input image (grayscale or color)
            whitelist: Characters to whitelist
            preprocess: Apply preprocessing

        Returns:
            Tuple of (character, confidence) where confidence is 0-1
        """
        if preprocess:
            image = self.preprocess_for_text(image)

        config = self._build_config(PSM.SINGLE_CHAR, whitelist)
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=config, output_type=Output.DICT
        )

        # Extract first character
        for text, conf in zip(data["text"], data["conf"]):
            if text.strip() and conf != -1:
                confidence = conf / 100.0
                log.debug(f"OCR single char: '{text}' (confidence: {confidence:.2f})")
                return text.strip()[0], float(confidence)

        return "", 0.0

    def read_multi_line(
        self,
        image: np.ndarray,
        whitelist: Optional[str] = None,
        preprocess: bool = True,
    ) -> tuple[list[str], float]:
        """
        Read multiple lines of text from an image.

        Args:
            image: Input image (grayscale or color)
            whitelist: Characters to whitelist
            preprocess: Apply preprocessing

        Returns:
            Tuple of (list of lines, avg_confidence) where confidence is 0-1
        """
        if preprocess:
            image = self.preprocess_for_text(image)

        config = self._build_config(PSM.AUTO, whitelist)
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=config, output_type=Output.DICT
        )

        # Group text by line number
        lines: dict[int, list[str]] = {}
        confidences: list[float] = []

        for i, text in enumerate(data["text"]):
            if text.strip():
                line_num = data["line_num"][i]
                conf = data["conf"][i]

                if line_num not in lines:
                    lines[line_num] = []

                lines[line_num].append(text)

                if conf != -1:
                    confidences.append(conf)

        # Build lines
        result = [" ".join(lines[line_num]) for line_num in sorted(lines.keys())]
        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

        log.debug(f"OCR multi-line: {len(result)} lines (confidence: {avg_confidence:.2f})")
        return result, float(avg_confidence)

    @staticmethod
    def preprocess_for_digits(image: np.ndarray, scale: int = 3) -> np.ndarray:
        """
        Preprocess image for digit recognition.

        Optimizations:
        - Upscale for better recognition
        - Convert to grayscale
        - Apply binary threshold (Otsu's method)
        - Optional morphological operations

        Args:
            image: Input image
            scale: Upscaling factor (default: 3)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Upscale for better OCR
        if scale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Optional: denoise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    @staticmethod
    def preprocess_for_text(image: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Preprocess image for general text recognition.

        Args:
            image: Input image
            scale: Upscaling factor (default: 2)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Upscale
        if scale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Apply adaptive thresholding for better handling of varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return binary

    @staticmethod
    def preprocess_invert(image: np.ndarray) -> np.ndarray:
        """
        Invert image colors (useful for white text on dark background).

        Args:
            image: Input image

        Returns:
            Inverted image
        """
        return cv2.bitwise_not(image)

    @staticmethod
    def preprocess_denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Apply denoising to image.

        Args:
            image: Input image
            strength: Denoising strength

        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    def _build_config(self, psm: PSM, whitelist: Optional[str] = None) -> str:
        """
        Build Tesseract configuration string.

        Args:
            psm: Page segmentation mode
            whitelist: Characters to whitelist

        Returns:
            Configuration string for Tesseract
        """
        config_parts = [f"--psm {psm.value}"]

        if whitelist:
            # Escape special characters in whitelist
            escaped = whitelist.replace("\\", "\\\\").replace("'", "\\'")
            config_parts.append(f"-c tessedit_char_whitelist='{escaped}'")

        # Add OEM (OCR Engine Mode) - use LSTM neural network
        config_parts.append("--oem 1")

        return " ".join(config_parts)
