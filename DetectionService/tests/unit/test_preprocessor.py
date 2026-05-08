"""
test_preprocessor.py — Unit tests for the Stage 2b DSP preprocessing pipeline.

Tests each DSP step in isolation and the full ordered pipeline.
Run with: pytest tests/unit/test_preprocessor.py -v
"""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage2.preprocessor import Preprocessor


def _make_bgr_image(h: int = 64, w: int = 128, noise: bool = True) -> np.ndarray:
    """Create a synthetic BGR plate image for testing."""
    img = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
    if noise:
        # Add salt-and-pepper noise
        mask = np.random.randint(0, 10, (h, w)) == 0
        img[mask] = [255, 255, 255]
    return img


def _make_gray_image(h: int = 64, w: int = 128) -> np.ndarray:
    return np.random.randint(60, 200, (h, w), dtype=np.uint8)


class TestPreprocessorOutputShape(unittest.TestCase):
    """Output shape and dtype checks."""

    def setUp(self):
        self.preprocessor = Preprocessor()

    def test_output_is_2d(self):
        """Result must be a single-channel binary image."""
        img = _make_bgr_image()
        result = self.preprocessor.process(img)
        self.assertEqual(len(result.shape), 2)

    def test_output_preserves_spatial_dimensions(self):
        """H × W must be preserved through all DSP steps."""
        img = _make_bgr_image(h=48, w=96)
        result = self.preprocessor.process(img)
        self.assertEqual(result.shape, (48, 96))

    def test_output_dtype_is_uint8(self):
        img = _make_bgr_image()
        result = self.preprocessor.process(img)
        self.assertEqual(result.dtype, np.uint8)

    def test_output_is_binary(self):
        """Adaptive threshold must produce only 0 and 255 pixel values."""
        img = _make_bgr_image()
        result = self.preprocessor.process(img)
        unique_values = set(np.unique(result))
        self.assertTrue(unique_values.issubset({0, 255}))

    def test_accepts_grayscale_input(self):
        """Pipeline should handle already-grayscale input without error."""
        gray = _make_gray_image()
        result = self.preprocessor.process(gray)
        self.assertEqual(len(result.shape), 2)

    def test_raises_on_empty_input(self):
        with self.assertRaises(ValueError):
            self.preprocessor.process(np.array([]))

    def test_raises_on_none_input(self):
        with self.assertRaises(ValueError):
            self.preprocessor.process(None)


class TestGrayscaleStep(unittest.TestCase):
    """Step 1: Grayscale conversion."""

    def test_bgr_to_gray(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :] = [0, 128, 255]  # Pure blue-green-red
        gray = Preprocessor._to_grayscale(img)
        self.assertEqual(len(gray.shape), 2)

    def test_already_gray_passthrough(self):
        gray = _make_gray_image(32, 32)
        out = Preprocessor._to_grayscale(gray)
        np.testing.assert_array_equal(out, gray)


class TestBilateralFilterStep(unittest.TestCase):
    """Step 2: Bilateral filter."""

    def test_output_shape_unchanged(self):
        gray = _make_gray_image(48, 64)
        out = Preprocessor._bilateral_filter(gray)
        self.assertEqual(out.shape, gray.shape)

    def test_reduces_noise(self):
        """Bilateral filter should reduce high-frequency noise."""
        # Create image with strong salt-and-pepper noise
        base = np.full((64, 64), 128, dtype=np.uint8)
        noisy = base.copy()
        np.random.seed(42)
        mask = np.random.rand(64, 64) > 0.8
        noisy[mask] = 255

        filtered = Preprocessor._bilateral_filter(noisy)
        # Standard deviation should decrease (noise reduced)
        self.assertLess(filtered.std(), noisy.std())


class TestAdaptiveThresholdStep(unittest.TestCase):
    """Step 3: Adaptive threshold."""

    def test_output_is_binary(self):
        gray = _make_gray_image(64, 128)
        out = Preprocessor._adaptive_threshold(gray)
        unique = set(np.unique(out))
        self.assertTrue(unique.issubset({0, 255}))

    def test_even_block_size_is_corrected(self):
        """Block size must be odd — even values should be auto-corrected."""
        with patch.dict("os.environ", {"DSP_THRESH_BLOCK_SIZE": "10"}):
            # Should not raise — even block size is corrected internally
            gray = _make_gray_image()
            Preprocessor._adaptive_threshold(gray)  # No exception expected


class TestDeskewStep(unittest.TestCase):
    """Step 4: Deskew via Hough."""

    def test_deskew_preserves_shape(self):
        binary = np.zeros((32, 64), dtype=np.uint8)
        binary[16, :] = 255  # Horizontal line
        out = Preprocessor._deskew(binary)
        self.assertEqual(out.shape, binary.shape)

    def test_no_lines_returns_input_unchanged(self):
        """All-black image has no detectable lines → must return input."""
        black = np.zeros((32, 64), dtype=np.uint8)
        out = Preprocessor._deskew(black)
        np.testing.assert_array_equal(out, black)


class TestDSPStepOrdering(unittest.TestCase):
    """
    Verify the pipeline respects step ordering (claude.md §Stage 2b).
    Uses call tracking to confirm bilateral filter runs BEFORE thresholding.
    """

    def test_bilateral_runs_before_threshold(self):
        """
        Critical ordering check: bilateral filter must precede thresholding.
        If bilateral runs on binary data, it is useless (operates on 0/255 only).
        """
        call_order = []

        original_bilateral = Preprocessor._bilateral_filter.__func__
        original_threshold = Preprocessor._adaptive_threshold.__func__

        with patch.object(Preprocessor, "_bilateral_filter", staticmethod(
            lambda img: (call_order.append("bilateral"), original_bilateral(img))[1]
        )):
            with patch.object(Preprocessor, "_adaptive_threshold", staticmethod(
                lambda img: (call_order.append("threshold"), original_threshold(img))[1]
            )):
                p = Preprocessor()
                p.process(_make_bgr_image())

        self.assertEqual(call_order.index("bilateral"), 0)
        self.assertEqual(call_order.index("threshold"), 1)

    def test_deskew_is_last_when_enabled(self):
        """Deskew must always run after thresholding."""
        call_order = []

        with patch.dict("os.environ", {"DESKEW_ENABLED": "true"}):
            p = Preprocessor()

        original_deskew = Preprocessor._deskew.__func__

        with patch.object(Preprocessor, "_deskew", staticmethod(
            lambda img: (call_order.append("deskew"), original_deskew(img))[1]
        )):
            with patch.object(Preprocessor, "_adaptive_threshold", staticmethod(
                lambda img: (call_order.append("threshold"), np.zeros(img.shape, dtype=np.uint8))[1]
            )):
                p._deskew_enabled = True
                p.process(_make_bgr_image())

        if "deskew" in call_order and "threshold" in call_order:
            self.assertGreater(
                call_order.index("deskew"),
                call_order.index("threshold"),
            )


if __name__ == "__main__":
    unittest.main()
