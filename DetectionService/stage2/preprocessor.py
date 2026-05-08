"""
preprocessor.py — DSP preprocessing pipeline for Stage 2b.

Applies the mandatory image cleaning pipeline to a license plate crop
before it is passed to the OCR engine in Stage 3.

CRITICAL: Steps must run in this exact order (claude.md §Stage 2):
    1. Grayscale conversion
    2. Bilateral filter      (edge-preserving noise reduction)
    3. Adaptive thresholding (local binarization — corrects uneven lighting)
    4. Deskew via Hough     (optional — enabled by DESKEW_ENABLED=true)

Do NOT reorder steps. Bilateral filter on binary data is useless.
Thresholding after grayscale is load-bearing for OCR quality.
"""

import logging
import math
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

_DESKEW_ENABLED = os.environ.get("DESKEW_ENABLED", "false").lower() == "true"

# Bilateral filter parameters
_BILATERAL_D = int(os.environ.get("DSP_BILATERAL_D", "9"))
_BILATERAL_SIGMA_COLOR = float(os.environ.get("DSP_BILATERAL_SIGMA_COLOR", "75"))
_BILATERAL_SIGMA_SPACE = float(os.environ.get("DSP_BILATERAL_SIGMA_SPACE", "75"))

# Adaptive threshold parameters
_THRESH_BLOCK_SIZE = int(os.environ.get("DSP_THRESH_BLOCK_SIZE", "11"))
_THRESH_C = float(os.environ.get("DSP_THRESH_C", "2"))

# Hough deskew parameters
_DESKEW_MAX_ANGLE = float(os.environ.get("DSP_DESKEW_MAX_ANGLE", "15"))
_DESKEW_THRESHOLD1 = int(os.environ.get("DSP_DESKEW_CANNY_T1", "50"))
_DESKEW_THRESHOLD2 = int(os.environ.get("DSP_DESKEW_CANNY_T2", "150"))


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Stateless DSP preprocessing pipeline.

    Call process(plate_crop) to receive a cleaned, binarized image
    ready for OCR.
    """

    def __init__(self) -> None:
        self._deskew_enabled = _DESKEW_ENABLED
        logger.info(
            "Preprocessor initialized — deskew=%s", self._deskew_enabled
        )

    def process(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Run the full DSP pipeline on a plate crop.

        Args:
            plate_crop: BGR image array (H × W × 3) of the license plate region.

        Returns:
            Binary (uint8) image ready for OCR. Shape: (H × W), single channel.
        """
        if plate_crop is None or plate_crop.size == 0:
            raise ValueError("plate_crop is empty or None")

        # ── Step 1: Grayscale conversion ──────────────────────────────────
        gray = self._to_grayscale(plate_crop)
        logger.debug("DSP step 1 done — grayscale shape=%s", gray.shape)

        # ── Step 2: Bilateral filter ──────────────────────────────────────
        # MUST run before thresholding (operates on continuous-tone data).
        filtered = self._bilateral_filter(gray)
        logger.debug("DSP step 2 done — bilateral filter applied")

        # ── Step 3: Adaptive threshold ────────────────────────────────────
        binary = self._adaptive_threshold(filtered)
        logger.debug("DSP step 3 done — adaptive threshold applied")

        # ── Step 4: Deskew (optional) ─────────────────────────────────────
        if self._deskew_enabled:
            binary = self._deskew(binary)
            logger.debug("DSP step 4 done — deskew applied")

        return binary

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Step 1: Convert BGR to single-channel grayscale."""
        if len(image.shape) == 2:
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _bilateral_filter(gray: np.ndarray) -> np.ndarray:
        """
        Step 2: Edge-preserving noise reduction.

        Bilateral filter blurs homogeneous regions while keeping edges
        sharp — essential for crisp character boundaries in OCR.
        """
        return cv2.bilateralFilter(
            gray,
            d=_BILATERAL_D,
            sigmaColor=_BILATERAL_SIGMA_COLOR,
            sigmaSpace=_BILATERAL_SIGMA_SPACE,
        )

    @staticmethod
    def _adaptive_threshold(filtered: np.ndarray) -> np.ndarray:
        """
        Step 3: Local binarization.

        Adaptive (Gaussian-weighted) thresholding handles uneven lighting
        across the plate (e.g., headlight glare on one side).
        Block size must be odd — enforce it.
        """
        block_size = _THRESH_BLOCK_SIZE
        if block_size % 2 == 0:
            block_size += 1

        return cv2.adaptiveThreshold(
            filtered,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=_THRESH_C,
        )

    @staticmethod
    def _deskew(binary: np.ndarray) -> np.ndarray:
        """
        Step 4: Correct plate tilt using Hough Line Transform.

        Detects dominant line angles in the binarized image and rotates
        to correct skew within ±MAX_ANGLE degrees.
        Returns the input unchanged if no dominant angle is found.
        """
        edges = cv2.Canny(binary, _DESKEW_THRESHOLD1, _DESKEW_THRESHOLD2)
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=60)

        if lines is None:
            return binary

        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert theta (0..π) to angle in degrees relative to horizontal
            angle_deg = math.degrees(theta) - 90.0
            if abs(angle_deg) <= _DESKEW_MAX_ANGLE:
                angles.append(angle_deg)

        if not angles:
            return binary

        median_angle = float(np.median(angles))
        logger.debug("Deskew angle: %.2f°", median_angle)

        h, w = binary.shape
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, median_angle, scale=1.0)
        deskewed = cv2.warpAffine(
            binary,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return deskewed
