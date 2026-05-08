"""
ocr.py — OCR engine wrapper for Stage 3.

Wraps Tesseract, EasyOCR, or a CRNN-based engine depending on the
OCR_ENGINE environment variable.

Character whitelist: A-Z and 0-9 only (plate standard).
Any result below OCR_MIN_CONFIDENCE is returned with confidence=0.0
so the aggregator can route it to the Dead Letter Queue.
"""

import logging
import os
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_OCR_ENGINE = os.environ.get("OCR_ENGINE", "tesseract").lower()
_OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.7"))

# Only alphanumeric characters are valid on a license plate
_PLATE_CHAR_PATTERN = re.compile(r"[^A-Z0-9]")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

class OCRResult:
    __slots__ = ("text", "confidence")

    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"OCRResult(text={self.text!r}, confidence={self.confidence:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Engine implementations
# ─────────────────────────────────────────────────────────────────────────────

class _TesseractEngine:
    """Tesseract OCR via pytesseract."""

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore
            self._pytesseract = pytesseract
            logger.info("OCR engine: Tesseract")
        except ImportError:
            raise RuntimeError(
                "pytesseract not installed. Run: pip install pytesseract "
                "and ensure Tesseract binary is on PATH."
            )

    def read(self, plate_image: np.ndarray) -> OCRResult:
        import pytesseract
        from PIL import Image

        pil_img = Image.fromarray(plate_image)

        # PSM 7 = treat image as single line of text (ideal for plates)
        config = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data = pytesseract.image_to_data(
            pil_img, config=config, output_type=pytesseract.Output.DICT
        )

        best_text = ""
        best_conf = 0.0

        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i]) / 100.0  # Tesseract gives 0–100
            if conf > best_conf:
                best_conf = conf
                best_text = text

        cleaned = _clean_plate_text(best_text)
        return OCRResult(text=cleaned, confidence=best_conf)


class _EasyOCREngine:
    """EasyOCR engine — GPU-accelerated when available."""

    def __init__(self) -> None:
        try:
            import easyocr  # type: ignore
            # Only English alphanumeric
            self._reader = easyocr.Reader(["en"], gpu=self._has_gpu())
            logger.info("OCR engine: EasyOCR (gpu=%s)", self._has_gpu())
        except ImportError:
            raise RuntimeError(
                "easyocr not installed. Run: pip install easyocr"
            )

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def read(self, plate_image: np.ndarray) -> OCRResult:
        results = self._reader.readtext(
            plate_image,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
        )

        if not results:
            return OCRResult(text="", confidence=0.0)

        # Pick the result with highest confidence
        best = max(results, key=lambda r: r[2])
        _bbox, text, confidence = best

        cleaned = _clean_plate_text(text.upper())
        return OCRResult(text=cleaned, confidence=float(confidence))


class _CRNNEngine:
    """
    Placeholder CRNN wrapper.

    Replace this stub with your actual CRNN model loading and inference.
    Falls back to Tesseract until a real model is provided.
    """

    def __init__(self) -> None:
        logger.warning(
            "CRNN engine stub active — falling back to Tesseract. "
            "Implement _CRNNEngine.read() with your trained model."
        )
        self._fallback = _TesseractEngine()

    def read(self, plate_image: np.ndarray) -> OCRResult:
        return self._fallback.read(plate_image)


# ─────────────────────────────────────────────────────────────────────────────
# Public facade
# ─────────────────────────────────────────────────────────────────────────────

class OCRProcessor:
    """
    Selects and wraps the configured OCR backend.

    Applies the minimum confidence filter per claude.md spec:
        Any read below OCR_MIN_CONFIDENCE is returned with
        confidence=0.0 so the aggregator sends it to the DLQ.
    """

    def __init__(self) -> None:
        self._engine = self._build_engine()
        self._min_confidence = _OCR_MIN_CONFIDENCE

    def _build_engine(self):
        engines = {
            "tesseract": _TesseractEngine,
            "easyocr": _EasyOCREngine,
            "crnn": _CRNNEngine,
        }
        engine_cls = engines.get(_OCR_ENGINE)
        if engine_cls is None:
            logger.warning(
                "Unknown OCR_ENGINE=%s; defaulting to tesseract", _OCR_ENGINE
            )
            engine_cls = _TesseractEngine

        try:
            return engine_cls()
        except RuntimeError as exc:
            logger.error("OCR engine init failed: %s — falling back to stub", exc)
            return _CRNNEngine()  # CRNN stub will further fall back to tesseract

    def read(self, plate_image: np.ndarray) -> OCRResult:
        """
        Run OCR on a preprocessed plate image.

        Returns:
            OCRResult. If confidence < OCR_MIN_CONFIDENCE, the result still
            contains the raw text but confidence is set to 0.0 to signal
            the aggregator to route this task to the DLQ.
        """
        result = self._engine.read(plate_image)

        if result.confidence < self._min_confidence:
            logger.debug(
                "OCR confidence %.3f below threshold %.3f — marking for DLQ (text=%r)",
                result.confidence,
                self._min_confidence,
                result.text,
            )
            result.confidence = 0.0  # Signal for DLQ routing

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean_plate_text(raw: str) -> str:
    """Strip non-alphanumeric characters and upper-case the result."""
    return _PLATE_CHAR_PATTERN.sub("", raw.upper()).strip()
