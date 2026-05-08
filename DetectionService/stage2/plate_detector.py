"""
plate_detector.py — License plate bounding-box detector for Stage 2a.

Runs a plate-specific YOLO model on the vehicle crop produced by Stage 1
to locate the license plate sub-region before DSP and OCR.

If the model file is missing, the entire vehicle crop is returned as a
best-effort fallback so Stage 3 OCR can still attempt a read.
"""

import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Path to plate-specific YOLO weights
_PLATE_MODEL_PATH = os.environ.get("PLATE_MODEL_PATH", "models/plate.pt")

# Minimum confidence to accept a plate detection
_PLATE_CONF_THRESHOLD = float(os.environ.get("PLATE_CONF_THRESHOLD", "0.40"))

# Padding around the detected plate bbox (pixels)
_PLATE_CROP_PADDING = int(os.environ.get("PLATE_CROP_PADDING", "8"))


class PlateDetector:
    """
    Runs a YOLO model to detect license plate bounding boxes within a
    vehicle crop. Falls back to returning the full crop if the model is
    unavailable or no plate is detected.

    Lazy-loads the YOLO model on first use to keep container startup fast.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_path = _PLATE_MODEL_PATH
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, vehicle_crop_b64: str) -> tuple[np.ndarray, float]:
        """
        Detect the license plate in a base64-encoded vehicle crop.

        Args:
            vehicle_crop_b64: Base64 JPEG string from the Snapshot Event.

        Returns:
            (plate_crop, detection_confidence) where plate_crop is a BGR
            numpy array. confidence is 0.0 if using the full-frame fallback.
        """
        vehicle_crop = self._decode_b64(vehicle_crop_b64)
        if vehicle_crop is None:
            raise ValueError("Failed to decode vehicle crop image")

        if not self._loaded:
            self._load_model()

        if self._model is None:
            logger.debug("Plate model unavailable — using full vehicle crop as fallback")
            return vehicle_crop, 0.0

        return self._run_inference(vehicle_crop)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the YOLO model. Sets self._model = None on failure."""
        self._loaded = True
        try:
            from ultralytics import YOLO  # type: ignore
            if not os.path.exists(self._model_path):
                logger.warning(
                    "Plate model not found at %s. Using full-crop fallback.",
                    self._model_path,
                )
                return
            self._model = YOLO(self._model_path)
            logger.info("Plate YOLO model loaded from %s", self._model_path)
        except ImportError:
            logger.warning(
                "ultralytics not installed. Plate detection disabled. "
                "Install with: pip install ultralytics"
            )
        except Exception as exc:
            logger.error("Failed to load plate model: %s", exc)

    def _run_inference(self, vehicle_crop: np.ndarray) -> tuple[np.ndarray, float]:
        """Run YOLO inference and return the best plate crop."""
        try:
            results = self._model(vehicle_crop, verbose=False, conf=_PLATE_CONF_THRESHOLD)
            best_conf = 0.0
            best_bbox = None

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_bbox = box.xyxy[0].cpu().numpy()

            if best_bbox is not None:
                plate_crop = self._crop(vehicle_crop, best_bbox)
                if plate_crop is not None and plate_crop.size > 0:
                    return plate_crop, best_conf

        except Exception as exc:
            logger.error("Plate inference error: %s", exc)

        # Fallback: return full vehicle crop
        logger.debug("No plate detected — using full vehicle crop fallback")
        return vehicle_crop, 0.0

    @staticmethod
    def _crop(image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract and pad a crop from the image using a bbox array."""
        h, w = image.shape[:2]
        x1 = max(0, int(bbox[0]) - _PLATE_CROP_PADDING)
        y1 = max(0, int(bbox[1]) - _PLATE_CROP_PADDING)
        x2 = min(w, int(bbox[2]) + _PLATE_CROP_PADDING)
        y2 = min(h, int(bbox[3]) + _PLATE_CROP_PADDING)

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    @staticmethod
    def _decode_b64(b64_string: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string into a BGR numpy array."""
        try:
            raw_bytes = base64.b64decode(b64_string)
            buf = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return img
        except Exception as exc:
            logger.error("Image decode failed: %s", exc)
            return None
