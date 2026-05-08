"""
speed_estimator.py — Vector displacement → km/h speed estimator for Stage 1.

Converts pixel-space centroid displacement between consecutive frames into
real-world speed (km/h) using a pre-calibrated homography matrix.

Speed is only emitted after a minimum of MIN_TRACK_FRAMES stable frames
to avoid noisy readings on first detection.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum frames before a speed estimate is considered reliable
MIN_TRACK_FRAMES = 3


class SpeedEstimator:
    """
    Estimates vehicle speed from bounding-box centroid displacement.

    Requires a homography matrix that maps pixel coordinates to real-world
    metres. Load via load_homography(path). Without a valid matrix, speed
    defaults to 0.0 — the process does NOT crash.

    Per claude.md:
        "Speed estimates will be meaningless (but won't crash) if this file
         is missing — speed will default to 0.0."
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._homography: Optional[np.ndarray] = None

        # track_id → deque of (frame_number, pixel_centroid)
        self._history: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)

        # Load homography if HOMOGRAPHY_PATH is set
        homography_path = os.environ.get("HOMOGRAPHY_PATH", "")
        if homography_path:
            self.load_homography(homography_path)

    # ------------------------------------------------------------------
    # Homography loading
    # ------------------------------------------------------------------

    def load_homography(self, path: str) -> bool:
        """
        Load a homography matrix from a JSON file produced by
        scripts/calibrate_homography.py.

        Returns True on success, False on failure (speed falls back to 0.0).
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            matrix = np.array(data["homography_matrix"], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"Expected 3×3 matrix, got {matrix.shape}")
            self._homography = matrix
            logger.info("Homography matrix loaded from %s", path)
            return True
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load homography from %s (%s). Speed will default to 0.0.",
                path,
                exc,
            )
            self._homography = None
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, track_id: int, frame_number: int, bbox: tuple[float, float, float, float]) -> float:
        """
        Update the track history and return the current speed estimate in km/h.

        Args:
            track_id:     MOT-assigned vehicle identity.
            frame_number: Current frame index in the stream session.
            bbox:         Bounding box as (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            Estimated speed in km/h. Returns 0.0 if fewer than MIN_TRACK_FRAMES
            have been seen, or if the homography matrix is not loaded.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        centroid = (cx, cy)

        history = self._history[track_id]
        history.append((frame_number, centroid))

        # Keep only the last 10 frames to bound memory usage
        if len(history) > 10:
            history.pop(0)

        if len(history) < MIN_TRACK_FRAMES:
            return 0.0

        return self._compute_speed(history)

    def evict(self, track_id: int) -> None:
        """Remove a track_id from history when it leaves the frame."""
        self._history.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_speed(self, history: list) -> float:
        """
        Compute speed from the earliest and latest centroid in the history.

        Without a homography, pixel distance is meaningless — return 0.0.
        """
        if self._homography is None:
            return 0.0

        first_frame, first_centroid = history[0]
        last_frame, last_centroid = history[-1]

        frame_delta = last_frame - first_frame
        if frame_delta == 0:
            return 0.0

        # Map pixel centroids → real-world metres via homography
        p1_world = self._pixel_to_world(first_centroid)
        p2_world = self._pixel_to_world(last_centroid)

        if p1_world is None or p2_world is None:
            return 0.0

        # Euclidean distance in metres
        dx = p2_world[0] - p1_world[0]
        dy = p2_world[1] - p1_world[1]
        distance_m = float(np.sqrt(dx**2 + dy**2))

        # Time elapsed in seconds
        time_s = frame_delta / self._fps

        # Speed in m/s → km/h
        speed_kmh = (distance_m / time_s) * 3.6
        return round(speed_kmh, 2)

    def _pixel_to_world(self, pixel: tuple[float, float]) -> Optional[tuple[float, float]]:
        """Apply the homography matrix to map a pixel point to world metres."""
        try:
            src = np.array([[[pixel[0], pixel[1]]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, self._homography)
            return float(dst[0][0][0]), float(dst[0][0][1])
        except cv2.error as exc:
            logger.debug("perspectiveTransform failed: %s", exc)
            return None
