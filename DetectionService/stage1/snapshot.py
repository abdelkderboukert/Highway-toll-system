"""
snapshot.py — Snapshot Event builder and queue pusher for Stage 1.

This is the fire-and-forget boundary between Stage 1 and the Buffer.
After calling push(), Stage 1 never waits for a response.

Critical Rule (claude.md §Critical Rules):
    Stage 1 must never block on downstream state. push() returns immediately.
    Any Redis failure is logged and the task is silently dropped — the system
    prefers a missed detection over a stalled camera process.

Data Contract (claude.md §Data Contracts):
    {
        "track_id":       int,
        "camera_id":      str,
        "timestamp_utc":  str,   # ISO 8601
        "frame_number":   int,
        "speed_kmh":      float,
        "confidence":     float,
        "image_b64":      str,   # Base64 JPEG crop
    }
"""

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import redis

logger = logging.getLogger(__name__)

# Padding (pixels) added to each side of the bounding box crop
_CROP_PADDING = int(os.environ.get("SNAPSHOT_CROP_PADDING", "20"))

# JPEG quality for the vehicle crop (85% per spec)
_JPEG_QUALITY = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "85"))

# Redis queue key
_QUEUE_KEY = os.environ.get("REDIS_QUEUE_KEY", "pipeline:tasks")


class SnapshotPublisher:
    """
    Builds a Snapshot Event from a detection and pushes it to the Redis
    FIFO queue using LPUSH (consumers use BRPOP → FIFO order).

    Instantiate once per Stage 1 process and reuse across frames.
    """

    def __init__(self, redis_client: redis.Redis, camera_id: str) -> None:
        self._redis = redis_client
        self._camera_id = camera_id
        self._queue_key = _QUEUE_KEY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        frame: np.ndarray,
        track_id: int,
        frame_number: int,
        bbox: tuple[float, float, float, float],
        speed_kmh: float,
        confidence: float,
    ) -> bool:
        """
        Build a Snapshot Event and push it to the Redis queue.

        Returns True on success, False on any failure.
        Never raises — Stage 1 must not crash on egress errors.
        """
        try:
            crop_b64 = self._crop_and_encode(frame, bbox)
            if crop_b64 is None:
                logger.warning(
                    "Snapshot skipped — invalid crop for track_id=%s frame=%s",
                    track_id,
                    frame_number,
                )
                return False

            payload = {
                "track_id": int(track_id),
                "camera_id": self._camera_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "frame_number": int(frame_number),
                "speed_kmh": float(round(speed_kmh, 2)),
                "confidence": float(round(confidence, 4)),
                "image_b64": crop_b64,
            }

            self._redis.lpush(self._queue_key, json.dumps(payload))

            logger.debug(
                "Snapshot pushed: track_id=%s camera=%s speed=%.1f km/h q=%s",
                track_id,
                self._camera_id,
                speed_kmh,
                self._queue_key,
            )
            return True

        except redis.RedisError as exc:
            logger.error(
                "Redis push failed for track_id=%s: %s", track_id, exc
            )
            return False
        except Exception as exc:
            logger.exception(
                "Unexpected error in snapshot push for track_id=%s: %s",
                track_id,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_and_encode(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> Optional[str]:
        """
        Crop the vehicle region from the frame, pad it, encode as JPEG,
        and return a Base64 string.

        Returns None if the crop is invalid (zero area, out-of-bounds).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Apply padding — clamped to frame boundaries
        x1 = max(0, int(x1) - _CROP_PADDING)
        y1 = max(0, int(y1) - _CROP_PADDING)
        x2 = min(w, int(x2) + _CROP_PADDING)
        y2 = min(h, int(y2) + _CROP_PADDING)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
        success, buf = cv2.imencode(".jpg", crop, encode_params)
        if not success:
            return None

        return base64.b64encode(buf.tobytes()).decode("utf-8")
