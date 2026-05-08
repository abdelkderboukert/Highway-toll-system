"""
detector.py — Stage 1 main inference loop.

Connects to an RTSP camera stream (or local video file for dev/testing),
runs YOLOv8n object detection at the stream's native FPS, and fires a
Snapshot Event for every first-clean detection of each vehicle track_id.

Critical Rules (claude.md §Critical Rules):
    - Stage 1 must NEVER block on downstream state.
    - Always check the Tracking Ledger before enqueueing.
    - Never replicate Stage 1 horizontally for the same camera.

Run directly:
    RTSP_URL=rtsp://... CAMERA_ID=cam_01 python stage1/detector.py
"""

import logging
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import redis

from stage1.ledger import TrackingLedger
from stage1.snapshot import SnapshotPublisher
from stage1.speed_estimator import SpeedEstimator
from stage1.tracker import Tracker

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("stage1.detector")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RTSP_URL = os.environ.get("RTSP_URL", "")
CAMERA_ID = os.environ.get("CAMERA_ID", "cam_00_unknown")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "models/yolov8n.pt")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None

# YOLO classes that represent vehicles (COCO class IDs)
VEHICLE_CLASS_IDS = {
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}

# Minimum YOLO confidence to consider a detection
YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.50"))

# Minimum frames a track must be active before a snapshot is fired
MIN_TRACK_AGE_FRAMES = int(os.environ.get("MIN_TRACK_AGE_FRAMES", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# YOLO loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_yolo(model_path: str):
    """Load YOLO model. Returns None if unavailable (for CI/test environments)."""
    try:
        from ultralytics import YOLO  # type: ignore
        if not os.path.exists(model_path):
            logger.warning(
                "YOLO model not found at %s. Running in detection-stub mode.", model_path
            )
            return None
        model = YOLO(model_path)
        logger.info("YOLO model loaded: %s", model_path)
        return model
    except ImportError:
        logger.warning("ultralytics not installed — running in detection-stub mode")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Redis connection
# ─────────────────────────────────────────────────────────────────────────────

def _connect_redis() -> redis.Redis:
    """Connect to Redis with retry logic. Blocks until connected."""
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    while True:
        try:
            client.ping()
            logger.info("Redis connected: %s:%d", REDIS_HOST, REDIS_PORT)
            return client
        except redis.RedisError as exc:
            logger.warning("Redis not ready (%s) — retrying in 3s ...", exc)
            time.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
# Detection stub (when YOLO is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _stub_detections(frame: np.ndarray) -> list[tuple]:
    """
    Returns a synthetic detection for testing without a real YOLO model.
    Emits a single centred bounding box per frame.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = min(w, h) // 4
    bbox = (cx - size, cy - size, cx + size, cy + size)
    return [(bbox, 0.91)]


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_loop(
    cap: cv2.VideoCapture,
    model,
    tracker: Tracker,
    speed_estimator: SpeedEstimator,
    ledger: TrackingLedger,
    publisher: SnapshotPublisher,
) -> None:
    """
    Core detection loop. Runs until the stream ends or is interrupted.
    Never blocks on Redis, the worker, or any external service.
    """
    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    speed_estimator._fps = fps

    logger.info(
        "Detection loop started — camera=%s fps=%.1f", CAMERA_ID, fps
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Stream ended or unreadable — stopping.")
            break

        frame_number += 1

        # ── Run detection ─────────────────────────────────────────────
        if model is not None:
            raw_results = model(
                frame, verbose=False, conf=YOLO_CONF_THRESHOLD, classes=list(VEHICLE_CLASS_IDS)
            )
            detections = []
            for result in raw_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASS_IDS:
                        continue
                    bbox = tuple(box.xyxy[0].cpu().numpy().tolist())
                    conf = float(box.conf[0])
                    detections.append((bbox, conf))
        else:
            detections = _stub_detections(frame)

        # ── Update tracker ────────────────────────────────────────────
        tracks = tracker.update(detections)

        # ── Per-track processing ──────────────────────────────────────
        for track in tracks:
            # Speed estimate — updated every frame regardless of ledger
            speed_kmh = speed_estimator.update(
                track.track_id, frame_number, track.bbox
            )

            # Deduplicate: skip if already in ledger
            if ledger.is_seen(track.track_id):
                continue

            # Require minimum track stability before firing snapshot
            if track.hits < MIN_TRACK_AGE_FRAMES:
                continue

            # Fire snapshot — fire-and-forget (returns immediately)
            success = publisher.push(
                frame=frame,
                track_id=track.track_id,
                frame_number=frame_number,
                bbox=track.bbox,
                speed_kmh=speed_kmh,
                confidence=track.confidence,
            )

            if success:
                ledger.mark_seen(track.track_id)

        # ── Evict stale tracks from speed estimator ───────────────────
        active_ids = {t.track_id for t in tracks}
        for old_id in list(speed_estimator._history.keys()):
            if old_id not in active_ids:
                speed_estimator.evict(old_id)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RTSP_URL:
        logger.error("RTSP_URL environment variable is not set. Exiting.")
        sys.exit(1)

    logger.info("Starting Stage 1 — camera=%s stream=%s", CAMERA_ID, RTSP_URL)

    # Open stream
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logger.error("Failed to open stream: %s", RTSP_URL)
        sys.exit(1)

    # Initialise subsystems
    redis_client = _connect_redis()
    model = _load_yolo(YOLO_MODEL_PATH)
    tracker = Tracker()
    speed_estimator = SpeedEstimator()
    ledger = TrackingLedger(redis_client, CAMERA_ID)
    publisher = SnapshotPublisher(redis_client, CAMERA_ID)

    try:
        run_detection_loop(cap, model, tracker, speed_estimator, ledger, publisher)
    except KeyboardInterrupt:
        logger.info("Stage 1 interrupted by user.")
    finally:
        cap.release()
        redis_client.close()
        logger.info("Stage 1 shut down cleanly.")


if __name__ == "__main__":
    main()
