"""
test_pipeline.py — End-to-end smoke test for the full pipeline.

Verifies that:
1. Stage 1 can produce a valid Snapshot Event and push it to Redis.
2. Stage 2 can run the DSP pipeline on a synthetic plate image.
3. Stage 3 can run OCR (stub mode) and produce a Final Detection Record.
4. The Final Detection Record matches the expected schema.

Run with: python scripts/test_pipeline.py
Requires Redis running at REDIS_HOST:REDIS_PORT.
"""

import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import redis

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None
TEST_QUEUE = "smoketest:pipeline:tasks"
TEST_DLQ = "smoketest:pipeline:dlq"
TEST_CAMERA_ID = "smoketest_cam_01"
TEST_TRACK_ID = 9999

PASS = "✓"
FAIL = "✗"


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def _err(msg: str) -> None:
    print(f"  {FAIL}  {msg}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """A synthetic traffic scene frame."""
    frame = np.full((h, w, 3), (30, 40, 30), dtype=np.uint8)
    # Draw a fake car rectangle
    cv2.rectangle(frame, (200, 150), (440, 350), (60, 60, 120), -1)
    cv2.rectangle(frame, (220, 200), (420, 280), (200, 200, 220), -1)  # windows
    return frame


def _make_synthetic_plate_image() -> np.ndarray:
    """A plate image with white text on black background."""
    img = np.zeros((60, 180, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (180, 60), (20, 20, 20), -1)
    cv2.putText(img, "16ABK123", (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 2)
    return img


def _encode_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Test steps
# ─────────────────────────────────────────────────────────────────────────────

def test_redis_connection(r: redis.Redis) -> bool:
    _section("Test 1 — Redis connectivity")
    try:
        r.ping()
        _ok(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return True
    except redis.RedisError as exc:
        _err(f"Redis unavailable: {exc}")
        return False


def test_stage1_snapshot_push(r: redis.Redis) -> bool:
    _section("Test 2 — Stage 1: Tracking Ledger + Snapshot push")

    from stage1.ledger import TrackingLedger
    from stage1.snapshot import SnapshotPublisher

    ledger = TrackingLedger(r, TEST_CAMERA_ID)
    ledger._ledger_key = f"smoketest:ledger:{TEST_CAMERA_ID}"
    publisher = SnapshotPublisher(r, TEST_CAMERA_ID)
    publisher._queue_key = TEST_QUEUE

    # Clean state
    r.delete(ledger._ledger_key, TEST_QUEUE)

    # 1. Track should not be seen yet
    assert not ledger.is_seen(TEST_TRACK_ID), "Ledger pre-condition failed"
    _ok("Ledger is_seen → False (new track)")

    # 2. Push snapshot
    frame = _make_synthetic_frame()
    bbox = (200.0, 150.0, 440.0, 350.0)
    success = publisher.push(frame, TEST_TRACK_ID, 42, bbox, speed_kmh=97.3, confidence=0.91)

    if not success:
        _err("SnapshotPublisher.push() returned False")
        return False
    _ok("Snapshot pushed to Redis queue")

    # 3. Verify queue depth
    depth = r.llen(TEST_QUEUE)
    if depth != 1:
        _err(f"Expected queue depth=1, got {depth}")
        return False
    _ok(f"Queue depth = {depth}")

    # 4. Mark seen and verify dedup
    ledger.mark_seen(TEST_TRACK_ID)
    assert ledger.is_seen(TEST_TRACK_ID), "Ledger mark_seen failed"
    _ok("Ledger deduplication works correctly")

    return True


def test_stage2_dsp(snapshot_event: dict) -> np.ndarray | None:
    _section("Test 3 — Stage 2: Plate detection + DSP pipeline")

    try:
        from stage2.plate_detector import PlateDetector
        from stage2.preprocessor import Preprocessor

        detector = PlateDetector()
        preprocessor = Preprocessor()

        plate_crop, plate_conf = detector.detect(snapshot_event["image_b64"])
        _ok(f"Plate detector returned crop shape={plate_crop.shape}, conf={plate_conf:.3f}")

        processed = preprocessor.process(plate_crop)
        _ok(f"DSP pipeline output shape={processed.shape}, dtype={processed.dtype}")

        unique_vals = set(np.unique(processed))
        assert unique_vals.issubset({0, 255}), f"Non-binary output: {unique_vals}"
        _ok("DSP output is binary (0 and 255 only)")

        return processed

    except Exception as exc:
        _err(f"Stage 2 failed: {exc}")
        return None


def test_stage3_ocr_and_aggregate(processed_plate: np.ndarray, snapshot_event: dict) -> dict | None:
    _section("Test 4 — Stage 3: OCR + Aggregation")

    try:
        from stage3.aggregator import Aggregator
        from stage3.ocr import OCRProcessor

        ocr = OCRProcessor()
        aggregator = Aggregator()

        task_start = time.monotonic()
        ocr_result = ocr.read(processed_plate)
        _ok(f"OCR result: text={ocr_result.text!r}, confidence={ocr_result.confidence:.3f}")

        record = aggregator.merge(snapshot_event, ocr_result, task_start)

        # Validate Final Detection Record schema
        required_keys = {
            "plate_number", "ocr_confidence", "speed_kmh",
            "timestamp_utc", "camera_id", "track_id", "processing_latency_ms",
        }
        missing = required_keys - set(record.keys())
        if missing:
            _err(f"Final Detection Record missing keys: {missing}")
            return None

        _ok("Final Detection Record schema is complete")
        _ok(f"  plate_number       = {record['plate_number']!r}")
        _ok(f"  ocr_confidence     = {record['ocr_confidence']:.4f}")
        _ok(f"  speed_kmh          = {record['speed_kmh']}")
        _ok(f"  camera_id          = {record['camera_id']}")
        _ok(f"  track_id           = {record['track_id']}")
        _ok(f"  processing_latency = {record['processing_latency_ms']} ms")

        return record

    except Exception as exc:
        _err(f"Stage 3 failed: {exc}")
        return None


def test_data_contract_passthrough(record: dict, snapshot_event: dict) -> bool:
    _section("Test 5 — Data contract: Stage 1 metadata pass-through")

    checks = [
        ("speed_kmh", snapshot_event["speed_kmh"], record["speed_kmh"]),
        ("camera_id", snapshot_event["camera_id"], record["camera_id"]),
        ("track_id", snapshot_event["track_id"], record["track_id"]),
        ("timestamp_utc", snapshot_event["timestamp_utc"], record["timestamp_utc"]),
    ]

    all_ok = True
    for field, expected, actual in checks:
        if expected == actual:
            _ok(f"{field}: {actual!r} ✓")
        else:
            _err(f"{field}: expected {expected!r}, got {actual!r}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup(r: redis.Redis) -> None:
    for key in r.scan_iter("smoketest:*"):
        r.delete(key)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  Traffic Pipeline — End-to-End Smoke Test")
    print("═" * 60)

    r = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
        decode_responses=True, socket_connect_timeout=5,
    )

    results = []

    # ── Test 1: Redis ────────────────────────────────────────────────────
    results.append(("Redis connection", test_redis_connection(r)))
    if not results[-1][1]:
        print("\nCannot continue without Redis. Exiting.")
        sys.exit(1)

    # ── Test 2: Stage 1 ─────────────────────────────────────────────────
    results.append(("Stage 1 snapshot push", test_stage1_snapshot_push(r)))

    # Pop the event from queue for downstream tests
    raw_item = r.brpop(TEST_QUEUE, timeout=2)
    if raw_item is None:
        _err("Could not pop snapshot event from queue")
        sys.exit(1)
    snapshot_event = json.loads(raw_item[1])

    # ── Test 3: Stage 2 ─────────────────────────────────────────────────
    processed_plate = test_stage2_dsp(snapshot_event)
    results.append(("Stage 2 DSP pipeline", processed_plate is not None))

    if processed_plate is None:
        # Use synthetic plate as fallback so tests can continue
        processed_plate = cv2.cvtColor(
            cv2.bilateralFilter(_make_synthetic_plate_image(), 9, 75, 75),
            cv2.COLOR_BGR2GRAY
        )

    # ── Test 4: Stage 3 ─────────────────────────────────────────────────
    record = test_stage3_ocr_and_aggregate(processed_plate, snapshot_event)
    results.append(("Stage 3 OCR + aggregate", record is not None))

    # ── Test 5: Data contract ────────────────────────────────────────────
    if record:
        results.append(("Data contract passthrough", test_data_contract_passthrough(record, snapshot_event)))

    # ── Summary ─────────────────────────────────────────────────────────
    _section("Results Summary")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")

    print(f"\n  Passed: {passed}/{total}")
    print("═" * 60 + "\n")

    _cleanup(r)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
