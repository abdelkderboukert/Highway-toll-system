"""
test_pipeline.py — Integration tests for the full Stage 1 → Redis → Worker pipeline.

Requires a live Redis instance. Start one with:
    docker-compose up -d redis

Run with: pytest tests/integration/test_pipeline.py -v
"""

import asyncio
import base64
import json
import os
import time
import unittest

import cv2
import numpy as np
import redis

# ─── Redis connection ─────────────────────────────────────────────────────────

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None

TEST_QUEUE_KEY = "test:pipeline:tasks"
TEST_LEDGER_PREFIX = "test:ledger:"
TEST_DLQ_KEY = "test:pipeline:dlq"


def _get_redis() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5,
    )


def _is_redis_available() -> bool:
    try:
        r = _get_redis()
        r.ping()
        return True
    except redis.RedisError:
        return False


def _make_synthetic_snapshot_event(track_id: int = 1, camera_id: str = "test_cam") -> dict:
    """Build a realistic Snapshot Event with a synthetic vehicle image."""
    # 100×50 BGR plate-like image
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (30, 30, 80)
    cv2.putText(img, "16ABK123", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "track_id": track_id,
        "camera_id": camera_id,
        "timestamp_utc": "2026-05-07T14:23:01.456Z",
        "frame_number": 42,
        "speed_kmh": 87.5,
        "confidence": 0.94,
        "image_b64": image_b64,
    }


@unittest.skipUnless(_is_redis_available(), "Redis not available — skipping integration tests")
class TestLedgerIntegration(unittest.TestCase):
    """Integration tests for the Tracking Ledger against a real Redis instance."""

    def setUp(self):
        self.redis = _get_redis()
        # Clean up any leftover test keys
        for key in self.redis.scan_iter(f"{TEST_LEDGER_PREFIX}*"):
            self.redis.delete(key)

    def tearDown(self):
        for key in self.redis.scan_iter(f"{TEST_LEDGER_PREFIX}*"):
            self.redis.delete(key)

    def test_ledger_deduplication_e2e(self):
        """A track_id added by mark_seen should be rejected by subsequent is_seen calls."""
        import os
        os.environ["LEDGER_TTL_SECONDS"] = "10"

        from stage1.ledger import TrackingLedger

        # Use isolated test key
        cam_id = "integration_test_cam"
        ledger = TrackingLedger(self.redis, cam_id)
        # Override ledger key to use test namespace
        ledger._ledger_key = f"{TEST_LEDGER_PREFIX}{cam_id}"

        self.assertFalse(ledger.is_seen(999))
        ledger.mark_seen(999)
        self.assertTrue(ledger.is_seen(999))

    def test_ledger_camera_isolation(self):
        """Two cameras must use separate ledger keys."""
        from stage1.ledger import TrackingLedger

        ledger_a = TrackingLedger(self.redis, "cam_a")
        ledger_a._ledger_key = f"{TEST_LEDGER_PREFIX}cam_a"
        ledger_b = TrackingLedger(self.redis, "cam_b")
        ledger_b._ledger_key = f"{TEST_LEDGER_PREFIX}cam_b"

        ledger_a.mark_seen(42)

        self.assertTrue(ledger_a.is_seen(42))
        self.assertFalse(ledger_b.is_seen(42))  # cam_b should NOT see cam_a's track


@unittest.skipUnless(_is_redis_available(), "Redis not available — skipping integration tests")
class TestQueueIntegration(unittest.TestCase):
    """Integration tests for the Redis FIFO queue mechanics."""

    def setUp(self):
        self.redis = _get_redis()
        self.redis.delete(TEST_QUEUE_KEY)
        self.redis.delete(TEST_DLQ_KEY)

    def tearDown(self):
        self.redis.delete(TEST_QUEUE_KEY)
        self.redis.delete(TEST_DLQ_KEY)

    def test_lpush_brpop_fifo_order(self):
        """Tasks pushed with LPUSH should be consumed in FIFO order via BRPOP."""
        events = [
            _make_synthetic_snapshot_event(track_id=i)
            for i in range(1, 4)
        ]

        for ev in events:
            self.redis.lpush(TEST_QUEUE_KEY, json.dumps(ev))

        consumed_ids = []
        for _ in range(3):
            item = self.redis.brpop(TEST_QUEUE_KEY, timeout=2)
            self.assertIsNotNone(item)
            payload = json.loads(item[1])
            consumed_ids.append(payload["track_id"])

        # FIFO: pushed 1,2,3 → consumed 1,2,3 (LPUSH pushes to head, BRPOP pops from tail)
        self.assertEqual(consumed_ids, [1, 2, 3])

    def test_snapshot_payload_survives_round_trip(self):
        """A Snapshot Event serialised to JSON and back must preserve all fields."""
        event = _make_synthetic_snapshot_event(track_id=77, camera_id="cam_south")
        self.redis.lpush(TEST_QUEUE_KEY, json.dumps(event))

        item = self.redis.brpop(TEST_QUEUE_KEY, timeout=2)
        recovered = json.loads(item[1])

        self.assertEqual(recovered["track_id"], 77)
        self.assertEqual(recovered["camera_id"], "cam_south")
        self.assertIn("image_b64", recovered)
        self.assertAlmostEqual(recovered["speed_kmh"], event["speed_kmh"])
        self.assertAlmostEqual(recovered["confidence"], event["confidence"])

    def test_queue_depth_reflects_backlog(self):
        """llen should accurately report the number of pending tasks."""
        for i in range(5):
            self.redis.lpush(TEST_QUEUE_KEY, json.dumps({"track_id": i}))

        depth = self.redis.llen(TEST_QUEUE_KEY)
        self.assertEqual(depth, 5)


@unittest.skipUnless(_is_redis_available(), "Redis not available — skipping integration tests")
class TestPreprocessorIntegration(unittest.TestCase):
    """Integration test: DSP pipeline applied to a realistic plate image."""

    def test_full_dsp_pipeline_on_synthetic_plate(self):
        """Full DSP pipeline must produce a non-empty binary image."""
        from stage2.preprocessor import Preprocessor

        # Synthetic plate: dark background, white text
        img = np.zeros((60, 180, 3), dtype=np.uint8)
        cv2.putText(img, "16ABK123", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 2)

        p = Preprocessor()
        result = p.process(img)

        self.assertEqual(len(result.shape), 2)  # Single channel
        self.assertEqual(result.dtype, np.uint8)

        # Binary: only 0 and 255
        unique = set(np.unique(result))
        self.assertTrue(unique.issubset({0, 255}))

        # Not entirely black or white — text creates mixed pixels
        white_ratio = (result == 255).mean()
        self.assertGreater(white_ratio, 0.05)
        self.assertLess(white_ratio, 0.95)


if __name__ == "__main__":
    unittest.main()
