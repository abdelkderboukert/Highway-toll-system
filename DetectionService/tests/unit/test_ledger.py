"""
test_ledger.py — Unit tests for the Stage 1 Tracking Ledger.

Tests deduplication logic, TTL handling, and Redis failure resilience.
Run with: pytest tests/unit/test_ledger.py -v
"""

import time
import unittest
from unittest.mock import MagicMock, call, patch

import redis

from stage1.ledger import TrackingLedger


class TestTrackingLedger(unittest.TestCase):
    """Tests for TrackingLedger deduplication behaviour."""

    def _make_ledger(self, camera_id: str = "cam_test") -> tuple[TrackingLedger, MagicMock]:
        """Return a TrackingLedger and its mock Redis client."""
        mock_redis = MagicMock(spec=redis.Redis)
        # Default: track_id not seen
        mock_redis.sismember.return_value = False
        ledger = TrackingLedger(mock_redis, camera_id)
        return ledger, mock_redis

    # ── is_seen ────────────────────────────────────────────────────────────

    def test_is_seen_returns_false_for_new_track(self):
        ledger, mock_redis = self._make_ledger()
        mock_redis.sismember.return_value = False
        self.assertFalse(ledger.is_seen(42))

    def test_is_seen_returns_true_for_existing_track(self):
        ledger, mock_redis = self._make_ledger()
        mock_redis.sismember.return_value = True
        self.assertTrue(ledger.is_seen(42))

    def test_is_seen_uses_correct_ledger_key(self):
        ledger, mock_redis = self._make_ledger("cam_north_01")
        ledger.is_seen(99)
        mock_redis.sismember.assert_called_once_with("ledger:cam_north_01", 99)

    def test_is_seen_returns_false_on_redis_failure(self):
        """On Redis error, is_seen should fail OPEN (allow through) not crash."""
        ledger, mock_redis = self._make_ledger()
        mock_redis.sismember.side_effect = redis.RedisError("connection refused")
        result = ledger.is_seen(42)
        self.assertFalse(result)  # Must not raise; defaults to False

    # ── mark_seen ──────────────────────────────────────────────────────────

    def test_mark_seen_uses_pipeline(self):
        """mark_seen must use a Redis pipeline for atomic SADD + EXPIRE."""
        ledger, mock_redis = self._make_ledger()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        ledger.mark_seen(7)

        mock_redis.pipeline.assert_called_once()
        mock_pipe.sadd.assert_called_once()
        mock_pipe.expire.assert_called_once()
        mock_pipe.execute.assert_called_once()

    def test_mark_seen_sets_correct_ttl(self):
        """Expire should be set to LEDGER_TTL_SECONDS."""
        with patch.dict("os.environ", {"LEDGER_TTL_SECONDS": "120"}):
            ledger, mock_redis = self._make_ledger()
            mock_pipe = MagicMock()
            mock_redis.pipeline.return_value = mock_pipe

            ledger.mark_seen(55)

            # Second call to mock_pipe.expire should include TTL=120
            expire_args = mock_pipe.expire.call_args[0]
            self.assertIn(120, expire_args)

    def test_mark_seen_does_not_raise_on_redis_failure(self):
        """mark_seen must never raise — Stage 1 cannot crash on egress errors."""
        ledger, mock_redis = self._make_ledger()
        mock_pipe = MagicMock()
        mock_pipe.execute.side_effect = redis.RedisError("timeout")
        mock_redis.pipeline.return_value = mock_pipe

        # Must not raise
        try:
            ledger.mark_seen(1)
        except Exception as exc:
            self.fail(f"mark_seen raised unexpectedly: {exc}")

    # ── is_seen → mark_seen deduplication cycle ────────────────────────────

    def test_full_dedup_cycle(self):
        """
        Simulate the full ledger flow:
            1. is_seen() → False (new track)
            2. mark_seen() called
            3. is_seen() → True (now deduplicated)
        """
        seen_set: set = set()

        def fake_sismember(key, tid):
            return tid in seen_set

        def fake_sadd(key, tid):
            seen_set.add(tid)
            return 1

        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.sismember.side_effect = fake_sismember

        mock_pipe = MagicMock()
        mock_pipe.sadd.side_effect = lambda k, v: seen_set.add(v)
        mock_pipe.expire.return_value = True
        mock_pipe.execute.return_value = [1, True]
        mock_redis.pipeline.return_value = mock_pipe

        ledger = TrackingLedger(mock_redis, "cam_test")

        self.assertFalse(ledger.is_seen(100))
        ledger.mark_seen(100)
        self.assertTrue(ledger.is_seen(100))

    # ── size ───────────────────────────────────────────────────────────────

    def test_size_returns_scard_result(self):
        ledger, mock_redis = self._make_ledger()
        mock_redis.scard.return_value = 17
        self.assertEqual(ledger.size(), 17)

    def test_size_returns_minus_one_on_error(self):
        ledger, mock_redis = self._make_ledger()
        mock_redis.scard.side_effect = redis.RedisError("unavailable")
        self.assertEqual(ledger.size(), -1)

    # ── camera isolation ───────────────────────────────────────────────────

    def test_different_cameras_use_different_keys(self):
        """Two cameras must never share the same ledger key."""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.sismember.return_value = False

        ledger_a = TrackingLedger(mock_redis, "cam_north")
        ledger_b = TrackingLedger(mock_redis, "cam_south")

        ledger_a.is_seen(1)
        ledger_b.is_seen(1)

        calls = mock_redis.sismember.call_args_list
        keys_used = {c[0][0] for c in calls}
        self.assertIn("ledger:cam_north", keys_used)
        self.assertIn("ledger:cam_south", keys_used)
        self.assertEqual(len(keys_used), 2)


if __name__ == "__main__":
    unittest.main()
