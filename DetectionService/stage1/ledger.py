"""
ledger.py — Redis-backed Tracking Ledger for Stage 1.

Stores seen track_id values in a Redis SET with TTL-based expiry.
This is the single deduplication gate that prevents duplicate Snapshot
Events from flooding the queue.

Critical Rule: ALWAYS call is_seen() before enqueueing. Never bypass.
"""

import logging
import os

import redis

logger = logging.getLogger(__name__)

# Redis key prefix for the per-camera tracking ledger
_LEDGER_KEY_PREFIX = "ledger:"


class TrackingLedger:
    """
    Redis SET–based deduplication ledger.

    Each camera maintains its own ledger key scoped by camera_id.
    A track_id entry expires after LEDGER_TTL_SECONDS, allowing
    re-entry if the same vehicle reappears later.
    """

    def __init__(self, redis_client: redis.Redis, camera_id: str) -> None:
        self._redis = redis_client
        self._camera_id = camera_id
        self._ledger_key = f"{_LEDGER_KEY_PREFIX}{camera_id}"
        self._ttl: int = int(os.environ.get("LEDGER_TTL_SECONDS", "300"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_seen(self, track_id: int) -> bool:
        """
        Return True if track_id has already been processed.

        This is a read-only check — it does NOT mark the track_id as seen.
        Use mark_seen() after a successful enqueue.
        """
        try:
            result = self._redis.sismember(self._ledger_key, track_id)
            return bool(result)
        except redis.RedisError as exc:
            # On Redis failure, default to False (allow through) to avoid
            # dropping detections. The worker deduplication will catch
            # any duplicates on the consumer side.
            logger.warning(
                "Ledger is_seen check failed for track_id=%s camera=%s: %s",
                track_id,
                self._camera_id,
                exc,
            )
            return False

    def mark_seen(self, track_id: int) -> None:
        """
        Add track_id to the ledger SET and refresh its TTL.

        Uses a pipeline so the SADD + EXPIRE are atomic.
        Fire-and-forget: any Redis error is logged, not raised.
        """
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.sadd(self._ledger_key, track_id)
            pipe.expire(self._ledger_key, self._ttl)
            pipe.execute()
            logger.debug(
                "Ledger marked track_id=%s camera=%s (TTL=%ds)",
                track_id,
                self._camera_id,
                self._ttl,
            )
        except redis.RedisError as exc:
            logger.error(
                "Ledger mark_seen failed for track_id=%s camera=%s: %s",
                track_id,
                self._camera_id,
                exc,
            )

    def size(self) -> int:
        """Return the number of entries currently tracked in the ledger."""
        try:
            return self._redis.scard(self._ledger_key)
        except redis.RedisError:
            return -1
