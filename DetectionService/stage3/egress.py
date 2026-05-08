"""
egress.py — Async HTTP egress dispatcher for Stage 3.

Dispatches the Final Detection Record to an external microservice via
an async HTTP POST using aiohttp.

Critical Rules (claude.md §Critical Rules):
    - NEVER use synchronous HTTP here. Use aiohttp + asyncio.
    - Retry with exponential backoff (max EGRESS_MAX_RETRIES attempts).
    - On all retries exhausted, push to the Dead Letter Queue.
    - NEVER crash the worker on egress failure.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

import aiohttp
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_EGRESS_URL = os.environ.get("EGRESS_URL", "")
_MAX_RETRIES = int(os.environ.get("EGRESS_MAX_RETRIES", "3"))
_DLQ_KEY = os.environ.get("DEAD_LETTER_QUEUE_KEY", "pipeline:dlq")

# Base delay for exponential backoff (seconds)
_BACKOFF_BASE = float(os.environ.get("EGRESS_BACKOFF_BASE", "1.0"))

# Timeout for each HTTP request (seconds)
_HTTP_TIMEOUT = float(os.environ.get("EGRESS_HTTP_TIMEOUT", "10.0"))


# ─────────────────────────────────────────────────────────────────────────────
# Egress dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class EgressDispatcher:
    """
    Async egress dispatcher.

    Owns a persistent aiohttp.ClientSession for connection reuse.
    Call dispatch() from within an asyncio event loop (e.g., the worker loop).
    On persistent HTTP failure, the task is routed to the DLQ via Redis.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._session: Optional[aiohttp.ClientSession] = None
        self._egress_url = _EGRESS_URL
        self._max_retries = _MAX_RETRIES
        self._dlq_key = _DLQ_KEY

        if not self._egress_url:
            logger.warning(
                "EGRESS_URL not set. Detection records will go to DLQ only."
            )

    # ------------------------------------------------------------------
    # Session lifecycle (call from async context)
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open the HTTP session. Call once before the worker loop starts."""
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT)
        self._session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        logger.info("Egress HTTP session opened → %s", self._egress_url)

    async def close(self) -> None:
        """Close the HTTP session gracefully."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Egress HTTP session closed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        record: dict[str, Any],
        snapshot_event: dict[str, Any],
    ) -> bool:
        """
        POST the Final Detection Record to the egress endpoint.

        Args:
            record:         Final Detection Record from the aggregator.
            snapshot_event: Original Snapshot Event (for DLQ payloads).

        Returns:
            True on success, False if the task was routed to the DLQ.
        """
        if not self._egress_url:
            await self._push_to_dlq(record, snapshot_event, reason="EGRESS_URL not configured")
            return False

        body = json.dumps(record)

        for attempt in range(1, self._max_retries + 1):
            try:
                async with self._session.post(self._egress_url, data=body) as resp:
                    if resp.status < 300:
                        logger.info(
                            "Egress OK [%d] track_id=%s plate=%r",
                            resp.status,
                            record.get("track_id"),
                            record.get("plate_number"),
                        )
                        return True

                    logger.warning(
                        "Egress HTTP %d for track_id=%s (attempt %d/%d)",
                        resp.status,
                        record.get("track_id"),
                        attempt,
                        self._max_retries,
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Egress request error (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )

            if attempt < self._max_retries:
                delay = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.debug("Retrying in %.1fs ...", delay)
                await asyncio.sleep(delay)

        # All retries exhausted → push to DLQ
        await self._push_to_dlq(
            record, snapshot_event, reason=f"HTTP failure after {self._max_retries} attempts"
        )
        return False

    # ------------------------------------------------------------------
    # Dead Letter Queue
    # ------------------------------------------------------------------

    async def _push_to_dlq(
        self,
        record: dict[str, Any],
        snapshot_event: dict[str, Any],
        reason: str,
    ) -> None:
        """Push a failed task to the Dead Letter Queue for manual review."""
        dlq_payload = {
            "reason": reason,
            "final_record": record,
            "snapshot_event_keys": {
                k: v for k, v in snapshot_event.items() if k != "image_b64"
            },
        }
        try:
            await self._redis.lpush(self._dlq_key, json.dumps(dlq_payload))
            logger.error(
                "Task routed to DLQ (track_id=%s, reason=%s)",
                record.get("track_id"),
                reason,
            )
        except Exception as exc:
            logger.critical(
                "DLQ push FAILED for track_id=%s: %s — detection LOST",
                record.get("track_id"),
                exc,
            )
