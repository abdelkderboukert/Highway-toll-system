"""
main.py — Stage 2 + 3 combined worker process.

Continuously pops tasks from the Redis FIFO queue (BRPOP) and runs:
    Stage 2a — Plate detection (YOLO)
    Stage 2b — DSP preprocessing pipeline
    Stage 3  — OCR → Aggregation → Async egress

Critical Rules (claude.md §Critical Rules):
    - Workers must be STATELESS. No in-memory state between tasks.
    - All shared state lives in Redis.
    - Use aiohttp async HTTP for egress — never synchronous requests.
    - Never crash the worker on a task failure; log and continue.
"""

import asyncio
import json
import logging
import os
import sys
import time

import redis.asyncio as aioredis

from stage2.plate_detector import PlateDetector
from stage2.preprocessor import Preprocessor
from stage3.aggregator import Aggregator
from stage3.egress import EgressDispatcher
from stage3.ocr import OCRProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("worker.main")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None
QUEUE_KEY = os.environ.get("REDIS_QUEUE_KEY", "pipeline:tasks")
DLQ_KEY = os.environ.get("DEAD_LETTER_QUEUE_KEY", "pipeline:dlq")

# BRPOP timeout (seconds) — 0 = block indefinitely
BRPOP_TIMEOUT = int(os.environ.get("WORKER_BRPOP_TIMEOUT", "5"))


# ─────────────────────────────────────────────────────────────────────────────
# Redis connection
# ─────────────────────────────────────────────────────────────────────────────

async def _connect_redis() -> aioredis.Redis:
    """Connect to Redis with retry. Blocks until connected."""
    client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    while True:
        try:
            await client.ping()
            logger.info("Redis connected: %s:%d", REDIS_HOST, REDIS_PORT)
            return client
        except Exception as exc:
            logger.warning("Redis not ready (%s) — retrying in 3s ...", exc)
            await asyncio.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
# Task processing pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def process_task(
    raw_payload: str,
    plate_detector: PlateDetector,
    preprocessor: Preprocessor,
    ocr: OCRProcessor,
    aggregator: Aggregator,
    egress: EgressDispatcher,
) -> None:
    """
    Full Stage 2 + 3 pipeline for a single task.

    Never raises — all exceptions are caught and logged.
    """
    task_start = time.monotonic()
    snapshot_event: dict = {}

    try:
        snapshot_event = json.loads(raw_payload)
        track_id = snapshot_event.get("track_id", "?")
        logger.info("Processing task track_id=%s", track_id)

        # ── Stage 2a: Plate detection ─────────────────────────────────
        image_b64 = snapshot_event.get("image_b64", "")
        if not image_b64:
            raise ValueError("Snapshot event missing image_b64")

        plate_crop, plate_conf = plate_detector.detect(image_b64)
        logger.debug("Plate detection conf=%.3f track_id=%s", plate_conf, track_id)

        # ── Stage 2b: DSP preprocessing ───────────────────────────────
        processed_plate = preprocessor.process(plate_crop)

        # ── Stage 3a: OCR ─────────────────────────────────────────────
        ocr_result = ocr.read(processed_plate)
        logger.info(
            "OCR result: plate=%r conf=%.3f track_id=%s",
            ocr_result.text,
            ocr_result.confidence,
            track_id,
        )

        # ── Stage 3b: Aggregate ───────────────────────────────────────
        record = aggregator.merge(snapshot_event, ocr_result, task_start)

        # ── Stage 3c: Egress ──────────────────────────────────────────
        await egress.dispatch(record, snapshot_event)

    except json.JSONDecodeError as exc:
        logger.error("Malformed task payload (JSON): %s", exc)
    except ValueError as exc:
        logger.error("Task processing error: %s", exc)
    except Exception as exc:
        logger.exception("Unexpected error processing task: %s", exc)
        # Best-effort DLQ for completely unexpected failures
        try:
            import redis as sync_redis  # noqa: F401
            # We can't easily call egress._push_to_dlq here without the
            # full record, so log it for operator review
            logger.critical(
                "Task dropped unexpectedly. snapshot keys: %s",
                list(snapshot_event.keys()),
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Worker loop
# ─────────────────────────────────────────────────────────────────────────────

async def worker_loop() -> None:
    """
    Main async worker loop.

    Pops tasks from the Redis FIFO queue using BRPOP (blocking pop),
    processes each task through the Stage 2 + 3 pipeline, and repeats.

    The loop never exits on individual task failure — only on SIGINT/SIGTERM
    or an unrecoverable Redis connection error.
    """
    redis_client = await _connect_redis()

    # Initialise all pipeline components once (stateless — safe to share)
    plate_detector = PlateDetector()
    preprocessor = Preprocessor()
    ocr = OCRProcessor()
    aggregator = Aggregator()
    egress = EgressDispatcher(redis_client)

    await egress.open()
    logger.info("Worker ready — consuming from queue key=%s", QUEUE_KEY)

    tasks_processed = 0
    tasks_failed = 0

    try:
        while True:
            # BRPOP blocks for BRPOP_TIMEOUT seconds waiting for a task
            item = await redis_client.brpop(QUEUE_KEY, timeout=BRPOP_TIMEOUT)

            if item is None:
                # Timeout — no task available; log queue health periodically
                if tasks_processed % 100 == 0:
                    q_depth = await redis_client.llen(QUEUE_KEY)
                    dlq_depth = await redis_client.llen(DLQ_KEY)
                    logger.info(
                        "Queue status — depth=%d DLQ=%d processed=%d failed=%d",
                        q_depth,
                        dlq_depth,
                        tasks_processed,
                        tasks_failed,
                    )
                continue

            _key, raw_payload = item
            tasks_processed += 1

            try:
                await process_task(
                    raw_payload,
                    plate_detector,
                    preprocessor,
                    ocr,
                    aggregator,
                    egress,
                )
            except Exception:
                tasks_failed += 1
                logger.exception("Unhandled exception in process_task (task dropped)")

    except asyncio.CancelledError:
        logger.info("Worker loop cancelled — shutting down.")
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user.")
    finally:
        await egress.close()
        await redis_client.aclose()
        logger.info(
            "Worker shut down. Total processed=%d failed=%d",
            tasks_processed,
            tasks_failed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Starting Stage 2+3 worker process")
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        pass
    logger.info("Worker exited.")


if __name__ == "__main__":
    main()
