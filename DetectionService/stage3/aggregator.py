"""
aggregator.py — Merges OCR result with Stage 1 metadata for Stage 3.

Produces the Final Detection Record that is dispatched to the external
egress endpoint.

Data Contract (claude.md §Data Contracts — Final Detection Record):
    {
        "plate_number":          str,   # "" if OCR failed / went to DLQ
        "ocr_confidence":        float,
        "speed_kmh":             float,
        "timestamp_utc":         str,
        "camera_id":             str,
        "track_id":              int,
        "processing_latency_ms": int,   # wall clock: snapshot push → egress dispatch
    }
"""

import logging
import time
from typing import Any

from stage3.ocr import OCRResult

logger = logging.getLogger(__name__)


class Aggregator:
    """
    Stateless aggregator — merges the OCR result with Stage 1 metadata
    to produce the Final Detection Record.

    Instantiate once and call merge() for each task consumed from the queue.
    """

    def merge(
        self,
        snapshot_event: dict[str, Any],
        ocr_result: OCRResult,
        task_enqueued_at: float,
    ) -> dict[str, Any]:
        """
        Build the Final Detection Record.

        Args:
            snapshot_event:   The deserialized Snapshot Event from Stage 1.
            ocr_result:       OCR read result (text + confidence).
            task_enqueued_at: Unix timestamp (seconds) when the task was
                              pushed to the queue by Stage 1.

        Returns:
            Final Detection Record dict ready for egress.
        """
        # Wall clock latency in milliseconds
        latency_ms = int((time.monotonic() - task_enqueued_at) * 1000)

        record = {
            "plate_number": ocr_result.text,
            "ocr_confidence": round(ocr_result.confidence, 4),
            "speed_kmh": float(snapshot_event.get("speed_kmh", 0.0)),
            "timestamp_utc": snapshot_event.get("timestamp_utc", ""),
            "camera_id": snapshot_event.get("camera_id", ""),
            "track_id": int(snapshot_event.get("track_id", -1)),
            "processing_latency_ms": latency_ms,
        }

        logger.debug(
            "Aggregated detection: plate=%r conf=%.3f speed=%.1f km/h latency=%dms",
            record["plate_number"],
            record["ocr_confidence"],
            record["speed_kmh"],
            latency_ms,
        )

        return record
