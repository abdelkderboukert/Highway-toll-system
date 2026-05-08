"""
tracker.py — Multi-Object Tracking (MOT) wrapper for Stage 1.

Wraps ByteTrack / StrongSORT when available, with a lightweight
IoU-based fallback tracker for environments without the full MOT stack.

The tracker assigns a persistent integer track_id to each vehicle
across consecutive frames.

Usage:
    tracker = Tracker()
    for frame in stream:
        detections = yolo.detect(frame)   # list of (bbox, confidence, class)
        tracks = tracker.update(detections)
        for track in tracks:
            print(track.track_id, track.bbox)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# MOT backend: "bytetrack", "strongsort", "iou" (fallback)
_MOT_BACKEND = os.environ.get("MOT_BACKEND", "iou").lower()

# IoU threshold for associating a detection to an existing track
_IOU_THRESHOLD = float(os.environ.get("MOT_IOU_THRESHOLD", "0.35"))

# Max frames a track can go unmatched before being evicted
_MAX_AGE = int(os.environ.get("MOT_MAX_AGE", "10"))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    Represents a single confirmed vehicle track.

    Attributes:
        track_id:   Unique integer identity for this camera session.
        bbox:       Latest bounding box as (x1, y1, x2, y2).
        confidence: Latest YOLO detection confidence.
        age:        Frames since last matched detection.
        hits:       Total frames this track has been matched.
    """
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    age: int = 0
    hits: int = 1
    _history: list = field(default_factory=list, repr=False)

    def update(self, bbox: tuple, confidence: float) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0
        self.hits += 1
        self._history.append(bbox)
        if len(self._history) > 30:
            self._history.pop(0)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# IoU Fallback Tracker
# ─────────────────────────────────────────────────────────────────────────────

class _IoUTracker:
    """
    Lightweight greedy IoU tracker. Zero external dependencies.

    Assigns track_ids by greedily matching new detections to existing
    tracks using highest-IoU pairing. Unmatched tracks age out after
    MAX_AGE frames.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    def update(self, detections: list[tuple]) -> list[Track]:
        """
        Args:
            detections: list of (bbox, confidence) where bbox=(x1,y1,x2,y2)

        Returns:
            List of active Track objects (only tracks with hits >= 1).
        """
        # Age all existing tracks
        for t in self._tracks.values():
            t.age += 1

        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # Greedy matching
        for det_idx in list(unmatched_dets):
            bbox_det, conf_det = detections[det_idx]
            best_iou = _IOU_THRESHOLD
            best_tid: Optional[int] = None

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = _iou(bbox_det, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid].update(bbox_det, conf_det)
                matched_track_ids.add(best_tid)
                unmatched_dets.remove(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox, conf = detections[det_idx]
            new_track = Track(
                track_id=self._next_id,
                bbox=bbox,
                confidence=conf,
            )
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # Evict stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.age > _MAX_AGE]
        for tid in stale:
            self._tracks.pop(tid)

        return list(self._tracks.values())


# ─────────────────────────────────────────────────────────────────────────────
# Public Tracker facade
# ─────────────────────────────────────────────────────────────────────────────

class Tracker:
    """
    Public MOT facade. Selects the backend at construction time based on
    the MOT_BACKEND environment variable.

    Supported backends:
        iou         — Built-in greedy IoU tracker (default, zero deps).
        bytetrack   — ByteTrack (requires boxmot package).
        strongsort  — StrongSORT (requires boxmot package).
    """

    def __init__(self) -> None:
        self._backend = _MOT_BACKEND
        self._impl = self._init_backend()

    def _init_backend(self):
        if self._backend in ("bytetrack", "strongsort"):
            try:
                from boxmot import BYTETracker, StrongSORT  # type: ignore
                if self._backend == "bytetrack":
                    impl = BYTETracker()
                    logger.info("MOT backend: ByteTrack (boxmot)")
                else:
                    impl = StrongSORT()
                    logger.info("MOT backend: StrongSORT (boxmot)")
                return impl
            except ImportError:
                logger.warning(
                    "boxmot not installed; falling back to IoU tracker. "
                    "Install with: pip install boxmot"
                )

        logger.info("MOT backend: IoU fallback tracker")
        return _IoUTracker()

    def update(self, detections: list[tuple]) -> list[Track]:
        """
        Update the tracker with new detections.

        Args:
            detections: list of (bbox, confidence) tuples.
                        bbox = (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            List of active Track objects.
        """
        if isinstance(self._impl, _IoUTracker):
            return self._impl.update(detections)

        # boxmot interface: expects numpy array [x1,y1,x2,y2,conf,cls]
        if not detections:
            dets_np = np.empty((0, 6))
        else:
            rows = []
            for bbox, conf in detections:
                rows.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, 0])
            dets_np = np.array(rows, dtype=np.float32)

        raw_tracks = self._impl.update(dets_np, img=None)

        result = []
        for row in raw_tracks:
            x1, y1, x2, y2, tid, *_ = row
            result.append(
                Track(
                    track_id=int(tid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(row[4]) if len(row) > 4 else 1.0,
                )
            )
        return result
