"""
calibrate_homography.py — Camera calibration tool for Stage 1.

Computes a pixel-to-metre homography matrix by letting the operator
click 4 points on a camera frame and enter their real-world coordinates.

Output JSON is loaded by Stage 1 via the HOMOGRAPHY_PATH env var.

Usage:
    python scripts/calibrate_homography.py \
        --rtsp-url rtsp://192.168.1.10/stream \
        --output infra/calibrations/cam_01.json

Controls (OpenCV window):
    Click 4 reference points on the frame (in order: TL, TR, BR, BL)
    Press 'c' to confirm and compute
    Press 'r' to reset points
    Press 'q' to quit without saving
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Globals (shared between OpenCV mouse callback and main loop)
# ─────────────────────────────────────────────────────────────────────────────

_clicked_points: list[tuple[int, int]] = []
MAX_POINTS = 4

POINT_LABELS = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
POINT_COLORS = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 0)]


def _mouse_callback(event, x: int, y: int, flags, param) -> None:
    """Capture click coordinates, up to MAX_POINTS."""
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked_points) < MAX_POINTS:
        _clicked_points.append((x, y))
        print(f"  [{len(_clicked_points)}/{MAX_POINTS}] {POINT_LABELS[len(_clicked_points)-1]}: pixel ({x}, {y})")


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw clicked points and instructions on the frame."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Instructions
    instructions = [
        "Click 4 reference points: TL -> TR -> BR -> BL",
        f"Points: {len(_clicked_points)}/{MAX_POINTS}",
        "'c' = confirm | 'r' = reset | 'q' = quit",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(overlay, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Clicked points
    for i, (px, py) in enumerate(_clicked_points):
        color = POINT_COLORS[i]
        cv2.circle(overlay, (px, py), 8, color, -1)
        cv2.circle(overlay, (px, py), 12, (255, 255, 255), 2)
        cv2.putText(overlay, POINT_LABELS[i][:2], (px + 15, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Draw polygon when 4 points selected
    if len(_clicked_points) == MAX_POINTS:
        pts = np.array(_clicked_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(overlay, "Press 'c' to compute homography", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return overlay


def _collect_world_coords(pixel_points: list[tuple]) -> Optional[list[tuple]]:
    """Prompt operator to enter real-world (X, Y) in metres for each pixel point."""
    world_points = []
    print("\nEnter real-world coordinates in METRES for each pixel point.")
    print("Tip: measure distances on the road using a tape or reference markers.\n")

    for i, (px, py) in enumerate(pixel_points):
        label = POINT_LABELS[i]
        print(f"  {label}: pixel ({px}, {py})")
        try:
            x_m = float(input(f"    Real-world X (metres): "))
            y_m = float(input(f"    Real-world Y (metres): "))
            world_points.append((x_m, y_m))
        except (ValueError, EOFError):
            print("  Invalid input. Aborting.")
            return None

    return world_points


def _compute_homography(
    pixel_points: list[tuple],
    world_points: list[tuple],
) -> np.ndarray:
    """Compute the 3×3 homography matrix mapping pixels → metres."""
    src = np.array(pixel_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("cv2.findHomography returned None — points may be collinear")
    return H


def _save_calibration(
    output_path: str,
    homography_matrix: np.ndarray,
    pixel_points: list,
    world_points: list,
    rtsp_url: str,
) -> None:
    """Write the calibration JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "homography_matrix": homography_matrix.tolist(),
        "calibration_metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rtsp_url_hint": rtsp_url[:40] + "..." if len(rtsp_url) > 40 else rtsp_url,
            "pixel_points": pixel_points,
            "world_points_metres": world_points,
        },
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n✓ Calibration saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pixel→metre homography matrix for a camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rtsp-url", required=True, help="Camera RTSP URL or video file path")
    parser.add_argument("--output", required=True, help="Output JSON path (e.g. infra/calibrations/cam_01.json)")
    parser.add_argument("--frame-skip", type=int, default=30, help="Skip N frames before capturing reference frame")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"ERROR: Cannot open stream: {args.rtsp_url}", file=sys.stderr)
        sys.exit(1)

    # Skip to a stable frame
    for _ in range(args.frame_skip):
        cap.read()

    ret, reference_frame = cap.read()
    cap.release()

    if not ret or reference_frame is None:
        print("ERROR: Failed to read frame from stream.", file=sys.stderr)
        sys.exit(1)

    print(f"Frame captured ({reference_frame.shape[1]}×{reference_frame.shape[0]})")
    print("Click 4 reference points in the window (TL → TR → BR → BL)\n")

    window_name = "Homography Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)

    while True:
        display = _draw_overlay(reference_frame)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)

        elif key == ord("r"):
            _clicked_points.clear()
            print("Points reset.")

        elif key == ord("c") and len(_clicked_points) == MAX_POINTS:
            cv2.destroyAllWindows()
            world_pts = _collect_world_coords(_clicked_points)
            if world_pts is None:
                sys.exit(1)

            try:
                H = _compute_homography(_clicked_points, world_pts)
            except ValueError as exc:
                print(f"ERROR computing homography: {exc}", file=sys.stderr)
                sys.exit(1)

            print("\nHomography matrix:")
            print(H)

            _save_calibration(args.output, H, _clicked_points, world_pts, args.rtsp_url)
            sys.exit(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
