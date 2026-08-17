"""
OpenCV drawing helpers shared between enrollment and recognition UIs.

All functions modify *frame* in-place and return None.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]   # BGR

# 5-point landmark indices (InsightFace order):
#   0 = left eye,  1 = right eye,  2 = nose tip,
#   3 = left mouth corner,  4 = right mouth corner
_KPS_COLORS: list[Color] = [
    (255, 100, 100),   # left eye     — blue
    (100, 100, 255),   # right eye    — red
    (100, 255, 100),   # nose tip     — green
    (200, 200, 0),     # mouth left   — cyan
    (0, 200, 200),     # mouth right  — yellow
]


def draw_bbox(frame: np.ndarray, bbox: np.ndarray, color: Color,
              thickness: int = 2) -> None:
    """Draw a bounding box from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_landmarks(frame: np.ndarray,
                   kps: Optional[np.ndarray]) -> None:
    """
    Draw 5-point facial landmarks when kps is not None.

    kps shape: (5, 2) — each row is (x, y).
    Enabled by settings.show_landmarks=True in the runner.
    """
    if kps is None:
        return
    for i, (x, y) in enumerate(kps):
        color = _KPS_COLORS[i % len(_KPS_COLORS)]
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def draw_guide_text(frame: np.ndarray, text: str, color: Color,
                    y_offset: int = 30) -> None:
    """Render a single guidance line near the top of the frame."""
    # Black shadow for readability on any background
    cv2.putText(frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 2, cv2.LINE_AA)


def draw_recognition_label(frame: np.ndarray, bbox: np.ndarray,
                            authorized: bool,
                            username: Optional[str],
                            similarity: float,
                            font_scale: float = 0.7,
                            kps: Optional[np.ndarray] = None,
                            show_landmarks: bool = False) -> None:
    """
    Overlay the recognition result on the face bounding box.

    Renders two lines above the box:
      AUTHORIZED  User: abba   Similarity: 91%
      UNKNOWN                  Similarity: 23%

    Optionally draws 5-point landmarks when show_landmarks=True.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)

    if authorized and username:
        label_top = f"AUTHORIZED  User: {username}"
        color: Color = (0, 220, 0)    # green
    else:
        label_top = "UNKNOWN"
        color = (0, 0, 230)           # red

    label_bot = f"Similarity: {similarity:.0%}"

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Background banner for the two text lines
    (tw1, th1), _ = cv2.getTextSize(
        label_top, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    (tw2, th2), _ = cv2.getTextSize(
        label_bot, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, 2)
    banner_w = max(tw1, tw2) + 10
    banner_h = th1 + th2 + 18
    top_y = max(0, y1 - banner_h)

    cv2.rectangle(frame, (x1, top_y), (x1 + banner_w, y1), color, -1)

    # Top label
    cv2.putText(frame, label_top,
                (x1 + 4, top_y + th1 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    # Bottom label
    cv2.putText(frame, label_bot,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Landmarks (optional)
    if show_landmarks and kps is not None:
        draw_landmarks(frame, kps)
