"""
OpenCV overlay helpers for the liveness challenge UI.

All functions modify *frame* in-place and return None.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from app.liveness.detector import LivenessState, LivenessResult

Color = Tuple[int, int, int]   # BGR

# State → colour mapping
_STATE_COLORS: dict[LivenessState, Color] = {
    LivenessState.WAITING:          (0, 165, 255),   # orange
    LivenessState.CHALLENGE_ACTIVE: (0, 200, 255),   # yellow-ish
    LivenessState.LIVE:             (0, 220, 0),     # green
    LivenessState.FAILED:           (0, 0, 220),     # red
    LivenessState.TIMEOUT:          (50, 50, 200),   # dark red
}


def draw_liveness_overlay(
    frame: np.ndarray,
    result: LivenessResult,
    font_scale: float = 0.7,
) -> None:
    """
    Draw the complete liveness UI overlay on *frame*.

    Renders:
      - Status banner at the top (state name + message)
      - Progress bar when CHALLENGE_ACTIVE (time remaining)
      - NOR debug readout when available
    """
    h, w = frame.shape[:2]
    color = _STATE_COLORS.get(result.state, (200, 200, 200))

    # ── Top banner ───────────────────────────────────────────────────────
    banner_h = 54
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # State label
    state_text = result.state.name.replace("_", " ")
    cv2.putText(frame, state_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 2, cv2.LINE_AA)

    # Message
    cv2.putText(frame, result.message, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (220, 220, 220), 1, cv2.LINE_AA)

    # ── Timeout progress bar (CHALLENGE_ACTIVE only) ──────────────────────
    if result.state == LivenessState.CHALLENGE_ACTIVE:
        timeout = result.timeout if result.timeout > 0 else 1.0
        frac_remaining = max(0.0, 1.0 - result.elapsed / timeout)
        bar_y = banner_h + 4
        bar_h = 6
        bar_w = w - 20
        # Background
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        # Fill — colour shifts red as time runs out
        fill_w = int(bar_w * frac_remaining)
        bar_color: Color = (
            (0, 200, 0) if frac_remaining > 0.5
            else (0, 165, 255) if frac_remaining > 0.25
            else (0, 0, 220)
        )
        if fill_w > 0:
            cv2.rectangle(frame, (10, bar_y), (10 + fill_w, bar_y + bar_h),
                          bar_color, -1)
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h),
                      (180, 180, 180), 1)

    # ── NOR readout (bottom-left, small) ─────────────────────────────────
    if result.nor is not None:
        nor_text = f"NOR: {result.nor:+.3f}"
        cv2.putText(frame, nor_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (160, 160, 160), 1, cv2.LINE_AA)

    # ── LIVE / FAILED / TIMEOUT — large centred result ────────────────────
    if result.state in (LivenessState.LIVE, LivenessState.FAILED,
                        LivenessState.TIMEOUT):
        label = {
            LivenessState.LIVE:    "LIVE ✓",
            LivenessState.FAILED:  "FAILED",
            LivenessState.TIMEOUT: "TIMEOUT",
        }[result.state]

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
        cx = (w - tw) // 2
        cy = h // 2 + th // 2

        # Shadow
        cv2.putText(frame, label, (cx + 2, cy + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (0, 0, 0), 5, cv2.LINE_AA)
        # Foreground
        cv2.putText(frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    color, 3, cv2.LINE_AA)
