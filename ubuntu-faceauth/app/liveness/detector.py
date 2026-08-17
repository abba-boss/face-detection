"""
Phase 2A — Liveness challenge-response module.

Challenge: "Turn your head LEFT"

Algorithm
---------
InsightFace provides 5-point facial landmarks (kps) in every DetectedFace:

    kps[0] = left eye   (x, y)
    kps[1] = right eye  (x, y)
    kps[2] = nose tip   (x, y)
    kps[3] = mouth-left (x, y)
    kps[4] = mouth-right(x, y)

Head pose is estimated via the *nose offset ratio* (NOR):

    eye_mid_x  = (kps[0].x + kps[1].x) / 2
    eye_span   = kps[1].x - kps[0].x          # always positive
    NOR        = (nose_x - eye_mid_x) / eye_span

Interpretation:
    NOR ≈  0.0   →  face looking straight ahead
    NOR <  -LEFT_THRESHOLD   →  turned LEFT   (nose left of eye midpoint)
    NOR >  +RIGHT_THRESHOLD  →  turned RIGHT

A valid LEFT-turn is confirmed when:
  1. A baseline NOR reading is established (face roughly frontal: |NOR| < 0.15)
  2. NOR drops below  baseline − LEFT_THRESHOLD  for at least
     MIN_CONFIRMED_FRAMES  consecutive frames
  3. The whole challenge completes within TIMEOUT_SECONDS

State machine
-------------
WAITING          → no face / not ready; waiting for a stable frontal face
CHALLENGE_ACTIVE → frontal face locked; user is prompted to turn left
LIVE             → challenge passed (sufficient left movement confirmed)
FAILED           → face lost during challenge or wrong movement
TIMEOUT          → challenge did not complete within the time limit

This module is:
  - Independent from FaceStore and the recognition threshold.
  - Pure Python / numpy — no camera I/O of its own.
  - Fully unit-testable with synthetic landmark arrays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ── Tunable constants (overridden via LivenessConfig) ────────────────────

_DEFAULT_TIMEOUT        = 8.0   # seconds the user has to complete the turn
_DEFAULT_LEFT_THRESHOLD = 0.18  # NOR drop required to count as a LEFT turn
_DEFAULT_FRONTAL_MAX    = 0.15  # |NOR| must be below this for baseline lock
_DEFAULT_MIN_FRAMES     = 3     # consecutive frames NOR must hold to confirm


# ── Public types ─────────────────────────────────────────────────────────

class LivenessState(Enum):
    WAITING          = auto()   # waiting for a stable frontal face
    CHALLENGE_ACTIVE = auto()   # challenge issued, monitoring movement
    LIVE             = auto()   # challenge passed — person is live
    FAILED           = auto()   # face lost / wrong movement
    TIMEOUT          = auto()   # ran out of time


@dataclass
class LivenessConfig:
    """All tunable parameters for the liveness detector."""
    timeout_seconds:    float = _DEFAULT_TIMEOUT
    left_threshold:     float = _DEFAULT_LEFT_THRESHOLD
    frontal_max_nor:    float = _DEFAULT_FRONTAL_MAX
    min_confirm_frames: int   = _DEFAULT_MIN_FRAMES


@dataclass
class LivenessResult:
    """Returned by LivenessDetector.update() on every frame."""
    state:         LivenessState
    message:       str           # human-readable instruction / status
    nor:           Optional[float] = None   # current nose-offset ratio
    baseline_nor:  Optional[float] = None   # locked baseline NOR
    elapsed:       float = 0.0              # seconds since challenge start
    confirm_count: int   = 0                # consecutive confirming frames
    timeout:       float = _DEFAULT_TIMEOUT # total timeout for progress bar


# ── Core detector ─────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Single-challenge liveness state machine.

    Usage:
        detector = LivenessDetector()
        # Call once per frame, passing kps (np.ndarray shape (5,2)) or None:
        result = detector.update(kps)
        if result.state == LivenessState.LIVE:
            # liveness confirmed
    
    Call reset() to reuse for a new session.
    """

    def __init__(self, config: Optional[LivenessConfig] = None):
        self._cfg = config or LivenessConfig()
        self._state            = LivenessState.WAITING
        self._baseline_nor: Optional[float] = None
        self._challenge_start: Optional[float] = None
        self._confirm_count    = 0
        self._message          = "Position your face and look straight ahead"

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def state(self) -> LivenessState:
        return self._state

    def reset(self) -> None:
        """Return to WAITING state for a fresh challenge."""
        self._state            = LivenessState.WAITING
        self._baseline_nor     = None
        self._challenge_start  = None
        self._confirm_count    = 0
        self._message          = "Position your face and look straight ahead"

    def update(self, kps: Optional[np.ndarray]) -> LivenessResult:
        """
        Advance the state machine by one frame.

        Parameters
        ----------
        kps : np.ndarray of shape (5, 2) or None
            5-point landmarks from DetectedFace.kps.
            Pass None when no face is detected in this frame.

        Returns
        -------
        LivenessResult with the current state and a human-readable message.
        """
        now = time.monotonic()

        # ── Terminal states — no further transitions ──────────────────────
        if self._state in (LivenessState.LIVE,
                           LivenessState.FAILED,
                           LivenessState.TIMEOUT):
            return self._result(now, nor=None)

        # ── No face ───────────────────────────────────────────────────────
        if kps is None or not _valid_kps(kps):
            return self._handle_no_face(now)

        nor = _nose_offset_ratio(kps)

        # ── WAITING: look for a stable frontal face to lock baseline ─────
        if self._state == LivenessState.WAITING:
            return self._handle_waiting(now, nor)

        # ── CHALLENGE_ACTIVE: monitor for left turn ───────────────────────
        if self._state == LivenessState.CHALLENGE_ACTIVE:
            return self._handle_challenge(now, nor)

        return self._result(now, nor=nor)   # unreachable but safe

    # ── Private state handlers ────────────────────────────────────────────

    def _handle_no_face(self, now: float) -> LivenessResult:
        """Face disappeared — fail if challenge was active."""
        if self._state == LivenessState.CHALLENGE_ACTIVE:
            self._state   = LivenessState.FAILED
            self._message = "Face lost — liveness check failed"
        elif self._state == LivenessState.WAITING:
            self._message = "Position your face and look straight ahead"
        return self._result(now, nor=None)

    def _handle_waiting(self, now: float, nor: float) -> LivenessResult:
        """Lock baseline when face is frontal; then issue the challenge."""
        if abs(nor) <= self._cfg.frontal_max_nor:
            self._baseline_nor    = nor
            self._challenge_start = now
            self._confirm_count   = 0
            self._state           = LivenessState.CHALLENGE_ACTIVE
            self._message         = "Now slowly turn your head LEFT"
        else:
            self._message = "Look straight ahead to begin"
        return self._result(now, nor=nor)

    def _handle_challenge(self, now: float, nor: float) -> LivenessResult:
        """Detect a sufficient left-turn relative to the baseline."""
        # Timeout check first
        elapsed = now - self._challenge_start
        if elapsed >= self._cfg.timeout_seconds:
            self._state   = LivenessState.TIMEOUT
            self._message = "Timed out — please try again"
            return self._result(now, nor=nor, elapsed=elapsed)

        required = self._baseline_nor - self._cfg.left_threshold
        if nor <= required:
            self._confirm_count += 1
            self._message = (
                f"Hold… ({self._confirm_count}/{self._cfg.min_confirm_frames})"
            )
            if self._confirm_count >= self._cfg.min_confirm_frames:
                self._state   = LivenessState.LIVE
                self._message = "Liveness confirmed ✓"
        else:
            # Movement not yet sufficient — reset confirm streak but keep going
            self._confirm_count = 0
            remaining = max(0, int(self._cfg.timeout_seconds - elapsed))
            self._message = f"Turn LEFT  ({remaining}s remaining)"

        return self._result(now, nor=nor, elapsed=elapsed)

    def _result(self, now: float,
                nor: Optional[float],
                elapsed: Optional[float] = None) -> LivenessResult:
        if elapsed is None and self._challenge_start is not None:
            elapsed = now - self._challenge_start
        return LivenessResult(
            state         = self._state,
            message       = self._message,
            nor           = nor,
            baseline_nor  = self._baseline_nor,
            elapsed       = elapsed or 0.0,
            confirm_count = self._confirm_count,
            timeout       = self._cfg.timeout_seconds,
        )


# ── Pure helper functions (easy to unit-test) ─────────────────────────────

def _valid_kps(kps: np.ndarray) -> bool:
    """Return True when kps is a usable (5, 2) landmark array."""
    return (
        isinstance(kps, np.ndarray)
        and kps.shape == (5, 2)
        and np.all(np.isfinite(kps))
        and kps[1, 0] > kps[0, 0]   # right-eye x must be > left-eye x
    )


def _nose_offset_ratio(kps: np.ndarray) -> float:
    """
    Compute the nose-offset ratio (NOR).

    NOR = (nose_x - eye_mid_x) / eye_span

    Range: roughly [-0.6, +0.6] in practice.
      ≈  0.0  → frontal
      < -0.18 → meaningful left turn
      > +0.18 → meaningful right turn
    """
    left_eye_x  = float(kps[0, 0])
    right_eye_x = float(kps[1, 0])
    nose_x      = float(kps[2, 0])

    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    eye_span  = right_eye_x - left_eye_x          # always > 0 (validated above)

    return (nose_x - eye_mid_x) / eye_span
