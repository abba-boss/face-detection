"""
Real-time recognition loop.

Opens the webcam, detects faces on every frame, runs identification,
and overlays the result.  Press Q or ESC to exit.

Smoothing
---------
Results are smoothed over a rolling window (_SMOOTH_WINDOW frames):
  - decision   = majority vote  (>50 % AUTHORIZED → show AUTHORIZED)
  - similarity = EMA-weighted average (α=0.4 — recent frames weighted higher)
  - username   = most frequent authorised identity in the window

EMA weighting means a single drop-out frame near the threshold no longer
causes a visible flicker, while a sustained change (e.g. a different
person stepping in) is reflected within 2–3 frames rather than waiting
for the full window to flush.

Debug mode
----------
Pass debug=True to RecognitionRunner.run() to print the full per-user
score table in the terminal on every frame.  Never enable this in
production — it reveals all enrolled identities.

Liveness gate (--liveness)
--------------------------
When liveness=True, the first time a face crosses the AUTHORIZED threshold
the main loop pauses and a LivenessSession challenge runs in the same
camera window.  The recognition result is only promoted to AUTHORIZED if
the challenge returns LIVE.  A FAILED or TIMEOUT result keeps the face
labelled UNKNOWN for a 3-second cooldown before the gate can be
triggered again for that face slot.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2
import numpy as np

from app.camera import Camera
from app.config import Settings
from app.detection import FaceDetector
from app.liveness.detector import LivenessDetector, LivenessState
from app.liveness.drawing import draw_liveness_overlay
from app.recognition.recognizer import Recognizer, RecognitionResult
from app.security import get_logger
from app.storage import FaceStore
from app.utils.drawing import draw_recognition_label, draw_guide_text


_SMOOTH_WINDOW    = 5
_FPS_WINDOW       = 30    # frames to average for FPS display
_LIVENESS_COOLDOWN = 3.0  # seconds before the gate can re-trigger after failure

# EMA smoothing — weight applied to the most-recent frame relative to older ones.
# 0.0 = plain mean (all frames equal weight)
# 1.0 = only the latest frame counts (no smoothing)
# 0.4 is a good default: recent frames matter more but older frames still
# dampen transient spikes near the threshold.
_EMA_ALPHA = 0.4


class RecognitionRunner:
    """Drives the live recognition UI."""

    def __init__(self, settings: Settings, detector: FaceDetector,
                 store: FaceStore):
        self._settings = settings
        self._recognizer = Recognizer(settings, store)
        self._detector = detector
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )
        self._buffers: Dict[int, Deque[RecognitionResult]] = {}

    def run(self, debug: bool = False, liveness: bool = False) -> None:
        self._log.info(
            "Recognition started  debug=%s  liveness=%s", debug, liveness
        )
        print("\nUbuntu FaceAuth — Recognition")
        if liveness:
            print("Liveness gate ENABLED — you will be challenged before access is granted.")
        print("Press Q or ESC to exit.\n")

        camera = Camera(self._settings)
        try:
            camera.open()
        except RuntimeError as exc:
            self._log.error("Camera error: %s", exc)
            print(f"[ERROR] {exc}")
            return

        last_status: Optional[str] = None
        frame_times: Deque[float] = deque(maxlen=_FPS_WINDOW)

        # ── Liveness gate state (per face slot) ──────────────────────────
        # liveness_passed[idx]  → True once that slot cleared the challenge
        # liveness_cooldown[idx]→ monotonic timestamp until gate re-arms
        liveness_passed:   Dict[int, bool]  = {}
        liveness_cooldown: Dict[int, float] = {}

        try:
            while True:
                t0 = time.monotonic()
                ok, frame = camera.read()
                if not ok or frame is None:
                    continue

                faces = self._detector.detect(frame)

                if not faces:
                    draw_guide_text(frame, "No face detected", (0, 165, 255))
                    self._buffers.clear()
                    liveness_passed.clear()
                    liveness_cooldown.clear()
                else:
                    # Prune buffers for faces that left the frame
                    active = set(range(len(faces)))
                    for k in [k for k in self._buffers if k not in active]:
                        del self._buffers[k]
                    for k in [k for k in liveness_passed if k not in active]:
                        del liveness_passed[k]
                    for k in [k for k in liveness_cooldown if k not in active]:
                        del liveness_cooldown[k]

                    for idx, face in enumerate(faces):
                        if face.embedding is None:
                            draw_guide_text(frame, "Generating embedding…",
                                            (0, 165, 255))
                            continue

                        raw = self._recognizer.identify(face.embedding)

                        # Debug: print full score table (never in production)
                        if debug and raw.all_scores:
                            scores_str = "  ".join(
                                f"{u}:{s:.3f}"
                                for u, s in sorted(
                                    raw.all_scores.items(),
                                    key=lambda x: x[1], reverse=True
                                )
                            )
                            print(f"\r[DEBUG] {scores_str}          ",
                                  end="", flush=True)

                        buf = self._buffers.setdefault(
                            idx, deque(maxlen=_SMOOTH_WINDOW)
                        )
                        buf.append(raw)
                        smoothed = _smooth(buf)

                        # ── Liveness gate ─────────────────────────────────
                        if liveness and smoothed.authorized:
                            now = time.monotonic()
                            already_passed  = liveness_passed.get(idx, False)
                            in_cooldown     = now < liveness_cooldown.get(idx, 0.0)

                            if not already_passed and not in_cooldown:
                                # Face just crossed the threshold — run challenge
                                self._log.info(
                                    "Liveness gate triggered for face slot %d "
                                    "(user=%s)", idx, smoothed.username
                                )
                                print(f"\n[LIVENESS] Challenge started for "
                                      f"{smoothed.username} …")
                                cv2.destroyWindow("Ubuntu FaceAuth — Recognition")

                                live_result = _run_liveness_inline(
                                    camera, self._detector, self._settings
                                )

                                if live_result == LivenessState.LIVE:
                                    liveness_passed[idx] = True
                                    self._log.info(
                                        "Liveness PASSED for face slot %d", idx
                                    )
                                    print("[LIVENESS] PASSED — access granted.")
                                else:
                                    # Failed or timed out — suppress AUTHORIZED
                                    # and impose a cooldown before re-triggering
                                    liveness_passed[idx] = False
                                    liveness_cooldown[idx] = (
                                        time.monotonic() + _LIVENESS_COOLDOWN
                                    )
                                    smoothed = RecognitionResult(
                                        authorized=False,
                                        username=None,
                                        similarity=smoothed.similarity,
                                    )
                                    # Reset smoothing buffer so the gate
                                    # doesn't immediately re-trigger
                                    self._buffers.pop(idx, None)
                                    self._log.warning(
                                        "Liveness FAILED for face slot %d "
                                        "(result=%s) — cooldown %.1fs",
                                        idx, live_result.name,
                                        _LIVENESS_COOLDOWN,
                                    )
                                    print(
                                        f"[LIVENESS] {live_result.name} — "
                                        f"access denied. "
                                        f"Retry in {_LIVENESS_COOLDOWN:.0f}s."
                                    )

                            elif in_cooldown and not already_passed:
                                # Still cooling down — show UNKNOWN
                                remaining = liveness_cooldown[idx] - time.monotonic()
                                smoothed = RecognitionResult(
                                    authorized=False,
                                    username=None,
                                    similarity=smoothed.similarity,
                                )
                                draw_guide_text(
                                    frame,
                                    f"Liveness failed — retry in "
                                    f"{remaining:.1f}s",
                                    (0, 0, 220),
                                )

                        # ── Draw label ────────────────────────────────────
                        draw_recognition_label(
                            frame, face.bbox,
                            authorized=smoothed.authorized,
                            username=smoothed.username,
                            similarity=smoothed.similarity,
                            font_scale=self._settings.font_scale,
                            kps=face.kps,
                            show_landmarks=self._settings.show_landmarks,
                        )

                        if not debug:
                            if smoothed.authorized:
                                status = (
                                    f"AUTHORIZED  User: {smoothed.username}"
                                    f"  Similarity: {smoothed.similarity:.2f}"
                                )
                            else:
                                status = (
                                    f"UNKNOWN"
                                    f"  Similarity: {smoothed.similarity:.2f}"
                                )
                            if status != last_status:
                                print(f"\r{status}          ",
                                      end="", flush=True)
                                last_status = status

                cv2.imshow("Ubuntu FaceAuth — Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

                # FPS tracking
                frame_times.append(time.monotonic() - t0)
                if len(frame_times) == _FPS_WINDOW:
                    fps = len(frame_times) / sum(frame_times)
                    h, w = frame.shape[:2]
                    fps_text = f"{fps:.0f} FPS"
                    cv2.putText(frame, fps_text,
                                (w - 80, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, fps_text,
                                (w - 80, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 200, 200), 1, cv2.LINE_AA)

        except KeyboardInterrupt:
            pass
        finally:
            print()
            camera.release()
            self._log.info("Recognition stopped")


# ── Helpers ──────────────────────────────────────────────────────────────

def _run_liveness_inline(
    camera: Camera,
    detector: FaceDetector,
    settings: Settings,
) -> LivenessState:
    """
    Run a liveness challenge reusing the already-open camera.

    Opens a dedicated window, runs the LivenessDetector state machine
    frame-by-frame, draws the overlay, and returns the terminal state.
    The caller is responsible for re-opening the recognition window
    afterward (imshow with the recognition window name on the next frame).
    """
    cfg = settings.liveness_config()
    ld  = LivenessDetector(cfg)

    terminal = {LivenessState.LIVE, LivenessState.FAILED, LivenessState.TIMEOUT}
    hold_frames   = max(1, int(2.0 * settings.camera_fps))
    hold_counter  = 0
    result        = None

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue

        faces = detector.detect(frame)
        kps   = faces[0].kps if faces and faces[0].kps is not None else None
        result = ld.update(kps)

        draw_liveness_overlay(frame, result, font_scale=settings.font_scale)
        cv2.imshow("Ubuntu FaceAuth — Liveness", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow("Ubuntu FaceAuth — Liveness")
            return LivenessState.FAILED

        if result.state in terminal:
            hold_counter += 1
            if hold_counter >= hold_frames:
                break

    cv2.destroyWindow("Ubuntu FaceAuth — Liveness")
    return result.state if result else LivenessState.FAILED


def _smooth(buf: Deque[RecognitionResult],
            alpha: float = _EMA_ALPHA) -> RecognitionResult:
    """
    Derive a stable result from a rolling buffer using EMA weighting.

    similarity
        Exponential moving average over the window — most-recent frame
        carries the highest weight.  Weights are geometric:
            w_i = alpha * (1 - alpha)^(n-1-i)   for i = 0 … n-1
        then normalised to sum to 1.  When alpha=0 this degenerates to a
        plain mean; when alpha=1 only the latest frame is used.

    authorized
        Majority vote across the window (>50 % AUTHORIZED → AUTHORIZED).
        Unchanged from Phase 1 — keeps the decision boundary stable.

    username
        Most frequent authorised identity in the window.
    """
    if not buf:
        return RecognitionResult(authorized=False, username=None, similarity=0.0)

    n = len(buf)
    # Build geometric weights oldest→newest: w_i = alpha*(1-alpha)^(n-1-i)
    # For n=1 this collapses to weight=[1.0].
    weights = np.array(
        [alpha * (1.0 - alpha) ** (n - 1 - i) for i in range(n)],
        dtype=np.float64,
    )
    w_sum = weights.sum()
    if w_sum < 1e-12:
        weights = np.ones(n, dtype=np.float64)
        w_sum = float(n)
    weights /= w_sum

    sims = np.array([r.similarity for r in buf], dtype=np.float64)
    ema_sim = float(np.dot(weights, sims))

    auth_count = sum(1 for r in buf if r.authorized)
    majority_auth = auth_count > n / 2

    best_username: Optional[str] = None
    if majority_auth:
        names = [r.username for r in buf if r.authorized and r.username]
        if names:
            best_username = max(set(names), key=names.count)

    return RecognitionResult(
        authorized=majority_auth,
        username=best_username,
        similarity=ema_sim,
    )
