"""
Face enrollment session.

Workflow:
  1. Open webcam (discards warmup frames automatically via Camera.open).
  2. 3-second countdown so the user can position their face.
  3. Detect faces frame-by-frame.
  4. Reject frames with 0 or >1 faces, faces < min_face_size,
     or face crops below the blur threshold (Laplacian variance).
  5. Collect settings.enrollment_samples valid embeddings.
  6. Average + L2-normalise → one representative embedding.
  7. Save to FaceStore.  No raw images are stored.

Press Q or ESC to cancel at any time.
"""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from app.camera import Camera
from app.config import Settings
from app.detection import FaceDetector
from app.security import get_logger
from app.storage import FaceStore
from app.utils.drawing import draw_bbox, draw_guide_text


class EnrollmentSession:
    """Runs a complete enrollment for one user."""

    def __init__(self, settings: Settings, detector: FaceDetector,
                 store: FaceStore):
        self._settings = settings
        self._detector = detector
        self._store = store
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )

    def run(self, username: str) -> bool:
        """
        Launch the enrollment UI.

        Returns True on success, False if the user cancelled or the
        session exhausted all attempts.
        """
        self._log.info("Enrollment started for user '%s'", username)
        print(f"\nUbuntu FaceAuth — Enrolling: {username}")
        print("Position your face in the frame.  Press Q or ESC to cancel.\n")

        camera = Camera(self._settings)
        try:
            camera.open()
        except RuntimeError as exc:
            self._log.error("Camera error: %s", exc)
            print(f"[ERROR] {exc}")
            return False

        embeddings: List[np.ndarray] = []
        attempts = 0
        target = self._settings.enrollment_samples
        max_attempts = self._settings.enrollment_max_attempts

        try:
            # ── Countdown ────────────────────────────────────────────────
            if not _countdown(camera, seconds=3):
                print("\nEnrollment cancelled.")
                return False

            # ── Collection loop ──────────────────────────────────────────
            while len(embeddings) < target and attempts < max_attempts:
                ok, frame = camera.read()
                if not ok or frame is None:
                    self._log.warning("Empty frame during enrollment")
                    continue

                attempts += 1

                faces = self._detector.detect(frame)

                # ── Guidance ─────────────────────────────────────────────
                if len(faces) == 0:
                    msg = "No face detected — move closer"
                    status_color = (0, 165, 255)   # orange

                elif len(faces) > 1:
                    msg = "Multiple faces — please be alone in frame"
                    status_color = (0, 0, 255)     # red

                else:
                    face = faces[0]
                    blur_ok = face.blur_score >= self._settings.enrollment_blur_threshold

                    if not blur_ok:
                        msg = (f"Hold still — image blurry "
                               f"(sharpness: {face.blur_score:.0f}/"
                               f"{self._settings.enrollment_blur_threshold:.0f})")
                        status_color = (0, 165, 255)
                        draw_bbox(frame, face.bbox, status_color)

                    elif face.embedding is None:
                        msg = "Face detected — generating embedding…"
                        status_color = (0, 165, 255)
                        draw_bbox(frame, face.bbox, status_color)

                    else:
                        embeddings.append(face.embedding.copy())
                        n = len(embeddings)
                        msg = f"Collecting samples: {n}/{target}"
                        status_color = (0, 200, 0)  # green
                        print(f"  Collecting samples: {n}/{target}")
                        draw_bbox(frame, face.bbox, status_color)

                draw_guide_text(frame, msg, status_color)
                _progress_bar(frame, len(embeddings), target)
                cv2.imshow("Ubuntu FaceAuth — Enrollment", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self._log.info("Enrollment cancelled by user")
                    print("\nEnrollment cancelled.")
                    return False

        except KeyboardInterrupt:
            print("\nEnrollment interrupted.")
            return False
        finally:
            camera.release()

        if len(embeddings) < target:
            print(
                f"\n[ERROR] Only collected {len(embeddings)}/{target} samples "
                f"after {max_attempts} frames.  Enrollment failed.\n"
                f"Tips:\n"
                f"  • Ensure good, even lighting on your face.\n"
                f"  • Hold still and face the camera directly.\n"
                f"  • If you see 'Hold still', increase stability.\n"
                f"  • Lower blur threshold: "
                f"settings.enrollment_blur_threshold = 10.0"
            )
            self._log.error(
                "Enrollment failed — only %d/%d samples in %d attempts",
                len(embeddings), target, attempts,
            )
            return False

        # ── Aggregate ────────────────────────────────────────────────────
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
        self._store.save(username, mean_emb)

        print(f"\nEnrollment completed successfully for user '{username}'.")
        self._log.info(
            "Enrollment completed for '%s' (%d samples, %d attempts)",
            username, len(embeddings), attempts,
        )
        return True


# ── Private helpers ───────────────────────────────────────────────────────

def _countdown(camera: Camera, seconds: int = 3) -> bool:
    """
    Show a live countdown overlay on the camera feed.

    Returns False if the user pressed Q/ESC during the countdown.
    """
    deadline = time.monotonic() + seconds
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue

        remaining = max(0, int(deadline - time.monotonic()) + 1)
        if remaining == 0:
            return True

        msg = f"Get ready… {remaining}"
        draw_guide_text(frame, msg, (255, 200, 0), y_offset=50)

        # Large centred countdown digit
        h, w = frame.shape[:2]
        cv2.putText(
            frame, str(remaining),
            (w // 2 - 30, h // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 3.0,
            (0, 0, 0), 8, cv2.LINE_AA,
        )
        cv2.putText(
            frame, str(remaining),
            (w // 2 - 30, h // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 3.0,
            (255, 200, 0), 4, cv2.LINE_AA,
        )

        cv2.imshow("Ubuntu FaceAuth — Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False


def _progress_bar(frame: np.ndarray, current: int, total: int) -> None:
    """Horizontal progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 12
    y = h - bar_h - 4
    cv2.rectangle(frame, (10, y), (w - 10, y + bar_h), (60, 60, 60), -1)
    if total > 0:
        fill_w = int((w - 20) * current / total)
        cv2.rectangle(frame, (10, y), (10 + fill_w, y + bar_h), (0, 200, 0), -1)
    cv2.rectangle(frame, (10, y), (w - 10, y + bar_h), (200, 200, 200), 1)
