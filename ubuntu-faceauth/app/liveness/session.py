"""
LivenessSession — camera-driven wrapper around LivenessDetector.

Mirrors the structure of EnrollmentSession and RecognitionRunner:
  - opens the webcam via Camera
  - calls FaceDetector.detect() per frame
  - passes kps to LivenessDetector.update()
  - draws the overlay
  - returns the final LivenessState when done

Usage from main.py:
    session = LivenessSession(settings, detector)
    state = session.run()
    if state == LivenessState.LIVE:
        ...
"""

from __future__ import annotations

import cv2

from app.camera import Camera
from app.config import Settings
from app.detection import FaceDetector
from app.liveness.detector import LivenessDetector, LivenessState
from app.liveness.drawing import draw_liveness_overlay
from app.security import get_logger


# How long (seconds) to hold the result frame visible before closing
_RESULT_HOLD_SECONDS = 2.0


class LivenessSession:
    """Run a single liveness challenge session with the webcam."""

    def __init__(self, settings: Settings, detector: FaceDetector):
        self._settings = settings
        self._detector = detector
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )

    def run(self) -> LivenessState:
        """
        Open the webcam and run the liveness challenge.

        Returns the terminal LivenessState:
          LIVE    — challenge passed
          FAILED  — face was lost during challenge
          TIMEOUT — user did not complete the turn in time
        """
        cfg     = self._settings.liveness_config()
        ld      = LivenessDetector(cfg)
        camera  = Camera(self._settings)

        self._log.info("Liveness session started")
        print("\nUbuntu FaceAuth — Liveness Check")
        print("Press Q or ESC to cancel.\n")

        try:
            camera.open()
        except RuntimeError as exc:
            self._log.error("Camera error: %s", exc)
            print(f"[ERROR] {exc}")
            return LivenessState.FAILED

        terminal_states = {
            LivenessState.LIVE,
            LivenessState.FAILED,
            LivenessState.TIMEOUT,
        }
        result = None
        hold_frames = int(_RESULT_HOLD_SECONDS * self._settings.camera_fps)
        hold_counter = 0

        try:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    continue

                faces = self._detector.detect(frame)

                # Pass the first face's kps (or None) to the state machine
                kps = None
                if faces and faces[0].kps is not None:
                    kps = faces[0].kps

                result = ld.update(kps)
                draw_liveness_overlay(frame, result,
                                      font_scale=self._settings.font_scale)

                cv2.imshow("Ubuntu FaceAuth — Liveness", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self._log.info("Liveness cancelled by user")
                    print("\nLiveness check cancelled.")
                    return LivenessState.FAILED

                # Hold the result frame briefly so the user can see the outcome
                if result.state in terminal_states:
                    hold_counter += 1
                    if hold_counter >= hold_frames:
                        break

        except KeyboardInterrupt:
            print("\nLiveness check interrupted.")
            return LivenessState.FAILED
        finally:
            camera.release()

        final = result.state if result else LivenessState.FAILED
        self._log.info("Liveness session ended — result=%s", final.name)

        status = {
            LivenessState.LIVE:    "LIVE — challenge passed.",
            LivenessState.FAILED:  "FAILED — face was lost.",
            LivenessState.TIMEOUT: "TIMEOUT — turn not completed in time.",
        }.get(final, final.name)
        print(f"\nLiveness: {status}")

        return final
