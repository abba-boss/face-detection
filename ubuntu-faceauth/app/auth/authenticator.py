"""
V3 — Ubuntu FaceAuth Authentication layer.

Provides a single-attempt, user-specific biometric authentication flow:

    Camera → Liveness → Face Recognition → Identity Verification

This is an ADDITIONAL authentication method.  It does not touch, replace,
or interact with Ubuntu passwords, PAM, GDM, /etc/passwd, or /etc/shadow.

Flow
----
1.  Pre-flight: confirm the requested user is enrolled.  Fail fast if not.
2.  Liveness challenge: run LivenessSession.  If not LIVE → DENIED.
3.  Face capture: open (or reuse) camera, detect a single good face frame.
    Retries up to auth_max_frames times before giving up.
4.  Recognition: compare captured embedding against the enrolled user only.
    Uses the same cosine-similarity threshold as the recognition runner.
5.  Identity verification: the recognised identity must match --user exactly.
    A mismatch (someone else's face) → DENIED.
6.  Return AuthResult with outcome, similarity score, and a human message.

Exit semantics (for CLI):
    AuthOutcome.SUCCESS  → exit code 0
    anything else        → exit code 1

Design principles
-----------------
- Reuses LivenessSession, FaceDetector, Recognizer, FaceStore unchanged.
- AuthSession owns no camera directly — it delegates to LivenessSession
  and a lightweight capture helper.
- Pure logic is separated from I/O so the class is fully unit-testable
  by injecting mocks.
- No passwords are created, read, or stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from app.camera import Camera
from app.config import Settings
from app.detection import FaceDetector
from app.liveness import LivenessSession, LivenessState
from app.recognition.recognizer import Recognizer
from app.security import get_logger
from app.storage import FaceStore
from app.utils.drawing import draw_guide_text


# ── Maximum frames to scan for a usable face during capture ──────────────
_DEFAULT_MAX_CAPTURE_FRAMES = 60


class AuthOutcome(Enum):
    SUCCESS        = auto()   # liveness passed + face matched --user
    DENIED_NOT_ENROLLED   = auto()   # user has no enrolled embedding
    DENIED_LIVENESS       = auto()   # liveness challenge failed or timed out
    DENIED_NO_FACE        = auto()   # could not detect a usable face
    DENIED_MISMATCH       = auto()   # face recognised but wrong identity
    DENIED_BELOW_THRESHOLD = auto()  # similarity too low (UNKNOWN face)
    ERROR                 = auto()   # unexpected hardware / model failure


@dataclass
class AuthResult:
    outcome:    AuthOutcome
    username:   str               # the --user that was requested
    message:    str               # human-readable explanation
    similarity: float = 0.0       # cosine similarity of the captured face
    matched_as: Optional[str] = None  # identity the model recognised (may differ)

    @property
    def success(self) -> bool:
        return self.outcome == AuthOutcome.SUCCESS


class AuthSession:
    """
    Single-attempt biometric authentication for one user.

    Parameters
    ----------
    settings  : Settings
    detector  : FaceDetector  (already loaded)
    store     : FaceStore
    max_capture_frames : int
        How many camera frames to scan before giving up on face capture.
    """

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetector,
        store: FaceStore,
        max_capture_frames: int = _DEFAULT_MAX_CAPTURE_FRAMES,
    ):
        self._settings = settings
        self._detector = detector
        self._store = store
        self._max_frames = max_capture_frames
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, username: str) -> AuthResult:
        """
        Execute the full authentication flow for *username*.

        Returns an AuthResult.  Never raises — all exceptions are caught
        and mapped to AuthOutcome.ERROR.
        """
        self._log.info("Authentication started for user '%s'", username)

        # ── Step 1: pre-flight ────────────────────────────────────────────
        preflight = self._check_enrolled(username)
        if preflight is not None:
            return preflight

        # ── Step 2: liveness ─────────────────────────────────────────────
        liveness_result = self._run_liveness()
        if liveness_result is not None:
            return liveness_result

        # ── Step 3 + 4 + 5: capture → recognise → verify ─────────────────
        try:
            return self._capture_and_verify(username)
        except Exception as exc:
            self._log.error(
                "Authentication error for '%s': %s", username, exc
            )
            return AuthResult(
                outcome=AuthOutcome.ERROR,
                username=username,
                message=f"Authentication error: {exc}",
            )

    # ── Private steps ─────────────────────────────────────────────────────

    def _check_enrolled(self, username: str) -> Optional[AuthResult]:
        """Return a DENIED result if the user is not enrolled, else None."""
        if not self._store.is_enrolled(username):
            msg = f"User '{username}' is not enrolled — run: python main.py enroll --user {username}"
            self._log.warning(
                "Authentication denied — '%s' not enrolled", username
            )
            return AuthResult(
                outcome=AuthOutcome.DENIED_NOT_ENROLLED,
                username=username,
                message=msg,
            )
        return None

    def _run_liveness(self) -> Optional[AuthResult]:
        """
        Run the liveness challenge.  Return a DENIED result on failure,
        None on success (LIVE).
        """
        self._log.info("Starting liveness challenge")
        session = LivenessSession(self._settings, self._detector)
        state = session.run()

        if state == LivenessState.LIVE:
            self._log.info("Liveness challenge passed")
            return None

        reason = {
            LivenessState.FAILED:  "face was lost during the challenge",
            LivenessState.TIMEOUT: "challenge timed out",
        }.get(state, state.name.lower())

        self._log.warning("Liveness challenge failed: %s", reason)
        return AuthResult(
            outcome=AuthOutcome.DENIED_LIVENESS,
            username="",   # not yet confirmed
            message=f"Liveness check failed — {reason}.",
        )

    def _capture_and_verify(self, username: str) -> AuthResult:
        """
        Open camera, grab one good face embedding, run recognition,
        and verify the identity matches *username*.
        """
        self._log.info(
            "Starting face capture for '%s' (max %d frames)",
            username, self._max_frames,
        )
        print("\nUbuntu FaceAuth — Face Capture")
        print("Look at the camera…\n")

        camera = Camera(self._settings)
        try:
            camera.open()
        except RuntimeError as exc:
            self._log.error("Camera failed to open: %s", exc)
            return AuthResult(
                outcome=AuthOutcome.ERROR,
                username=username,
                message=f"Camera error: {exc}",
            )

        embedding: Optional[np.ndarray] = None

        try:
            for _ in range(self._max_frames):
                ok, frame = camera.read()
                if not ok or frame is None:
                    continue

                faces = self._detector.detect(frame)

                if faces and faces[0].embedding is not None:
                    # Accept the first sharp, embedded face
                    face = faces[0]
                    if face.blur_score >= self._settings.enrollment_blur_threshold:
                        embedding = face.embedding.copy()
                        draw_guide_text(frame, "Face captured ✓", (0, 220, 0))
                    else:
                        draw_guide_text(
                            frame,
                            f"Hold still — sharpness "
                            f"{face.blur_score:.0f}/"
                            f"{self._settings.enrollment_blur_threshold:.0f}",
                            (0, 165, 255),
                        )
                elif not faces:
                    draw_guide_text(frame, "No face detected", (0, 165, 255))
                else:
                    draw_guide_text(frame, "Generating embedding…", (0, 165, 255))

                cv2.imshow("Ubuntu FaceAuth — Authentication", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self._log.info("Authentication cancelled by user")
                    return AuthResult(
                        outcome=AuthOutcome.DENIED_NO_FACE,
                        username=username,
                        message="Authentication cancelled.",
                    )

                if embedding is not None:
                    # Hold the captured frame briefly
                    cv2.waitKey(500)
                    break

        finally:
            camera.release()

        if embedding is None:
            self._log.warning(
                "No usable face captured for '%s' after %d frames",
                username, self._max_frames,
            )
            return AuthResult(
                outcome=AuthOutcome.DENIED_NO_FACE,
                username=username,
                message="Could not capture a clear face — try better lighting.",
            )

        # ── Recognise ────────────────────────────────────────────────────
        recognizer = Recognizer(self._settings, self._store)
        result = recognizer.identify(embedding)

        self._log.info(
            "Recognition result — authorized=%s  matched_as=%s  similarity=%.3f",
            result.authorized, result.username, result.similarity,
        )

        # ── Verify identity ───────────────────────────────────────────────
        if not result.authorized:
            return AuthResult(
                outcome=AuthOutcome.DENIED_BELOW_THRESHOLD,
                username=username,
                message=(
                    f"Face not recognised (similarity {result.similarity:.2f} "
                    f"< threshold {self._settings.recognition_threshold:.2f})."
                ),
                similarity=result.similarity,
                matched_as=result.username,
            )

        if result.username != username:
            self._log.warning(
                "Identity mismatch — requested '%s', recognised '%s'",
                username, result.username,
            )
            return AuthResult(
                outcome=AuthOutcome.DENIED_MISMATCH,
                username=username,
                message=(
                    f"Identity mismatch — face does not belong to '{username}'."
                ),
                similarity=result.similarity,
                matched_as=result.username,
            )

        # ── SUCCESS ───────────────────────────────────────────────────────
        self._log.info(
            "Authentication SUCCESS for '%s'  similarity=%.3f",
            username, result.similarity,
        )
        return AuthResult(
            outcome=AuthOutcome.SUCCESS,
            username=username,
            message=f"Authentication successful — welcome, {username}.",
            similarity=result.similarity,
            matched_as=result.username,
        )
