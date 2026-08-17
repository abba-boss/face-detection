"""
Headless authentication session for PAM/GDM use.

Identical flow to AuthSession (liveness → face capture → recognition)
but with zero GUI dependencies:
  - No cv2.imshow(), cv2.waitKey(), or Qt calls anywhere.
  - No DISPLAY or WAYLAND_DISPLAY required.
  - Reads frames directly from /dev/video0 via cv2.VideoCapture
    (V4L2 kernel driver — works without a graphical session).
  - All progress is written to stdout/stderr so pam_exec.so can
    log it via the 'stdout' option.

Exit semantics (identical to AuthSession — main.py maps these to codes):
    AuthOutcome.SUCCESS  → exit code 0
    anything else        → exit code 1

Liveness algorithm
------------------
The same LivenessDetector state machine used by the GUI session:
  WAITING          → looking for a stable frontal face (|NOR| < 0.15)
  CHALLENGE_ACTIVE → prompts the user to turn left
  LIVE             → turn confirmed for ≥ 3 consecutive frames → PASS
  FAILED / TIMEOUT → liveness denied

Face capture
------------
After liveness passes, a separate short capture loop grabs the first
sharp frame with a valid embedding.  This mirrors _capture_and_verify()
in AuthSession but without any imshow calls.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from app.camera import Camera
from app.config import Settings
from app.detection import FaceDetector
from app.liveness.detector import LivenessDetector, LivenessState
from app.recognition.recognizer import Recognizer
from app.security import get_logger
from app.storage import FaceStore

# Re-use the same result types so main.py needs no changes.
from app.auth.authenticator import AuthOutcome, AuthResult

# How many sharp, valid embeddings to collect and average for recognition.
# Mirrors enrollment (which averages 10 samples) — more samples → better
# representation and higher cosine similarity against the stored mean.
_CAPTURE_SAMPLES = 8

# Maximum frames to scan while collecting _CAPTURE_SAMPLES embeddings.
# At 30 fps this is ~4 seconds of scanning time.
_MAX_CAPTURE_FRAMES = 120

# Frames to discard after liveness ends before starting face capture.
# After the head-turn the face is often mid-motion or at an angle.
# Discarding ~0.5 s of frames lets it settle back to frontal.
_POST_LIVENESS_SETTLE_FRAMES = 15

# How often (seconds) to print a progress heartbeat during the liveness loop.
# Keeps PAM/journald logs alive so the session doesn't look hung.
_HEARTBEAT_INTERVAL = 2.0


class HeadlessAuthSession:
    """
    Liveness + face recognition with no GUI.

    Designed to be called from pam_exec.so where no X/Wayland
    display is available.

    Parameters
    ----------
    settings  : Settings
    detector  : FaceDetector  (already loaded)
    store     : FaceStore
    """

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetector,
        store: FaceStore,
    ):
        self._settings = settings
        self._detector = detector
        self._store = store
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, username: str) -> AuthResult:
        """
        Execute full headless authentication for *username*.

        Returns AuthResult.  Never raises — all exceptions are caught
        and returned as AuthOutcome.ERROR so PAM always gets exit 1.
        """
        self._log.info("[headless] Authentication started for '%s'", username)
        print(f"[FaceAuth] Authenticating user: {username}")

        # ── Step 1: pre-flight ────────────────────────────────────────────
        if not self._store.is_enrolled(username):
            msg = f"User '{username}' is not enrolled."
            self._log.warning("[headless] %s", msg)
            print(f"[FaceAuth] {msg}")
            return AuthResult(
                outcome=AuthOutcome.DENIED_NOT_ENROLLED,
                username=username,
                message=msg,
            )

        # ── Step 2: open camera once for the whole session ────────────────
        camera = Camera(self._settings)
        try:
            camera.open()
        except RuntimeError as exc:
            msg = f"Camera error: {exc}"
            self._log.error("[headless] %s", msg)
            print(f"[FaceAuth] {msg}")
            return AuthResult(
                outcome=AuthOutcome.ERROR,
                username=username,
                message=msg,
            )

        try:
            # ── Step 3: liveness ─────────────────────────────────────────
            liveness_state = self._run_liveness(camera)

            if liveness_state != LivenessState.LIVE:
                reason = {
                    LivenessState.FAILED:  "face was lost during the challenge",
                    LivenessState.TIMEOUT: "challenge timed out",
                }.get(liveness_state, liveness_state.name.lower())
                msg = f"Liveness check failed — {reason}."
                self._log.warning("[headless] %s", msg)
                print(f"[FaceAuth] {msg}")
                return AuthResult(
                    outcome=AuthOutcome.DENIED_LIVENESS,
                    username=username,
                    message=msg,
                )

            self._log.info("[headless] Liveness passed")
            print("[FaceAuth] Liveness: PASSED")

            # ── Step 4: face capture + recognition + verify ───────────────
            return self._capture_and_verify(camera, username)

        except Exception as exc:
            msg = f"Unexpected error: {exc}"
            self._log.error("[headless] %s", msg, exc_info=True)
            print(f"[FaceAuth] {msg}")
            return AuthResult(
                outcome=AuthOutcome.ERROR,
                username=username,
                message=msg,
            )
        finally:
            camera.release()

    # ── Private: liveness (no GUI) ────────────────────────────────────────

    def _run_liveness(self, camera: Camera) -> LivenessState:
        """
        Run the liveness state machine headlessly.

        Reads frames from *camera*, feeds landmarks to LivenessDetector,
        prints text-only status updates, and returns the terminal state.
        """
        cfg = self._settings.liveness_config()
        ld  = LivenessDetector(cfg)

        terminal_states = {
            LivenessState.LIVE,
            LivenessState.FAILED,
            LivenessState.TIMEOUT,
        }

        print("[FaceAuth] Liveness: look straight at the camera, then turn LEFT")

        last_heartbeat = time.monotonic()
        last_state_msg = ""

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            faces = self._detector.detect(frame)
            kps = None
            if faces and faces[0].kps is not None:
                kps = faces[0].kps

            result = ld.update(kps)

            # Print status only when it changes — avoids log spam
            if result.message != last_state_msg:
                print(f"[FaceAuth] Liveness: {result.message}")
                last_state_msg = result.message

            # Periodic heartbeat so the log doesn't look hung
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                if result.state == LivenessState.CHALLENGE_ACTIVE:
                    remaining = max(
                        0.0, cfg.timeout_seconds - result.elapsed
                    )
                    print(
                        f"[FaceAuth] Liveness: waiting… "
                        f"{remaining:.0f}s remaining"
                    )
                last_heartbeat = now

            if result.state in terminal_states:
                break

        return result.state

    # ── Private: capture + recognise + verify (no GUI) ────────────────────

    def _capture_and_verify(
        self, camera: Camera, username: str
    ) -> AuthResult:
        """
        Collect multiple sharp face embeddings, average them, then recognise.

        Mirrors the enrollment strategy: N valid embeddings are gathered,
        stacked, and L2-normalised into a single mean vector before being
        compared against the stored embedding.  This gives cosine similarities
        consistent with the GUI authenticate mode (0.65–0.85) rather than the
        single-frame approach which can yield 0.20–0.40 on a non-frontal face.
        """
        self._log.info(
            "[headless] Face capture starting for '%s' "
            "(need %d samples, max %d frames)",
            username, _CAPTURE_SAMPLES, _MAX_CAPTURE_FRAMES,
        )
        print(
            f"[FaceAuth] Face capture: look straight at the camera "
            f"(collecting {_CAPTURE_SAMPLES} samples)…"
        )

        # ── Settle: discard frames while face returns to frontal ──────────
        settled = 0
        for _ in range(_MAX_CAPTURE_FRAMES):
            ok, _ = camera.read()
            if ok:
                settled += 1
            if settled >= _POST_LIVENESS_SETTLE_FRAMES:
                break

        # ── Collect multiple sharp embeddings ─────────────────────────────
        embeddings = []
        best_blur   = 0.0
        frames_read = 0

        for _ in range(_MAX_CAPTURE_FRAMES):
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            frames_read += 1

            faces = self._detector.detect(frame)
            if not faces:
                continue

            face = faces[0]
            if face.embedding is None:
                continue

            if face.blur_score < self._settings.enrollment_blur_threshold:
                # Too blurry — skip but keep trying
                continue

            embeddings.append(face.embedding.copy())
            if face.blur_score > best_blur:
                best_blur = face.blur_score

            n = len(embeddings)
            print(
                f"[FaceAuth] Sample {n}/{_CAPTURE_SAMPLES}  "
                f"blur={face.blur_score:.1f}"
            )
            self._log.info(
                "[headless] Sample %d/%d  blur=%.1f  sim_placeholder",
                n, _CAPTURE_SAMPLES, face.blur_score,
            )

            if n >= _CAPTURE_SAMPLES:
                break

        if not embeddings:
            msg = (
                "Could not capture a clear face — "
                "try better lighting or move closer."
            )
            self._log.warning("[headless] %s  frames_read=%d", msg, frames_read)
            print(f"[FaceAuth] {msg}")
            return AuthResult(
                outcome=AuthOutcome.DENIED_NO_FACE,
                username=username,
                message=msg,
            )

        # ── Average + L2-normalise (same as enrollment) ───────────────────
        stacked  = np.stack(embeddings, axis=0)          # (N, 512)
        mean_emb = stacked.mean(axis=0).astype(np.float32)
        norm     = np.linalg.norm(mean_emb)
        if norm > 1e-10:
            mean_emb = mean_emb / norm

        self._log.info(
            "[headless] Averaged %d embeddings  best_blur=%.1f",
            len(embeddings), best_blur,
        )
        print(
            f"[FaceAuth] Averaged {len(embeddings)} samples  "
            f"best_blur={best_blur:.1f}"
        )

        # ── Recognise ─────────────────────────────────────────────────────
        recognizer = Recognizer(self._settings, self._store)
        result = recognizer.identify(mean_emb)

        self._log.info(
            "[headless] Recognition — authorized=%s  matched=%s  sim=%.3f",
            result.authorized, result.username, result.similarity,
        )
        print(
            f"[FaceAuth] Similarity={result.similarity:.3f}  "
            f"threshold={self._settings.recognition_threshold:.2f}  "
            f"authorized={result.authorized}"
        )

        if not result.authorized:
            msg = (
                f"Face not recognised "
                f"(similarity {result.similarity:.2f} "
                f"< threshold {self._settings.recognition_threshold:.2f})."
            )
            print(f"[FaceAuth] DENIED — {msg}")
            return AuthResult(
                outcome=AuthOutcome.DENIED_BELOW_THRESHOLD,
                username=username,
                message=msg,
                similarity=result.similarity,
                matched_as=result.username,
            )

        if result.username != username:
            msg = (
                f"Identity mismatch — face does not belong to '{username}'."
            )
            self._log.warning(
                "[headless] Mismatch — requested '%s', recognised '%s'",
                username, result.username,
            )
            print(f"[FaceAuth] DENIED — {msg}")
            return AuthResult(
                outcome=AuthOutcome.DENIED_MISMATCH,
                username=username,
                message=msg,
                similarity=result.similarity,
                matched_as=result.username,
            )

        # ── SUCCESS ────────────────────────────────────────────────────────
        msg = f"Authentication successful — welcome, {username}."
        self._log.info(
            "[headless] SUCCESS for '%s'  similarity=%.3f",
            username, result.similarity,
        )
        print(f"[FaceAuth] SUCCESS — {msg}")
        return AuthResult(
            outcome=AuthOutcome.SUCCESS,
            username=username,
            message=msg,
            similarity=result.similarity,
            matched_as=result.username,
        )
