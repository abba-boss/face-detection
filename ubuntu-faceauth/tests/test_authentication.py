"""
Tests for V3 — AuthSession (app/auth/authenticator.py).

All tests are camera-free and model-free.  Strategy:
  - Patch LivenessSession.run() to control liveness outcome.
  - Patch Camera so read() returns synthetic frames.
  - Patch FaceDetector.detect() to return synthetic DetectedFace objects.
  - Patch cv2 GUI calls so no window opens.
  - Inject real FaceStore + Settings pointing at tmp_path.

Scenarios covered
-----------------
1.  User not enrolled            → DENIED_NOT_ENROLLED (no camera opened)
2.  Liveness FAILED              → DENIED_LIVENESS
3.  Liveness TIMEOUT             → DENIED_LIVENESS
4.  Liveness LIVE, no face found → DENIED_NO_FACE
5.  Liveness LIVE, face too blurry → DENIED_NO_FACE
6.  Liveness LIVE, face similarity below threshold → DENIED_BELOW_THRESHOLD
7.  Liveness LIVE, face matches wrong user → DENIED_MISMATCH
8.  Happy path — all steps pass  → SUCCESS
9.  success property True only for SUCCESS
10. Camera open error            → ERROR
11. Q key during capture         → DENIED_NO_FACE (cancelled)
12. CLI: authenticate --user abba exits 0 on success
13. CLI: authenticate --user abba exits 1 on denial
14. AuthResult.success property
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.detection.detector import DetectedFace
from app.liveness.detector import LivenessState
from app.storage import FaceStore
from app.auth import AuthSession, AuthResult, AuthOutcome


# ── Helpers ───────────────────────────────────────────────────────────────

def _unit(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_face(
    embedding: Optional[np.ndarray] = None,
    blur_score: float = 50.0,
) -> DetectedFace:
    if embedding is None:
        embedding = _unit(0)
    bbox = np.array([100, 100, 220, 220], dtype=np.float32)
    face = DetectedFace(bbox=bbox, confidence=0.99, embedding=embedding)
    face.blur_score = blur_score
    return face


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", log_to_file=False)


@pytest.fixture
def store(settings) -> FaceStore:
    fs = FaceStore(settings)
    fs.save("abba", _unit(0))   # enroll user with seed-0 embedding
    return fs


@pytest.fixture
def detector():
    from app.detection import FaceDetector
    return MagicMock(spec=FaceDetector)


def _make_session(settings, detector, store, max_frames=10):
    return AuthSession(settings, detector, store, max_capture_frames=max_frames)


# ── cv2 suppress context ──────────────────────────────────────────────────

def _cv2_suppressed():
    """Return a context manager that silences all cv2 GUI calls."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch("app.auth.authenticator.cv2.imshow"), \
             patch("app.auth.authenticator.cv2.waitKey", return_value=0), \
             patch("app.auth.authenticator.cv2.destroyAllWindows"):
            yield

    return _ctx()


# ═══════════════════════════════════════════════════════════════════════════
# 1. User not enrolled
# ═══════════════════════════════════════════════════════════════════════════

class TestNotEnrolled:

    def test_denied_immediately_if_not_enrolled(self, settings, store, detector):
        session = _make_session(settings, detector, store)

        with patch("app.auth.authenticator.LivenessSession") as mock_live:
            result = session.run("ghost")

        assert result.outcome == AuthOutcome.DENIED_NOT_ENROLLED
        assert result.username == "ghost"
        assert not result.success
        mock_live.assert_not_called()   # camera never opened


# ═══════════════════════════════════════════════════════════════════════════
# 2 & 3. Liveness failures
# ═══════════════════════════════════════════════════════════════════════════

class TestLivenessDenied:

    @pytest.mark.parametrize("live_state", [
        LivenessState.FAILED,
        LivenessState.TIMEOUT,
    ])
    def test_denied_on_liveness_failure(
        self, settings, store, detector, live_state
    ):
        session = _make_session(settings, detector, store)

        mock_live_instance = MagicMock()
        mock_live_instance.run.return_value = live_state

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live_instance):
            result = session.run("abba")

        assert result.outcome == AuthOutcome.DENIED_LIVENESS
        assert not result.success

    def test_liveness_message_contains_reason(self, settings, store, detector):
        session = _make_session(settings, detector, store)

        mock_live_instance = MagicMock()
        mock_live_instance.run.return_value = LivenessState.TIMEOUT

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live_instance):
            result = session.run("abba")

        # message should mention timed out / timeout
        assert "time" in result.message.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 4. No face captured after liveness passes
# ═══════════════════════════════════════════════════════════════════════════

class TestNoFaceCaptured:

    def test_denied_when_no_face_detected(self, settings, store, detector):
        session = _make_session(settings, detector, store, max_frames=5)
        detector.detect.return_value = []   # never finds a face

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             _cv2_suppressed():
            result = session.run("abba")

        assert result.outcome == AuthOutcome.DENIED_NO_FACE
        assert not result.success


# ═══════════════════════════════════════════════════════════════════════════
# 5. Face too blurry
# ═══════════════════════════════════════════════════════════════════════════

class TestBlurryFace:

    def test_denied_when_face_too_blurry(self, settings, store, detector):
        session = _make_session(settings, detector, store, max_frames=5)
        # blur_score=1.0 → well below enrollment_blur_threshold (20.0)
        blurry_face = _make_face(_unit(0), blur_score=1.0)
        detector.detect.return_value = [blurry_face]

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             _cv2_suppressed():
            result = session.run("abba")

        assert result.outcome == AuthOutcome.DENIED_NO_FACE


# ═══════════════════════════════════════════════════════════════════════════
# 6. Similarity below threshold
# ═══════════════════════════════════════════════════════════════════════════

class TestBelowThreshold:

    def test_denied_when_similarity_too_low(self, settings, store, detector):
        session = _make_session(settings, detector, store)
        # seed 99 → very different embedding from enrolled abba (seed 0)
        unknown_face = _make_face(_unit(99), blur_score=50.0)
        detector.detect.return_value = [unknown_face]

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             _cv2_suppressed():
            result = session.run("abba")

        assert result.outcome == AuthOutcome.DENIED_BELOW_THRESHOLD
        assert result.similarity < settings.recognition_threshold
        assert not result.success


# ═══════════════════════════════════════════════════════════════════════════
# 7. Identity mismatch
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentityMismatch:

    def test_denied_when_wrong_user_authenticated(self, settings, store, detector):
        """
        bob is enrolled; his face appears; we asked for abba → DENIED_MISMATCH.
        """
        # Enroll bob with a clearly different embedding
        store.save("bob", _unit(1))

        session = _make_session(settings, detector, store)
        # Present bob's face (seed 1 embedding, high blur score)
        bob_face = _make_face(_unit(1), blur_score=50.0)
        detector.detect.return_value = [bob_face]

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             _cv2_suppressed():
            result = session.run("abba")   # <-- asking for abba, not bob

        assert result.outcome == AuthOutcome.DENIED_MISMATCH
        assert result.matched_as == "bob"
        assert not result.success


# ═══════════════════════════════════════════════════════════════════════════
# 8. Happy path — full SUCCESS
# ═══════════════════════════════════════════════════════════════════════════

class TestSuccess:

    def test_authentication_success(self, settings, store, detector):
        """Liveness LIVE + abba's own face → SUCCESS."""
        session = _make_session(settings, detector, store)
        abba_face = _make_face(_unit(0), blur_score=50.0)
        detector.detect.return_value = [abba_face]

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             _cv2_suppressed():
            result = session.run("abba")

        assert result.outcome == AuthOutcome.SUCCESS
        assert result.success
        assert result.username == "abba"
        assert result.matched_as == "abba"
        assert result.similarity >= settings.recognition_threshold

    def test_success_message_contains_username(self, settings, store, detector):
        session = _make_session(settings, detector, store)
        abba_face = _make_face(_unit(0), blur_score=50.0)
        detector.detect.return_value = [abba_face]

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             _cv2_suppressed():
            result = session.run("abba")

        assert "abba" in result.message


# ═══════════════════════════════════════════════════════════════════════════
# 9. AuthResult.success property
# ═══════════════════════════════════════════════════════════════════════════

class TestAuthResultSuccess:

    @pytest.mark.parametrize("outcome, expected", [
        (AuthOutcome.SUCCESS,               True),
        (AuthOutcome.DENIED_NOT_ENROLLED,   False),
        (AuthOutcome.DENIED_LIVENESS,       False),
        (AuthOutcome.DENIED_NO_FACE,        False),
        (AuthOutcome.DENIED_MISMATCH,       False),
        (AuthOutcome.DENIED_BELOW_THRESHOLD, False),
        (AuthOutcome.ERROR,                 False),
    ])
    def test_success_property(self, outcome, expected):
        r = AuthResult(outcome=outcome, username="u", message="m")
        assert r.success is expected


# ═══════════════════════════════════════════════════════════════════════════
# 10. Camera open error
# ═══════════════════════════════════════════════════════════════════════════

class TestCameraError:

    def test_error_when_camera_fails(self, settings, store, detector):
        session = _make_session(settings, detector, store)

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.open.side_effect = RuntimeError("no camera")

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera):
            result = session.run("abba")

        assert result.outcome == AuthOutcome.ERROR
        assert "camera" in result.message.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 11. Q key during capture
# ═══════════════════════════════════════════════════════════════════════════

class TestCancelDuringCapture:

    def test_q_key_during_capture_returns_denied(self, settings, store, detector):
        session = _make_session(settings, detector, store)
        detector.detect.return_value = []

        mock_live = MagicMock()
        mock_live.run.return_value = LivenessState.LIVE
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, _blank_frame())

        with patch("app.auth.authenticator.LivenessSession",
                   return_value=mock_live), \
             patch("app.auth.authenticator.Camera", return_value=mock_camera), \
             patch("app.auth.authenticator.cv2.imshow"), \
             patch("app.auth.authenticator.cv2.waitKey",
                   return_value=ord("q")), \
             patch("app.auth.authenticator.cv2.destroyAllWindows"):
            result = session.run("abba")

        assert result.outcome == AuthOutcome.DENIED_NO_FACE


# ═══════════════════════════════════════════════════════════════════════════
# 12 & 13. CLI wiring
# ═══════════════════════════════════════════════════════════════════════════

class TestAuthCLI:

    def test_cli_exits_0_on_success(self, settings, store):
        import main as main_module

        mock_detector = MagicMock()
        mock_session  = MagicMock()
        mock_session.run.return_value = AuthResult(
            outcome=AuthOutcome.SUCCESS,
            username="abba",
            message="ok",
            similarity=0.9,
            matched_as="abba",
        )

        with patch("main.Settings", return_value=settings), \
             patch("main.FaceDetector", return_value=mock_detector), \
             patch("main.AuthSession", return_value=mock_session), \
             patch.object(sys, "argv",
                          ["ubuntu-faceauth", "authenticate", "--user", "abba"]):
            code = main_module.main()

        assert code == 0
        mock_session.run.assert_called_once_with("abba")

    def test_cli_exits_1_on_denial(self, settings, store):
        import main as main_module

        mock_detector = MagicMock()
        mock_session  = MagicMock()
        mock_session.run.return_value = AuthResult(
            outcome=AuthOutcome.DENIED_LIVENESS,
            username="abba",
            message="failed",
        )

        with patch("main.Settings", return_value=settings), \
             patch("main.FaceDetector", return_value=mock_detector), \
             patch("main.AuthSession", return_value=mock_session), \
             patch.object(sys, "argv",
                          ["ubuntu-faceauth", "authenticate", "--user", "abba"]):
            code = main_module.main()

        assert code == 1

    def test_cli_exits_1_when_not_enrolled(self, settings):
        """--user ghost → not enrolled → exit 1 without opening camera."""
        import main as main_module

        # Use a store with no enrollments
        empty_store = FaceStore(settings)

        mock_detector = MagicMock()

        with patch("main.Settings", return_value=settings), \
             patch("main.FaceDetector", return_value=mock_detector), \
             patch("main.FaceStore", return_value=empty_store), \
             patch.object(sys, "argv",
                          ["ubuntu-faceauth", "authenticate", "--user", "ghost"]):
            code = main_module.main()

        assert code == 1
