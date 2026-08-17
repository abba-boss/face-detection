"""
Real-world integration tests — REQUIRE a physical webcam.

These tests are skipped automatically when:
  - /dev/video0 is not present, or
  - the FACEAUTH_REALWORLD environment variable is not set to "1"

To run them explicitly:
    FACEAUTH_REALWORLD=1 pytest tests/test_realworld.py -v

They do NOT require a real human face — they verify that the full
detect() + embedding pipeline produces valid output from a live
camera frame, regardless of what the camera sees.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── Skip guard ─────────────────────────────────────────────────────────────

_CAMERA_PRESENT = Path("/dev/video0").exists()
_ENABLED = os.environ.get("FACEAUTH_REALWORLD", "0") == "1"
_SKIP_REASON = (
    "Real-world tests skipped. "
    "Set FACEAUTH_REALWORLD=1 and ensure /dev/video0 is accessible to run."
)

pytestmark = pytest.mark.skipif(
    not (_CAMERA_PRESENT and _ENABLED),
    reason=_SKIP_REASON,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def live_settings(tmp_path_factory):
    from app.config import Settings
    d = tmp_path_factory.mktemp("rw_data")
    return Settings(data_dir=d, log_to_file=False, camera_warmup_frames=5)


@pytest.fixture(scope="module")
def live_frame(live_settings):
    """Capture one real frame from /dev/video0."""
    cap = cv2.VideoCapture(live_settings.camera_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  live_settings.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, live_settings.camera_height)
    for _ in range(live_settings.camera_warmup_frames):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    assert ok, "Could not read a frame from the webcam"
    return frame


@pytest.fixture(scope="module")
def loaded_detector(live_settings):
    from app.detection import FaceDetector
    d = FaceDetector(live_settings)
    d.load()
    return d


# ── Tests ──────────────────────────────────────────────────────────────────

class TestLiveCamera:

    def test_camera_opens_and_reads(self, live_settings):
        from app.camera import Camera
        with Camera(live_settings) as cam:
            ok, frame = cam.read()
        assert ok
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3   # BGR

    def test_frame_has_correct_shape(self, live_settings, live_frame):
        h, w = live_frame.shape[:2]
        # Not enforcing exact size — camera may cap at different resolution
        assert w > 0 and h > 0

    def test_frame_is_not_all_black(self, live_frame):
        """A connected camera should return non-zero pixel data."""
        assert live_frame.mean() > 0


class TestLiveDetection:

    def test_detect_does_not_crash(self, loaded_detector, live_frame):
        """detect() must return a list (empty or not) without exception."""
        faces = loaded_detector.detect(live_frame)
        assert isinstance(faces, list)

    def test_detected_faces_have_valid_bbox(self, loaded_detector, live_frame):
        faces = loaded_detector.detect(live_frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            assert x2 > x1, "bbox x2 must be greater than x1"
            assert y2 > y1, "bbox y2 must be greater than y1"

    def test_detected_faces_have_embeddings(self, loaded_detector, live_frame):
        faces = loaded_detector.detect(live_frame)
        for face in faces:
            assert face.embedding is not None
            assert face.embedding.shape == (512,)
            assert face.embedding.dtype == np.float32

    def test_detected_faces_have_blur_score(self, loaded_detector, live_frame):
        faces = loaded_detector.detect(live_frame)
        for face in faces:
            assert face.blur_score >= 0.0

    def test_detected_faces_have_kps(self, loaded_detector, live_frame):
        faces = loaded_detector.detect(live_frame)
        for face in faces:
            if face.kps is not None:
                assert face.kps.shape == (5, 2)

    def test_embedding_is_unit_length(self, loaded_detector, live_frame):
        """
        Detector must L2-normalise embeddings before returning them.
        All downstream code (recogniser, store) expects unit vectors.
        """
        faces = loaded_detector.detect(live_frame)
        for face in faces:
            if face.embedding is not None:
                norm = float(np.linalg.norm(face.embedding))
                assert abs(norm - 1.0) < 1e-4, (
                    f"Expected unit-length embedding, got norm={norm:.6f}"
                )

    def test_no_future_warning_emitted(self, loaded_detector, live_frame):
        """
        The scikit-image FutureWarning must not escape detect() at runtime.

        Note: We cannot use warnings.simplefilter('always') here because
        that would override the suppressor *inside* detect() — that's a
        test-framework artefact, not a runtime condition.  Instead we
        verify that detect() runs without raising when FutureWarning is
        promoted to an error (the strictest possible check for runtime leakage).
        """
        import warnings

        # If the warning leaks, this will raise FutureWarning as an exception.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r".*estimate.*deprecated.*",
                category=FutureWarning,
            )
            try:
                loaded_detector.detect(live_frame)
            except FutureWarning as exc:
                pytest.fail(
                    f"FutureWarning escaped from detect(): {exc}"
                )


class TestLiveRecognition:
    """End-to-end: enroll from live frame embedding → recognise → cleanup."""

    def test_enroll_and_recognise(self, live_settings, live_frame, loaded_detector):
        from app.storage import FaceStore
        from app.recognition import Recognizer

        store = FaceStore(live_settings)
        rec = Recognizer(live_settings, store)

        faces = loaded_detector.detect(live_frame)
        if not faces:
            pytest.skip("No face detected in live frame — cannot test recognition")

        emb = faces[0].embedding
        assert emb is not None

        store.save("realworld_user", emb)
        rec.reload()
        result = rec.identify(emb)

        # Identical embedding vs enrolled embedding — must be authorised
        assert result.authorized, (
            f"Expected AUTHORIZED but got similarity={result.similarity:.3f}"
        )
        assert result.username == "realworld_user"

        # Cleanup
        store.delete("realworld_user")
