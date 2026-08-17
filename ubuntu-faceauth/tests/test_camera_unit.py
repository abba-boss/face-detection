"""
Camera unit tests — no display required, no real face needed.

These tests exercise the Camera class in isolation by mocking
OpenCV's VideoCapture, so they run headlessly in CI.

Camera-dependent tests that actually open /dev/video0 are in
tests/test_realworld.py and are skipped unless a display is available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from app.config import Settings
from app.camera import Camera


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", log_to_file=False,
                    camera_warmup_frames=3)


class TestCameraOpen:

    def test_open_raises_on_failed_device(self, settings):
        """RuntimeError when VideoCapture can't open the device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("app.camera.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = Camera(settings)
            with pytest.raises(RuntimeError, match="Cannot open camera"):
                cam.open()

    def test_open_sets_resolution(self, settings):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch("app.camera.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = Camera(settings)
            cam.open()

        calls = mock_cap.set.call_args_list
        import cv2
        props_set = {c.args[0] for c in calls}
        assert cv2.CAP_PROP_FRAME_WIDTH  in props_set
        assert cv2.CAP_PROP_FRAME_HEIGHT in props_set
        assert cv2.CAP_PROP_FPS          in props_set

    def test_warmup_frames_are_discarded(self, settings):
        """open() must call read() exactly camera_warmup_frames times."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch("app.camera.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = Camera(settings)
            cam.open()

        assert mock_cap.read.call_count == settings.camera_warmup_frames

    def test_context_manager_releases(self, settings):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch("app.camera.camera.cv2.VideoCapture", return_value=mock_cap):
            with patch("app.camera.camera.cv2.destroyAllWindows"):
                with Camera(settings) as cam:
                    assert cam.is_open

        mock_cap.release.assert_called_once()


class TestCameraRead:

    def _open_camera(self, settings) -> tuple[Camera, MagicMock]:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        with patch("app.camera.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = Camera(settings)
            cam.open()
        # Reset read call count after warmup
        mock_cap.read.reset_mock()
        mock_cap.read.return_value = (True, frame)
        cam._cap = mock_cap
        return cam, mock_cap

    def test_read_returns_frame(self, settings):
        cam, mock_cap = self._open_camera(settings)
        ok, frame = cam.read()
        assert ok
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_read_returns_false_on_failure(self, settings):
        cam, mock_cap = self._open_camera(settings)
        mock_cap.read.return_value = (False, None)
        ok, frame = cam.read()
        assert not ok
        assert frame is None

    def test_read_without_open_returns_false(self, settings):
        cam = Camera(settings)
        ok, frame = cam.read()
        assert not ok
        assert frame is None

    def test_is_open_false_before_open(self, settings):
        cam = Camera(settings)
        assert not cam.is_open


class TestCameraRelease:

    def test_release_calls_cap_release(self, settings):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch("app.camera.camera.cv2.VideoCapture", return_value=mock_cap):
            with patch("app.camera.camera.cv2.destroyAllWindows") as mock_destroy:
                cam = Camera(settings)
                cam.open()
                cam.release()

        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    def test_release_safe_if_never_opened(self, settings):
        """release() must not raise if open() was never called."""
        with patch("app.camera.camera.cv2.destroyAllWindows"):
            cam = Camera(settings)
            cam.release()   # should not raise
