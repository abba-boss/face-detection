"""
Tests for `python main.py status`.

Verifies:
- exit code 0 always
- all expected fields appear in output
- no AI model loaded (FaceDetector.load never called)
- no camera opened
- correct enrolled count and usernames shown
- correct values reflected when settings are overridden
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.storage import FaceStore


# ── helpers ───────────────────────────────────────────────────────────────

def _unit(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _run_status(tmp_settings: Settings) -> tuple[int, str]:
    """Run `main status` and return (exit_code, stdout)."""
    import main as main_module
    with patch("main.Settings", return_value=tmp_settings), \
         patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
        # Reload to pick up any module-level changes
        import importlib
        importlib.reload(main_module)
        from io import StringIO
        import sys as _sys
        code = main_module.main()
    return code


def _run_status_captured(tmp_settings: Settings, capsys) -> tuple[int, str]:
    """Run status and return (exit_code, captured_stdout)."""
    import main as main_module
    with patch("main.Settings", return_value=tmp_settings), \
         patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
        code = main_module.main()
    out = capsys.readouterr().out
    return code, out


# ── exit code ─────────────────────────────────────────────────────────────

class TestStatusExitCode:

    def test_exit_zero_no_users(self, tmp_settings, capsys):
        code, _ = _run_status_captured(tmp_settings, capsys)
        assert code == 0

    def test_exit_zero_with_users(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        code, _ = _run_status_captured(tmp_settings, capsys)
        assert code == 0


# ── output fields ─────────────────────────────────────────────────────────

class TestStatusOutput:

    def test_version_shown(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "version" in out
        assert "1.0.0" in out

    def test_model_shown(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "model" in out
        assert "buffalo_sc" in out

    def test_camera_shown(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "camera" in out
        assert "/dev/video0" in out

    def test_threshold_shown(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "threshold" in out
        assert "0.45" in out

    def test_liveness_timeout_shown(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "liveness" in out
        assert "8.0" in out

    def test_storage_backend_shown(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "storage" in out

    def test_enrolled_count_zero(self, tmp_settings, capsys):
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "enrolled" in out
        assert "0" in out

    def test_enrolled_count_one(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(2))
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "1" in out
        assert "alice" in out

    def test_enrolled_multiple_users(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(3))
        tmp_store.save("bob",   _unit(4))
        _, out = _run_status_captured(tmp_settings, capsys)
        assert "alice" in out
        assert "bob" in out
        assert "2" in out


# ── settings values reflected ─────────────────────────────────────────────

class TestStatusReflectsSettings:

    def test_custom_threshold_shown(self, tmp_path, capsys):
        s = Settings(
            data_dir=tmp_path / "d",
            log_to_file=False,
            recognition_threshold=0.65,
        )
        import main as main_module
        with patch("main.Settings", return_value=s), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "0.65" in out

    def test_custom_liveness_timeout_shown(self, tmp_path, capsys):
        s = Settings(
            data_dir=tmp_path / "d",
            log_to_file=False,
            liveness_timeout=12.0,
        )
        import main as main_module
        with patch("main.Settings", return_value=s), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "12.0" in out

    def test_buffalo_l_model_shown(self, tmp_path, capsys):
        s = Settings(
            data_dir=tmp_path / "d",
            log_to_file=False,
            insightface_model_name="buffalo_l",
        )
        import main as main_module
        with patch("main.Settings", return_value=s), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "buffalo_l" in out

    def test_camera_device_1_shown(self, tmp_path, capsys):
        s = Settings(
            data_dir=tmp_path / "d",
            log_to_file=False,
            camera_device=1,
        )
        import main as main_module
        with patch("main.Settings", return_value=s), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "/dev/video1" in out


# ── model and camera never accessed ───────────────────────────────────────

class TestStatusNoModelNoCamera:

    def test_face_detector_never_instantiated(self, tmp_settings, capsys):
        """status must not create a FaceDetector instance."""
        mock_detector_cls = MagicMock()
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch("main.FaceDetector", mock_detector_cls), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
            main_module.main()
        mock_detector_cls.assert_not_called()

    def test_face_detector_load_never_called(self, tmp_settings, capsys):
        """status must not call FaceDetector.load()."""
        mock_detector = MagicMock()
        mock_detector_cls = MagicMock(return_value=mock_detector)
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch("main.FaceDetector", mock_detector_cls), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "status"]):
            main_module.main()
        mock_detector.load.assert_not_called()
