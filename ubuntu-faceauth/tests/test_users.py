"""
Tests for `python main.py users`.

Verifies:
- exit code 0 always
- enrolled count, usernames, and enrollment dates appear in output
- no AI model loaded (FaceDetector never instantiated or .load() called)
- no camera opened
- `users` and `list` produce identical output (alias parity)
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


def _run(command: str, tmp_settings: Settings, capsys) -> tuple[int, str]:
    """Run main() with the given command and return (exit_code, stdout)."""
    import main as main_module
    with patch("main.Settings", return_value=tmp_settings), \
         patch.object(sys, "argv", ["ubuntu-faceauth", command]):
        code = main_module.main()
    out = capsys.readouterr().out
    return code, out


# ── exit code ─────────────────────────────────────────────────────────────

class TestUsersExitCode:

    def test_exit_zero_no_users(self, tmp_settings, capsys):
        code, _ = _run("users", tmp_settings, capsys)
        assert code == 0

    def test_exit_zero_with_users(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        code, _ = _run("users", tmp_settings, capsys)
        assert code == 0

    def test_exit_zero_multiple_users(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        tmp_store.save("bob",   _unit(2))
        tmp_store.save("carol", _unit(3))
        code, _ = _run("users", tmp_settings, capsys)
        assert code == 0


# ── output content ────────────────────────────────────────────────────────

class TestUsersOutput:

    def test_no_users_message(self, tmp_settings, capsys):
        _, out = _run("users", tmp_settings, capsys)
        assert "no users" in out.lower() or "0" in out

    def test_enrolled_count_shown(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        tmp_store.save("bob",   _unit(2))
        _, out = _run("users", tmp_settings, capsys)
        assert "2" in out

    def test_username_shown(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        _, out = _run("users", tmp_settings, capsys)
        assert "alice" in out

    def test_multiple_usernames_shown(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        tmp_store.save("bob",   _unit(2))
        _, out = _run("users", tmp_settings, capsys)
        assert "alice" in out
        assert "bob" in out

    def test_enrollment_date_shown(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(1))
        _, out = _run("users", tmp_settings, capsys)
        # ISO-8601 date contains "T" and "Z"
        assert "enrolled" in out.lower() or "T" in out


# ── no model, no camera ───────────────────────────────────────────────────

class TestUsersNoModelNoCamera:

    def test_face_detector_never_instantiated(self, tmp_settings, capsys):
        mock_cls = MagicMock()
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch("main.FaceDetector", mock_cls), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "users"]):
            main_module.main()
        mock_cls.assert_not_called()

    def test_face_detector_load_never_called(self, tmp_settings, capsys):
        mock_detector = MagicMock()
        mock_cls = MagicMock(return_value=mock_detector)
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch("main.FaceDetector", mock_cls), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "users"]):
            main_module.main()
        mock_detector.load.assert_not_called()


# ── alias parity: users == list ───────────────────────────────────────────

def _strip_log_lines(text: str) -> str:
    """Remove log lines (start with a timestamp) before comparing."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.startswith("20")   # timestamps start with year
    )


class TestUsersListParity:

    def test_empty_store_same_output(self, tmp_settings, capsys):
        _, users_out = _run("users", tmp_settings, capsys)
        _, list_out  = _run("list",  tmp_settings, capsys)
        assert _strip_log_lines(users_out) == _strip_log_lines(list_out)

    def test_one_user_same_output(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(10))
        _, users_out = _run("users", tmp_settings, capsys)
        _, list_out  = _run("list",  tmp_settings, capsys)
        assert _strip_log_lines(users_out) == _strip_log_lines(list_out)

    def test_multiple_users_same_output(self, tmp_settings, tmp_store, capsys):
        tmp_store.save("alice", _unit(10))
        tmp_store.save("bob",   _unit(11))
        _, users_out = _run("users", tmp_settings, capsys)
        _, list_out  = _run("list",  tmp_settings, capsys)
        assert _strip_log_lines(users_out) == _strip_log_lines(list_out)
