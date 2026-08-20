"""
Tests for `python main.py logs` and the underlying app.logs module.

Covers:
- read_logs() unit tests (filtering, limit, missing file, empty file)
- run_logs() output formatting
- CLI integration: exit code, --limit flag, no model/camera loaded
- Security: sensitive patterns never appear in output
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logs import read_logs, run_logs, _DEFAULT_LIMIT
from app.config import Settings


# ── helpers ───────────────────────────────────────────────────────────────

def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ts(n: int = 0) -> str:
    """Deterministic fake timestamp."""
    return f"2026-08-{10 + n:02d} 12:00:{n:02d}"


def _line(module: str, message: str, level: str = "INFO", n: int = 0) -> str:
    return f"{_ts(n)} | {level:<8} | {module} | {message}"


# Convenience log lines that should be KEPT
AUTH_SUCCESS  = _line("faceauth.main",
                      "Authentication outcome — user=abba  outcome=SUCCESS  similarity=0.713")
AUTH_DENIED   = _line("app.auth.headless",
                      "[headless] Authentication started for 'abba'")
AUTH_OUTCOME  = _line("faceauth.main",
                      "Authentication outcome — user=abba  outcome=DENIED_LIVENESS  similarity=0.000")
ENROLL_DONE   = _line("app.enrollment.enroll",
                      "Enrollment completed for 'abba' (10 samples, 10 attempts)")
ENROLL_BLOCK  = _line("faceauth.main",
                      "Enrollment blocked — 'abba' already enrolled (use --re-enroll)",
                      level="WARNING")
CMD_STARTED   = _line("faceauth.main",
                      "Ubuntu FaceAuth started — command=authenticate")
HEADLESS_OK   = _line("app.auth.headless",
                      "[headless] SUCCESS for 'abba'  similarity=0.563")

# Lines that should be DROPPED (noise)
NOISE_FRAME   = _line("app.recognition.recognizer",
                      "Known face — user: abba  similarity: 0.720")
NOISE_UNKNOWN = _line("app.recognition.recognizer",
                      "Unknown face — best: abba  similarity: 0.120  threshold: 0.45")
NOISE_MODEL   = _line("app.detection.detector",
                      "Initialising InsightFace model 'buffalo_sc' (CPU)")
NOISE_CAMERA  = _line("app.camera.camera",
                      "Camera ready — 640x480 @ 30 fps  (discarded 10 warmup frame(s))")
NOISE_CACHE   = _line("app.recognition.recognizer",
                      "Loaded 1 enrolled user(s) into recognition cache")


# ── read_logs unit tests ──────────────────────────────────────────────────

class TestReadLogs:

    def test_missing_file_returns_empty(self, tmp_path):
        result = read_logs(tmp_path / "nonexistent.log")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        log = tmp_path / "faceauth.log"
        log.write_text("")
        result = read_logs(log)
        assert result == []

    def test_only_noise_returns_empty(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [NOISE_FRAME, NOISE_UNKNOWN, NOISE_MODEL, NOISE_CAMERA])
        result = read_logs(log)
        assert result == []

    def test_auth_success_kept(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [AUTH_SUCCESS])
        result = read_logs(log)
        assert len(result) == 1
        assert "SUCCESS" in result[0]
        assert "abba" in result[0]

    def test_auth_outcome_kept(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [AUTH_OUTCOME])
        result = read_logs(log)
        assert len(result) == 1
        assert "Authentication outcome" in result[0]

    def test_enrollment_completed_kept(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [ENROLL_DONE])
        result = read_logs(log)
        assert len(result) == 1
        assert "Enrollment completed" in result[0]

    def test_enrollment_blocked_kept(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [ENROLL_BLOCK])
        result = read_logs(log)
        assert len(result) == 1
        assert "Enrollment blocked" in result[0]

    def test_command_started_kept(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [CMD_STARTED])
        result = read_logs(log)
        assert len(result) == 1
        assert "command=authenticate" in result[0]

    def test_headless_success_kept(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [HEADLESS_OK])
        result = read_logs(log)
        assert len(result) == 1
        assert "SUCCESS" in result[0]

    def test_per_frame_scores_dropped(self, tmp_path):
        log = tmp_path / "faceauth.log"
        # Mix of 100 noise lines + 1 auth event
        lines = [NOISE_FRAME] * 100 + [AUTH_SUCCESS]
        _write_log(log, lines)
        result = read_logs(log, limit=50)
        assert len(result) == 1
        assert "Authentication outcome" in result[0] or "SUCCESS" in result[0]

    def test_limit_respected(self, tmp_path):
        log = tmp_path / "faceauth.log"
        lines = [
            _line("faceauth.main", f"Ubuntu FaceAuth started — command=list", n=i)
            for i in range(30)
        ]
        _write_log(log, lines)
        result = read_logs(log, limit=10)
        assert len(result) == 10

    def test_limit_returns_most_recent(self, tmp_path):
        """The last N events should be the most recent ones."""
        log = tmp_path / "faceauth.log"
        lines = [
            _line("faceauth.main",
                  f"Ubuntu FaceAuth started — command=cmd{i}", n=i)
            for i in range(25)
        ]
        _write_log(log, lines)
        result = read_logs(log, limit=5)
        assert len(result) == 5
        assert "cmd24" in result[-1]

    def test_default_limit_is_20(self, tmp_path):
        assert _DEFAULT_LIMIT == 20
        log = tmp_path / "faceauth.log"
        lines = [
            _line("faceauth.main", f"Ubuntu FaceAuth started — command=list", n=i)
            for i in range(30)
        ]
        _write_log(log, lines)
        result = read_logs(log)   # no limit arg — uses default
        assert len(result) == 20

    def test_mixed_noise_and_events(self, tmp_path):
        log = tmp_path / "faceauth.log"
        lines = [
            NOISE_MODEL,
            CMD_STARTED,
            NOISE_FRAME,
            NOISE_FRAME,
            AUTH_SUCCESS,
            NOISE_CAMERA,
            ENROLL_DONE,
            NOISE_CACHE,
        ]
        _write_log(log, lines)
        result = read_logs(log)
        assert len(result) == 3   # CMD_STARTED, AUTH_SUCCESS, ENROLL_DONE
        assert any("command=authenticate" in r for r in result)
        assert any("Authentication outcome" in r or "SUCCESS" in r for r in result)
        assert any("Enrollment completed" in r for r in result)

    def test_malformed_lines_ignored(self, tmp_path):
        log = tmp_path / "faceauth.log"
        _write_log(log, [
            "this is not a valid log line",
            "also not valid | missing fields",
            AUTH_SUCCESS,
        ])
        result = read_logs(log)
        assert len(result) == 1


# ── security: sensitive data never shown ─────────────────────────────────

class TestLogsSecurity:

    def test_raw_similarity_stream_not_shown(self, tmp_path):
        """Per-frame similarity scores must be filtered out."""
        log = tmp_path / "faceauth.log"
        # 50 per-frame scores interspersed with real events
        lines = []
        for i in range(50):
            lines.append(
                _line("app.recognition.recognizer",
                      f"Known face — user: abba  similarity: 0.{700 + i:03d}", n=i)
            )
        lines.append(AUTH_SUCCESS)
        _write_log(log, lines)
        result = read_logs(log, limit=100)
        # Only the auth outcome should remain
        assert all("Known face" not in r for r in result)

    def test_embedding_data_never_logged(self, tmp_path):
        """Embeddings are never written to the log by design — verify not present."""
        log = tmp_path / "faceauth.log"
        _write_log(log, [AUTH_SUCCESS, ENROLL_DONE])
        result = read_logs(log)
        combined = " ".join(result)
        assert "embedding" not in combined.lower()
        assert "float32" not in combined.lower()
        assert "ndarray" not in combined.lower()

    def test_password_never_in_output(self, tmp_path):
        """Passwords are never in FaceAuth logs — sanity check."""
        log = tmp_path / "faceauth.log"
        _write_log(log, [AUTH_SUCCESS, ENROLL_DONE, CMD_STARTED])
        result = read_logs(log)
        combined = " ".join(result)
        assert "password" not in combined.lower()


# ── run_logs output ───────────────────────────────────────────────────────

class TestRunLogs:

    def test_no_log_file_message(self, tmp_path, capsys):
        code = run_logs(tmp_path / "nonexistent.log")
        out = capsys.readouterr().out
        assert code == 0
        assert "no log file" in out.lower() or "not been run" in out.lower()

    def test_empty_events_message(self, tmp_path, capsys):
        log = tmp_path / "faceauth.log"
        _write_log(log, [NOISE_FRAME, NOISE_MODEL])
        code = run_logs(log)
        out = capsys.readouterr().out
        assert code == 0
        assert "no authentication events" in out.lower()

    def test_events_shown(self, tmp_path, capsys):
        log = tmp_path / "faceauth.log"
        _write_log(log, [CMD_STARTED, AUTH_SUCCESS])
        code = run_logs(log)
        out = capsys.readouterr().out
        assert code == 0
        assert "command=authenticate" in out
        assert "SUCCESS" in out

    def test_header_shown(self, tmp_path, capsys):
        log = tmp_path / "faceauth.log"
        _write_log(log, [AUTH_SUCCESS])
        run_logs(log)
        out = capsys.readouterr().out
        assert "Recent FaceAuth events" in out

    def test_limit_applied(self, tmp_path, capsys):
        log = tmp_path / "faceauth.log"
        lines = [
            _line("faceauth.main", f"Ubuntu FaceAuth started — command=list", n=i)
            for i in range(30)
        ]
        _write_log(log, lines)
        run_logs(log, limit=5)
        out = capsys.readouterr().out
        # Header says "(last 5)"
        assert "last 5" in out


# ── CLI integration ───────────────────────────────────────────────────────

class TestLogsCommand:

    def _run(self, argv: list, tmp_settings: Settings, capsys) -> tuple[int, str]:
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch.object(sys, "argv", ["ubuntu-faceauth"] + argv):
            code = main_module.main()
        out = capsys.readouterr().out
        return code, out

    def test_exit_zero_no_log(self, tmp_settings, capsys):
        # tmp_settings points to tmp_path — no log file exists
        code, _ = self._run(["logs"], tmp_settings, capsys)
        assert code == 0

    def test_exit_zero_with_log(self, tmp_settings, capsys):
        tmp_settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        _write_log(tmp_settings.log_file, [CMD_STARTED])
        code, _ = self._run(["logs"], tmp_settings, capsys)
        assert code == 0

    def test_limit_flag_accepted(self, tmp_settings, capsys):
        code, _ = self._run(["logs", "--limit", "5"], tmp_settings, capsys)
        assert code == 0

    def test_no_model_instantiated(self, tmp_settings, capsys):
        mock_cls = MagicMock()
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch("main.FaceDetector", mock_cls), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "logs"]):
            main_module.main()
        mock_cls.assert_not_called()

    def test_no_model_load_called(self, tmp_settings, capsys):
        mock_detector = MagicMock()
        mock_cls = MagicMock(return_value=mock_detector)
        import main as main_module
        with patch("main.Settings", return_value=tmp_settings), \
             patch("main.FaceDetector", mock_cls), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "logs"]):
            main_module.main()
        mock_detector.load.assert_not_called()

    def test_log_content_shown(self, tmp_settings, capsys):
        tmp_settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        _write_log(tmp_settings.log_file, [CMD_STARTED, AUTH_SUCCESS])
        _, out = self._run(["logs"], tmp_settings, capsys)
        assert "command=authenticate" in out
        assert "SUCCESS" in out
