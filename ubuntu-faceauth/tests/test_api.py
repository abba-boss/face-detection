"""
Tests for the Ubuntu FaceAuth FastAPI layer.

Uses FastAPI's TestClient (backed by httpx) — no real server started.
No AI model is loaded for GET endpoint tests.
FaceDetector is mocked for the authenticate endpoint tests.

Coverage:
  - GET /api/health
  - GET /api/status
  - GET /api/users
  - GET /api/logs
  - GET /api/doctor
  - POST /api/authenticate
  - Security: embeddings never exposed, passwords never exposed
  - GET endpoints do not instantiate FaceDetector
  - CLI `api` subcommand registered (does not start server in test)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.api.routes import create_app
from app.auth.authenticator import AuthOutcome, AuthResult
from app.config import Settings
from app.storage import FaceStore


# ── helpers ───────────────────────────────────────────────────────────────

def _unit(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_client(tmp_settings: Settings,
                 tmp_store: FaceStore) -> TestClient:
    app = create_app(settings=tmp_settings, store=tmp_store)
    return TestClient(app, raise_server_exceptions=True)


# ── GET /api/health ───────────────────────────────────────────────────────

class TestHealth:

    def test_status_200(self, tmp_settings, tmp_store):
        client = _make_client(tmp_settings, tmp_store)
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_body_status_ok(self, tmp_settings, tmp_store):
        client = _make_client(tmp_settings, tmp_store)
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_body_version_present(self, tmp_settings, tmp_store):
        client = _make_client(tmp_settings, tmp_store)
        data = client.get("/api/health").json()
        assert "version" in data
        assert data["version"] == "1.0.0"


# ── GET /api/status ───────────────────────────────────────────────────────

class TestStatus:

    def test_status_200(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/status")
        assert r.status_code == 200

    def test_contains_version(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert data["version"] == "1.0.0"

    def test_contains_model(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert data["model"] == "buffalo_sc"

    def test_contains_threshold(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert data["threshold"] == 0.45

    def test_contains_liveness_timeout(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert "liveness_timeout" in data
        assert data["liveness_timeout"] == 8.0

    def test_contains_storage(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert "storage" in data

    def test_enrolled_count_zero(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert data["enrolled"] == 0
        assert data["users"] == []

    def test_enrolled_count_reflects_store(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        tmp_store.save("bob",   _unit(2))
        data = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert data["enrolled"] == 2
        assert "alice" in data["users"]
        assert "bob" in data["users"]

    def test_no_embeddings_in_response(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        resp = _make_client(tmp_settings, tmp_store).get("/api/status").json()
        assert "embedding" not in str(resp)

    def test_no_face_detector_loaded(self, tmp_settings, tmp_store):
        mock_cls = MagicMock()
        with patch("app.api.routes.FaceDetector", mock_cls, create=True):
            _make_client(tmp_settings, tmp_store).get("/api/status")
        mock_cls.assert_not_called()


# ── GET /api/users ────────────────────────────────────────────────────────

class TestUsers:

    def test_status_200(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/users")
        assert r.status_code == 200

    def test_empty_store(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/users").json()
        assert data["count"] == 0
        assert data["users"] == []

    def test_returns_usernames(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        data = _make_client(tmp_settings, tmp_store).get("/api/users").json()
        assert data["count"] == 1
        assert data["users"][0]["username"] == "alice"

    def test_returns_enrolled_at(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        data = _make_client(tmp_settings, tmp_store).get("/api/users").json()
        user = data["users"][0]
        assert "enrolled_at" in user
        # ISO-8601 timestamp
        assert "T" in user["enrolled_at"]

    def test_multiple_users(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        tmp_store.save("bob",   _unit(2))
        tmp_store.save("carol", _unit(3))
        data = _make_client(tmp_settings, tmp_store).get("/api/users").json()
        assert data["count"] == 3
        names = [u["username"] for u in data["users"]]
        assert "alice" in names
        assert "bob"   in names
        assert "carol" in names

    def test_no_embeddings_in_response(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        resp = _make_client(tmp_settings, tmp_store).get("/api/users").json()
        assert "embedding" not in str(resp)

    def test_no_face_detector_loaded(self, tmp_settings, tmp_store):
        mock_cls = MagicMock()
        with patch("app.api.routes.FaceDetector", mock_cls, create=True):
            _make_client(tmp_settings, tmp_store).get("/api/users")
        mock_cls.assert_not_called()


# ── GET /api/logs ─────────────────────────────────────────────────────────

class TestLogs:

    def test_status_200(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/logs")
        assert r.status_code == 200

    def test_no_log_file(self, tmp_settings, tmp_store):
        # tmp_settings has no log file yet
        data = _make_client(tmp_settings, tmp_store).get("/api/logs").json()
        assert data["count"] == 0
        assert data["events"] == []

    def test_limit_param_accepted(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/logs?limit=5")
        assert r.status_code == 200
        assert r.json()["limit"] == 5

    def test_limit_default_20(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/logs").json()
        assert data["limit"] == 20

    def test_limit_too_large_rejected(self, tmp_settings, tmp_store):
        # limit > 500 should be rejected by query validation
        r = _make_client(tmp_settings, tmp_store).get("/api/logs?limit=9999")
        assert r.status_code == 422

    def test_log_events_shown(self, tmp_settings, tmp_store):
        # Write an auth event to the log file
        log = tmp_settings.log_file
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "2026-08-20 18:00:00 | INFO     | faceauth.main | "
            "Authentication outcome — user=abba  outcome=SUCCESS  similarity=0.713\n"
        )
        data = _make_client(tmp_settings, tmp_store).get("/api/logs").json()
        assert data["count"] >= 1
        assert any("SUCCESS" in e for e in data["events"])

    def test_no_face_detector_loaded(self, tmp_settings, tmp_store):
        mock_cls = MagicMock()
        with patch("app.api.routes.FaceDetector", mock_cls, create=True):
            _make_client(tmp_settings, tmp_store).get("/api/logs")
        mock_cls.assert_not_called()


# ── GET /api/doctor ───────────────────────────────────────────────────────

class TestDoctor:

    def test_status_200(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/doctor")
        assert r.status_code == 200

    def test_response_has_ok_field(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/doctor").json()
        assert "ok" in data
        assert isinstance(data["ok"], bool)

    def test_response_has_failures_field(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/doctor").json()
        assert "failures" in data
        assert isinstance(data["failures"], int)

    def test_response_has_checks_list(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/doctor").json()
        assert "checks" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) > 0

    def test_each_check_has_required_fields(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/doctor").json()
        for chk in data["checks"]:
            assert "name" in chk
            assert "ok"   in chk
            assert isinstance(chk["ok"], bool)

    def test_python_check_passes(self, tmp_settings, tmp_store):
        data = _make_client(tmp_settings, tmp_store).get("/api/doctor").json()
        python_chk = next(
            (c for c in data["checks"] if c["name"] == "Python"), None
        )
        assert python_chk is not None
        assert python_chk["ok"] is True

    def test_no_embeddings_in_response(self, tmp_settings, tmp_store):
        # "Embedding directory" check name is fine — we're checking no raw
        # float32 array data leaks, not the word "embedding" itself
        resp = _make_client(tmp_settings, tmp_store).get("/api/doctor").json()
        resp_str = str(resp)
        assert "float32"  not in resp_str
        assert "ndarray"  not in resp_str
        assert "512,"     not in resp_str   # embedding vector dimensions

    def test_no_face_detector_loaded(self, tmp_settings, tmp_store):
        mock_cls = MagicMock()
        with patch("app.api.routes.FaceDetector", mock_cls, create=True):
            _make_client(tmp_settings, tmp_store).get("/api/doctor")
        mock_cls.assert_not_called()


# ── POST /api/authenticate ────────────────────────────────────────────────

class TestAuthenticate:

    def _auth(self, client: TestClient, user: str) -> dict:
        return client.post("/api/authenticate", json={"user": user}).json()

    def test_status_200(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).post(
            "/api/authenticate", json={"user": "alice"}
        )
        assert r.status_code == 200

    def test_not_enrolled_returns_denied(self, tmp_settings, tmp_store):
        data = self._auth(_make_client(tmp_settings, tmp_store), "nobody")
        assert data["success"] is False
        assert "DENIED_NOT_ENROLLED" in data["outcome"] or \
               "not enrolled" in data["message"].lower()

    def test_not_enrolled_no_camera_opened(self, tmp_settings, tmp_store):
        """Pre-flight check must prevent camera from opening for unknown users."""
        mock_cam = MagicMock()
        with patch("app.auth.headless.Camera", mock_cam):
            self._auth(_make_client(tmp_settings, tmp_store), "nobody")
        mock_cam.assert_not_called()

    def test_empty_user_rejected(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).post(
            "/api/authenticate", json={"user": "   "}
        )
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert r.json()["success"] is False

    def test_missing_user_field_rejected(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).post(
            "/api/authenticate", json={}
        )
        assert r.status_code == 422

    def test_success_response_structure(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))

        mock_result = AuthResult(
            outcome=AuthOutcome.SUCCESS,
            username="alice",
            message="Authentication successful",
            similarity=0.75,
            matched_as="alice",
        )
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_detector = MagicMock()

        with patch("app.api.routes.FaceDetector", return_value=mock_detector), \
             patch("app.api.routes.HeadlessAuthSession", return_value=mock_session):
            data = self._auth(_make_client(tmp_settings, tmp_store), "alice")

        assert data["success"] is True
        assert data["user"]    == "alice"
        assert data["similarity"] == 0.75
        assert data["outcome"] == "SUCCESS"
        assert "message" in data

    def test_failure_response_structure(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))

        mock_result = AuthResult(
            outcome=AuthOutcome.DENIED_LIVENESS,
            username="alice",
            message="Liveness check failed",
            similarity=0.0,
        )
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_detector = MagicMock()

        with patch("app.api.routes.FaceDetector", return_value=mock_detector), \
             patch("app.api.routes.HeadlessAuthSession", return_value=mock_session):
            data = self._auth(_make_client(tmp_settings, tmp_store), "alice")

        assert data["success"]  is False
        assert data["outcome"]  == "DENIED_LIVENESS"
        assert data["similarity"] == 0.0

    def test_no_embedding_in_response(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))

        mock_result = AuthResult(
            outcome=AuthOutcome.SUCCESS,
            username="alice",
            message="ok",
            similarity=0.8,
            matched_as="alice",
        )
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_detector = MagicMock()

        with patch("app.api.routes.FaceDetector", return_value=mock_detector), \
             patch("app.api.routes.HeadlessAuthSession", return_value=mock_session):
            resp = _make_client(tmp_settings, tmp_store).post(
                "/api/authenticate", json={"user": "alice"}
            )

        resp_str = str(resp.json())
        assert "float32"  not in resp_str
        assert "ndarray"  not in resp_str
        assert "512,"     not in resp_str

    def test_no_password_in_response(self, tmp_settings, tmp_store):
        resp = _make_client(tmp_settings, tmp_store).post(
            "/api/authenticate", json={"user": "nobody"}
        )
        # No dict key named "password" in the response
        assert "password" not in _all_keys(resp.json())

    def test_detector_exception_returns_error(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))

        with patch("app.api.routes.FaceDetector",
                   side_effect=RuntimeError("model exploded")):
            data = self._auth(_make_client(tmp_settings, tmp_store), "alice")

        assert data["success"]  is False
        assert data["outcome"]  == "ERROR"


# ── Security: no sensitive data ever exposed ──────────────────────────────

class TestSecurity:

    def test_health_no_embeddings(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/health")
        assert "embedding" not in r.text

    def test_status_no_embeddings(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        r = _make_client(tmp_settings, tmp_store).get("/api/status")
        resp_str = r.text
        assert "float32" not in resp_str
        assert "ndarray" not in resp_str

    def test_users_no_embeddings(self, tmp_settings, tmp_store):
        tmp_store.save("alice", _unit(1))
        r = _make_client(tmp_settings, tmp_store).get("/api/users")
        resp_str = r.text
        assert "float32" not in resp_str
        assert "ndarray" not in resp_str

    def test_doctor_no_embeddings(self, tmp_settings, tmp_store):
        r = _make_client(tmp_settings, tmp_store).get("/api/doctor")
        resp_str = r.text
        assert "float32" not in resp_str
        assert "ndarray" not in resp_str
        assert "512,"    not in resp_str

    def test_no_passwords_in_any_response(self, tmp_settings, tmp_store):
        # Check that no credential-style password fields leak.
        # "gdm-password" in the doctor check name is a PAM filename, not a
        # credential — we check for "password:" key patterns instead.
        client = _make_client(tmp_settings, tmp_store)
        for path in ["/api/health", "/api/status", "/api/users",
                     "/api/logs",  "/api/doctor"]:
            r = client.get(path)
            data = r.json()
            # Verify no dict key named "password" anywhere in the response
            assert "password" not in _all_keys(data), \
                f"'password' key found in {path} response"


def _all_keys(obj) -> set:
    """Recursively collect all dict keys from a JSON-like object."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


# ── CLI: api subcommand registered ────────────────────────────────────────

class TestApiCliCommand:

    def test_api_subcommand_registered(self):
        """The 'api' subcommand must appear in the parser without error."""
        import main as main_module
        parser = main_module.build_parser()
        # parse_args should not raise for `api` with default args
        args = parser.parse_args(["api"])
        assert args.command == "api"
        assert args.host == "127.0.0.1"
        assert args.port == 8765

    def test_api_custom_host_port(self):
        import main as main_module
        parser = main_module.build_parser()
        args = parser.parse_args(["api", "--host", "0.0.0.0", "--port", "9000"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_api_command_calls_run_server(self, tmp_settings):
        """main.main() with `api` must call run_server without touching model."""
        import main as main_module

        mock_run_server = MagicMock()

        with patch("main.Settings", return_value=tmp_settings), \
             patch("server.run_server", mock_run_server), \
             patch.object(sys, "argv", ["ubuntu-faceauth", "api"]):
            main_module.main()

        mock_run_server.assert_called_once()

    def test_existing_commands_unaffected(self, tmp_settings, capsys):
        """Adding the api command must not break list, status, users, logs."""
        import main as main_module
        for cmd in ["list", "status", "users"]:
            with patch("main.Settings", return_value=tmp_settings), \
                 patch.object(sys, "argv", ["ubuntu-faceauth", cmd]):
                code = main_module.main()
            assert code == 0, f"command '{cmd}' returned non-zero after api addition"
