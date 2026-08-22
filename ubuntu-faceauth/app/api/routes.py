"""
FastAPI route definitions for Ubuntu FaceAuth.

All routes delegate to existing modules:
  /api/health        → static response
  /api/status        → Settings + FaceStore (same data as `main.py status`)
  /api/users         → FaceStore.list_users() + enrollment_info()
  /api/logs          → app.logs.read_logs()
  /api/doctor        → app.doctor (structured JSON version)
  /api/authenticate  → HeadlessAuthSession (same flow as CLI --headless)

No embeddings, passwords, or raw biometric data are ever returned.
FaceDetector is NOT loaded for GET endpoints.
"""

from __future__ import annotations

import importlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import Settings
from app.detection import FaceDetector
from app.auth import HeadlessAuthSession
from app.logs import read_logs
from app.storage import FaceStore

# Version string — kept in sync with main.py
_VERSION = "1.0.0"


# ── Request / Response models ─────────────────────────────────────────────

class AuthRequest(BaseModel):
    user: str


class AuthResponse(BaseModel):
    success: bool
    user: str
    similarity: float
    message: str
    outcome: str


class UserInfo(BaseModel):
    username: str
    enrolled_at: str


class DoctorCheck(BaseModel):
    name: str
    ok: bool
    note: str = ""


# ── App factory ───────────────────────────────────────────────────────────

def create_app(settings: Optional[Settings] = None,
               store: Optional[FaceStore] = None) -> FastAPI:
    """
    Build and return the FastAPI application.

    *settings* and *store* are injected for testing.
    When None, fresh instances are created per request.
    """
    app = FastAPI(
        title="Ubuntu FaceAuth API",
        version=_VERSION,
        description="Local biometric face authentication REST API",
    )

    # CORS — allow the Vite dev server and localhost origins.
    # Restricted to localhost only; not exposed to external origins.
    # Covers all ports Vite may auto-select (5173–5179) on both hostname variants.
    _vite_origins = [
        f"http://localhost:{p}"
        for p in range(5173, 5180)
    ] + [
        f"http://127.0.0.1:{p}"
        for p in range(5173, 5180)
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_vite_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def _get_settings() -> Settings:
        return settings if settings is not None else Settings()

    def _get_store(s: Settings) -> FaceStore:
        return store if store is not None else FaceStore(s)

    # ── GET /api/health ───────────────────────────────────────────────────

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": _VERSION}

    # ── GET /api/status ───────────────────────────────────────────────────

    @app.get("/api/status")
    def status() -> Dict[str, Any]:
        s = _get_settings()
        st = _get_store(s)
        users = st.list_users()
        # Report what's actually on disk, not just the config default
        db_path = s.data_dir / "faceauth.db"
        npz_files = list(s.embeddings_dir.glob("*.npz")) \
            if s.embeddings_dir.exists() else []
        actual_backend = "sqlite" if db_path.exists() else \
                         "npz"    if npz_files else \
                         s.storage_backend
        return {
            "version":          _VERSION,
            "model":            s.insightface_model_name,
            "camera":           f"/dev/video{s.camera_device}",
            "threshold":        s.recognition_threshold,
            "liveness_timeout": s.liveness_timeout,
            "storage":          actual_backend,
            "enrolled":         len(users),
            "users":            users,
        }

    # ── GET /api/users ────────────────────────────────────────────────────

    @app.get("/api/users")
    def users() -> Dict[str, Any]:
        s = _get_settings()
        st = _get_store(s)
        user_list = st.list_users()
        result: List[Dict[str, str]] = []
        for username in user_list:
            info = st.enrollment_info(username)
            result.append({
                "username":    username,
                "enrolled_at": info["enrolled_at"] if info else "unknown",
            })
        return {"count": len(result), "users": result}

    # ── GET /api/logs ─────────────────────────────────────────────────────

    @app.get("/api/logs")
    def logs(limit: int = Query(default=20, ge=1, le=500)) -> Dict[str, Any]:
        s = _get_settings()
        lines = read_logs(s.log_file, limit=limit)
        return {
            "limit":  limit,
            "count":  len(lines),
            "events": lines,
        }

    # ── GET /api/doctor ───────────────────────────────────────────────────

    @app.get("/api/doctor")
    def doctor() -> Dict[str, Any]:
        s = _get_settings()
        st = _get_store(s)
        checks: List[Dict[str, Any]] = []
        failures = 0

        def chk(name: str, ok: bool, note: str = "") -> None:
            nonlocal failures
            checks.append({"name": name, "ok": ok, "note": note})
            if not ok:
                failures += 1

        # Python interpreter
        chk("Python", bool(sys.executable), sys.executable)

        # Packages
        for mod, label in [("cv2", "OpenCV"),
                            ("insightface", "InsightFace"),
                            ("onnxruntime", "ONNX Runtime")]:
            try:
                importlib.import_module(mod)
                chk(label, True)
            except ImportError:
                chk(label, False, f"{mod} not importable")

        # Camera
        cam_path = Path(f"/dev/video{s.camera_device}")
        chk(f"Camera /dev/video{s.camera_device}",
            cam_path.exists() and os.access(cam_path, os.R_OK))

        # Face model
        model_dir = s.insightface_root / "models" / s.insightface_model_name
        model_ok = model_dir.exists() and any(model_dir.iterdir())
        chk(f"Face model ({s.insightface_model_name})", model_ok, str(model_dir))

        # Model cache
        chk("Model cache", s.insightface_root.exists(), str(s.insightface_root))

        # Enrollment
        try:
            enrolled = st.list_users()
            chk("Enrollment", len(enrolled) > 0,
                f"{len(enrolled)} user(s): {', '.join(enrolled)}")
        except Exception as exc:
            chk("Enrollment", False, str(exc))

        # Storage
        db_path = s.data_dir / "faceauth.db"
        npz_files = list(s.embeddings_dir.glob("*.npz")) \
            if s.embeddings_dir.exists() else []
        if db_path.exists():
            perms_ok = _perms(db_path, 0o600)
            chk("SQLite database", True, str(db_path))
            chk("SQLite permissions", perms_ok)
        elif npz_files:
            chk("NPZ storage", True, f"{len(npz_files)} file(s)")
        else:
            chk("Storage", False, "No database or NPZ files found")

        # Embedding directory
        chk("Embedding directory", s.embeddings_dir.exists(),
            str(s.embeddings_dir))

        # PAM wrapper
        pam_path = "/opt/faceauth/pam_faceauth.sh"
        pam_dir  = Path("/opt/faceauth")
        pam_ok = False
        try:
            rc = subprocess.call(["sudo", "-n", "test", "-x", pam_path],
                                 timeout=2, stderr=subprocess.DEVNULL)
            pam_ok = rc == 0
        except Exception:
            pass
        if not pam_ok and pam_dir.exists():
            pam_ok = True
        chk("PAM wrapper", pam_ok, pam_path)

        # GDM PAM config
        gdm_pam = Path("/etc/pam.d/gdm-password")
        pam_configured = False
        try:
            if gdm_pam.exists():
                content = gdm_pam.read_text()
                pam_configured = ("pam_exec.so" in content and
                                  "/opt/faceauth/pam_faceauth.sh" in content)
        except (OSError, PermissionError):
            pass
        chk("GDM PAM configuration", pam_configured)

        # Permissions
        chk("Data permissions",      _perms(s.data_dir, 0o700),
            str(s.data_dir))
        chk("Embedding permissions", _perms(s.embeddings_dir, 0o700),
            str(s.embeddings_dir))

        return {
            "ok":       failures == 0,
            "failures": failures,
            "checks":   checks,
        }

    # ── POST /api/authenticate ────────────────────────────────────────────

    @app.post("/api/authenticate")
    def authenticate(body: AuthRequest) -> AuthResponse:
        """
        Run headless liveness + face recognition for *body.user*.

        This opens the camera and performs the full authentication flow.
        Returns success/failure with similarity score and a human message.
        No embedding data is returned.
        """
        username = body.user.strip()
        if not username:
            raise HTTPException(status_code=422, detail="user field is required")

        s  = _get_settings()
        st = _get_store(s)

        # Pre-flight: must be enrolled before opening camera
        if not st.is_enrolled(username):
            return AuthResponse(
                success=False,
                user=username,
                similarity=0.0,
                message=f"User '{username}' is not enrolled.",
                outcome="DENIED_NOT_ENROLLED",
            )

        try:
            detector = FaceDetector(s)
            detector.load()

            session = HeadlessAuthSession(s, detector, st)
            result  = session.run(username)

            return AuthResponse(
                success=result.success,
                user=result.username or username,
                similarity=round(result.similarity, 4),
                message=result.message,
                outcome=result.outcome.name,
            )

        except Exception as exc:
            return AuthResponse(
                success=False,
                user=username,
                similarity=0.0,
                message=f"Authentication error: {exc}",
                outcome="ERROR",
            )

    return app


# ── Helper ────────────────────────────────────────────────────────────────

def _perms(path: Path, expected: int) -> bool:
    try:
        return stat.S_IMODE(path.stat().st_mode) == expected
    except OSError:
        return False
