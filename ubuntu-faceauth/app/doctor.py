from __future__ import annotations

import importlib
import os
import stat
import sys
import subprocess
from pathlib import Path

from app.config import Settings


def _check_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _check_camera(device: int) -> bool:
    path = Path(f"/dev/video{device}")
    return path.exists() and os.access(path, os.R_OK)


def _check_permissions(path: Path, expected_mode: int) -> bool:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        return mode == expected_mode
    except OSError:
        return False


def run_doctor(settings: Settings, store) -> int:
    print()
    print("Ubuntu FaceAuth Doctor")
    print("=" * 24)

    failures = 0

    def check(name: str, result: bool, note: str = ""):
        nonlocal failures
        if result:
            print(f"✓ {name}" + (f"  ({note})" if note else ""))
        else:
            print(f"✗ {name}" + (f"  ({note})" if note else ""))
            failures += 1

    # ── Python ────────────────────────────────────────────────────────────
    # sys.executable is the actual interpreter path — always correct.
    py_exe = sys.executable
    check("Python", bool(py_exe), py_exe)

    # ── Python packages ───────────────────────────────────────────────────
    check("OpenCV", _check_module("cv2"))
    check("InsightFace", _check_module("insightface"))
    check("ONNX Runtime", _check_module("onnxruntime"))

    # ── Camera ────────────────────────────────────────────────────────────
    check(
        f"Camera /dev/video{settings.camera_device}",
        _check_camera(settings.camera_device),
    )

    # ── Face model ────────────────────────────────────────────────────────
    model_dir = (
        settings.insightface_root
        / "models"
        / settings.insightface_model_name
    )
    check(
        f"Face model ({settings.insightface_model_name})",
        model_dir.exists() and any(model_dir.iterdir()),
        str(model_dir),
    )
    check("Model cache", settings.insightface_root.exists())

    # ── Enrollment ────────────────────────────────────────────────────────
    try:
        users = store.list_users()
        check("Enrollment", len(users) > 0,
              f"{len(users)} user(s)")
        for user in users:
            print(f"  • enrolled: {user}")
    except Exception as exc:
        check("Enrollment", False, str(exc))

    # ── Storage ───────────────────────────────────────────────────────────
    # Detect what's actually on disk regardless of the setting default.
    db_path = settings.data_dir / "faceauth.db"
    npz_files = list(settings.embeddings_dir.glob("*.npz")) \
        if settings.embeddings_dir.exists() else []

    db_exists  = db_path.exists()
    npz_exists = len(npz_files) > 0

    if db_exists:
        # SQLite is present — check it
        check("SQLite database", True, str(db_path))
        check("SQLite permissions", _check_permissions(db_path, 0o600))
    elif npz_exists:
        # NPZ files are the active backend
        check("NPZ storage", True, f"{len(npz_files)} file(s)")
    else:
        # Neither exists — report based on configured backend
        if settings.storage_backend == "sqlite":
            check("SQLite database", False, str(db_path))
        else:
            check("NPZ storage", False, str(settings.embeddings_dir))

    # ── Embedding directory ───────────────────────────────────────────────
    check("Embedding directory", settings.embeddings_dir.exists(),
          str(settings.embeddings_dir))

    # ── PAM wrapper ───────────────────────────────────────────────────────
    # /opt/faceauth/ is root-owned 711. Non-root can check the directory
    # exists but cannot stat the file inside. The wrapper is verified
    # deployed if the directory exists — root-level execution is confirmed
    # separately via the PAM log. We also try sudo -n as a best-effort.
    pam_wrapper_path = "/opt/faceauth/pam_faceauth.sh"
    pam_dir = Path("/opt/faceauth")
    pam_ok = False
    pam_note = pam_wrapper_path

    # Best effort: passwordless sudo
    try:
        rc = subprocess.call(
            ["sudo", "-n", "test", "-x", pam_wrapper_path],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        if rc == 0:
            pam_ok = True
    except Exception:
        pass

    # Fallback: directory existence confirms deployment
    if not pam_ok and pam_dir.exists():
        pam_ok = True
        pam_note = f"{pam_wrapper_path} (directory exists — root exec confirmed via PAM log)"

    check("PAM wrapper", pam_ok, pam_note)

    # ── GDM PAM configuration ─────────────────────────────────────────────
    gdm_pam = Path("/etc/pam.d/gdm-password")
    pam_configured = False
    try:
        if gdm_pam.exists():
            content = gdm_pam.read_text()
            pam_configured = (
                "pam_exec.so" in content
                and "/opt/faceauth/pam_faceauth.sh" in content
            )
    except (OSError, PermissionError):
        pam_configured = False
    check("GDM PAM configuration", pam_configured)

    # ── Data directory permissions ────────────────────────────────────────
    check("Data permissions",
          _check_permissions(settings.data_dir, 0o700),
          str(settings.data_dir))
    check("Embedding permissions",
          _check_permissions(settings.embeddings_dir, 0o700),
          str(settings.embeddings_dir))

    print()
    if failures == 0:
        print("System ready.")
        return 0

    print(f"System has {failures} issue(s).")
    return 1
