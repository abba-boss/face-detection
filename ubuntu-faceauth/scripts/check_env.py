#!/usr/bin/env python3
"""
scripts/check_env.py — Ubuntu FaceAuth environment diagnostics.

Run this before first use to verify all dependencies are in order:

    conda activate face-detection
    python scripts/check_env.py
"""

import importlib
import platform
import sys
from pathlib import Path

# ── ANSI helpers ─────────────────────────────────────────────────────────

PASS  = "\033[32m✓\033[0m"
FAIL  = "\033[31m✗\033[0m"
WARN  = "\033[33m⚠\033[0m"


def _ok(label: str, detail: str = "") -> bool:
    print(f"  {PASS}  {label}" + (f"  ({detail})" if detail else ""))
    return True


def _fail(label: str, detail: str = "") -> bool:
    print(f"  {FAIL}  {label}" + (f"  ({detail})" if detail else ""))
    return False


def _warn(label: str, detail: str = "") -> None:
    print(f"  {WARN}  {label}" + (f"  ({detail})" if detail else ""))


def _check_import(module: str, attr: str = "__version__",
                  min_ver: str = "") -> bool:
    """Try to import *module* and optionally enforce a minimum version."""
    try:
        mod = importlib.import_module(module)
        ver = str(getattr(mod, attr, "?"))
    except ImportError as exc:
        return _fail(module, str(exc))

    if min_ver:
        try:
            from packaging.version import Version
            if Version(ver) < Version(min_ver):
                return _fail(module, f"v{ver} — need >= {min_ver}")
        except Exception:
            pass   # packaging not available — skip version check

    return _ok(module, f"v{ver}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n╔══════════════════════════════════════════╗")
    print("║   Ubuntu FaceAuth — Environment Check   ║")
    print("╚══════════════════════════════════════════╝\n")

    results: list[bool] = []

    # ── System ────────────────────────────────────────────────────────────
    print("System:")
    results.append(_ok("Platform", platform.platform()))
    py_ok = sys.version_info >= (3, 10)
    results.append(
        _ok("Python >= 3.10", sys.version.split()[0]) if py_ok
        else _fail("Python >= 3.10", sys.version.split()[0])
    )

    # ── Core dependencies ─────────────────────────────────────────────────
    print("\nCore dependencies:")
    results.append(_check_import("cv2",          min_ver="4.0.0"))
    results.append(_check_import("numpy",        min_ver="2.0.0"))
    results.append(_check_import("onnxruntime",  min_ver="1.19.0"))
    results.append(_check_import("insightface",  min_ver="0.7.3"))
    results.append(_check_import("PIL",          attr="__version__"))
    results.append(_check_import("sklearn",      min_ver="1.3.0"))
    results.append(_check_import("skimage",      min_ver="0.21.0"))
    results.append(_check_import("scipy",        min_ver="1.11.0"))
    results.append(_check_import("packaging",    min_ver="23.0"))

    # ── Camera devices ────────────────────────────────────────────────────
    print("\nCamera devices:")
    found_any = False
    for i in range(4):
        dev = Path(f"/dev/video{i}")
        if dev.exists():
            _ok(f"/dev/video{i}", "present")
            found_any = True
    if not found_any:
        results.append(_fail("Camera devices", "none found under /dev/video*"))
    else:
        results.append(True)

    # ── InsightFace model cache ────────────────────────────────────────────
    print("\nInsightFace model:")
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_sc"
    cached = model_dir.exists() and any(model_dir.iterdir())
    if cached:
        det  = model_dir / "det_500m.onnx"
        recg = model_dir / "w600k_mbf.onnx"
        both = det.exists() and recg.exists()
        results.append(
            _ok("buffalo_sc model cached", str(model_dir)) if both
            else _warn("buffalo_sc partial — re-run to complete download") or True
        )
    else:
        _warn("buffalo_sc not cached", "will download ~14 MB on first run")
        results.append(True)   # not a hard failure

    # ── Project data directory ────────────────────────────────────────────
    print("\nProject data directory:")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    results.append(
        _ok("data/ exists", str(data_dir)) if data_dir.exists()
        else _fail("data/ missing", str(data_dir))
    )
    emb_dir = data_dir / "embeddings"
    if emb_dir.exists():
        users = sorted(emb_dir.glob("*.npz"))
        _ok(f"embeddings/  ({len(users)} user(s) enrolled)", str(emb_dir))
    else:
        _ok("embeddings/  (none yet)", "will be created on first enroll")

    # ── Quick camera open test ─────────────────────────────────────────────
    print("\nQuick camera test:")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        cap.release()
        results.append(
            _ok("cv2.VideoCapture(0)", "opened OK") if opened
            else _fail("cv2.VideoCapture(0)", "FAILED — check /dev/video0 permissions")
        )
    except Exception as exc:
        results.append(_fail("cv2.VideoCapture(0)", str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────
    all_ok = all(results)
    print()
    if all_ok:
        print(f"{PASS}  All checks passed — ready to run Ubuntu FaceAuth.\n")
        print("Next steps:")
        print("  python main.py enroll --user <name>")
        print("  python main.py recognize\n")
    else:
        failed = results.count(False)
        print(f"{FAIL}  {failed} check(s) failed — fix the issues above before running.\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
