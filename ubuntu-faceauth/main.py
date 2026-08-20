#!/usr/bin/env python3
"""
Ubuntu FaceAuth — Phase 1 + 2A + V3 CLI entry point.

Usage
-----
Enroll a user:
    python main.py enroll --user abba

Re-enroll (overwrite existing):
    python main.py enroll --user abba --re-enroll

Recognise faces live:
    python main.py recognize

Authenticate a specific user (liveness + recognition):
    python main.py authenticate --user abba

Run standalone liveness challenge:
    python main.py liveness

List enrolled users:
    python main.py list

Delete an enrolled user:
    python main.py delete --user abba

Show version info:
    python main.py version

Run system diagnostics:
    python main.py doctor

Environment check:
    python scripts/check_env.py

Liveness demo (no camera):
    python scripts/test_liveness_demo.py

Inference benchmark:
    python scripts/benchmark.py
"""

import argparse
import sys

from app.config import Settings
from app.detection import FaceDetector
from app.enrollment import EnrollmentSession
from app.liveness import LivenessSession, LivenessState
from app.recognition import RecognitionRunner
from app.security import get_logger
from app.storage import FaceStore
from app.auth import AuthSession, AuthOutcome, HeadlessAuthSession
from app.doctor import run_doctor
from app.logs import run_logs

__version__ = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ubuntu-faceauth",
        description="Ubuntu FaceAuth — local biometric face recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py enroll --user abba\n"
            "  python main.py recognize --threshold 0.50\n"
            "  python main.py liveness --timeout 10\n"
            "  python main.py list\n"
            "  python main.py delete --user abba\n"
            "  python main.py doctor\n"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Ubuntu FaceAuth {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── enroll ──────────────────────────────────────────────────────────
    enroll_p = sub.add_parser("enroll", help="Enroll a new face")
    enroll_p.add_argument(
        "--user",
        required=True,
        help="Username to associate with the enrolled face",
    )
    enroll_p.add_argument(
        "--re-enroll",
        action="store_true",
        dest="re_enroll",
        help="Overwrite an existing enrollment for this user",
    )
    enroll_p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override number of enrollment samples (default: 10)",
    )
    enroll_p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index (default: 0 → /dev/video0)",
    )
    enroll_p.add_argument(
        "--model",
        choices=["buffalo_sc", "buffalo_l"],
        default=None,
        help=(
            "InsightFace model (default: buffalo_sc — fast CPU; "
            "buffalo_l — higher accuracy, ~500 MB download on first use)"
        ),
    )

    # ── recognize ───────────────────────────────────────────────────────
    rec_p = sub.add_parser(
        "recognize",
        help="Run real-time face recognition",
    )
    rec_p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index (default: 0)",
    )
    rec_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold to authorise (0.0–1.0, default: 0.45)",
    )
    rec_p.add_argument(
        "--debug",
        action="store_true",
        help="Print per-user similarity scores in the terminal (for tuning)",
    )
    rec_p.add_argument(
        "--liveness",
        action="store_true",
        help="Require a head-turn liveness challenge before granting access",
    )
    rec_p.add_argument(
        "--model",
        choices=["buffalo_sc", "buffalo_l"],
        default=None,
        help="InsightFace model (default: buffalo_sc)",
    )

    # ── list / users ─────────────────────────────────────────────────────
    sub.add_parser("list",  help="List enrolled users")
    sub.add_parser("users", help="List enrolled users (alias for 'list')")

    # ── version ─────────────────────────────────────────────────────────
    sub.add_parser("version", help="Show version and package info")

    # ── doctor ──────────────────────────────────────────────────────────
    sub.add_parser("doctor", help="Run system diagnostics")

    # ── status ──────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show current configuration and enrollment")

    # ── logs ────────────────────────────────────────────────────────────
    logs_p = sub.add_parser("logs", help="Show recent authentication events")
    logs_p.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        help="Number of recent events to show (default: 20)",
    )

    # ── liveness ─────────────────────────────────────────────────────────
    live_p = sub.add_parser(
        "liveness",
        help="Run a standalone liveness challenge",
    )
    live_p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index (default: 0)",
    )
    live_p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override challenge timeout in seconds (default: 8.0)",
    )
    live_p.add_argument(
        "--model",
        choices=["buffalo_sc", "buffalo_l"],
        default=None,
        help="InsightFace model (default: buffalo_sc)",
    )

    # ── delete ──────────────────────────────────────────────────────────
    del_p = sub.add_parser("delete", help="Delete an enrolled user")
    del_p.add_argument(
        "--user",
        required=True,
        help="Username to remove",
    )

    # ── authenticate ────────────────────────────────────────────────────
    auth_p = sub.add_parser(
        "authenticate",
        help="Authenticate a user: liveness + face recognition",
    )
    auth_p.add_argument(
        "--user",
        required=True,
        help="Username to authenticate against",
    )
    auth_p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index (default: 0)",
    )
    auth_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override similarity threshold (default: 0.45)",
    )
    auth_p.add_argument(
        "--model",
        choices=["buffalo_sc", "buffalo_l"],
        default=None,
        help="InsightFace model (default: buffalo_sc)",
    )
    auth_p.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without any GUI (no cv2.imshow/waitKey). "
            "Required when called from PAM/GDM where no display is available. "
            "All output goes to stdout for pam_exec.so logging."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ── Build settings ───────────────────────────────────────────────────
    settings = Settings()

    if getattr(args, "camera", None) is not None:
        settings.camera_device = args.camera

    if getattr(args, "samples", None) is not None:
        settings.enrollment_samples = args.samples

    if getattr(args, "threshold", None) is not None:
        settings.recognition_threshold = args.threshold

    if getattr(args, "timeout", None) is not None:
        settings.liveness_timeout = args.timeout

    if getattr(args, "model", None) is not None:
        settings.insightface_model_name = args.model

    log = get_logger(
        "faceauth.main",
        log_file=settings.log_file if settings.log_to_file else None,
        level=settings.log_level,
    )

    log.info(
        "Ubuntu FaceAuth started — command=%s",
        args.command,
    )

    print("Ubuntu FaceAuth")

    store = FaceStore(settings)

    # ── list / users ─────────────────────────────────────────────────────
    if args.command in ("list", "users"):
        users = store.list_users()

        if users:
            print(f"Enrolled users ({len(users)}):")

            for u in users:
                info = store.enrollment_info(u)
                date = info["enrolled_at"] if info else "unknown"
                print(f"  • {u:<20}  enrolled: {date}")
        else:
            print("No users enrolled yet.")

        return 0

    # ── version ──────────────────────────────────────────────────────────
    if args.command == "version":
        import importlib

        print(f"Ubuntu FaceAuth  {__version__}")
        print(f"  model            {settings.insightface_model_name}")

        pkgs = [
            ("insightface", "insightface"),
            ("onnxruntime", "onnxruntime"),
            ("cv2", "opencv"),
            ("numpy", "numpy"),
            ("skimage", "scikit-image"),
        ]

        for mod, label in pkgs:
            try:
                v = importlib.import_module(mod).__version__
            except Exception:
                v = "not installed"

            print(f"  {label:<16} {v}")

        return 0

    # ── doctor ───────────────────────────────────────────────────────────
    if args.command == "doctor":
        return run_doctor(settings, store)

    # ── status ───────────────────────────────────────────────────────────
    if args.command == "status":
        users = store.list_users()
        print(f"  version    : {__version__}")
        print(f"  model      : {settings.insightface_model_name}")
        print(f"  camera     : /dev/video{settings.camera_device}")
        print(f"  threshold  : {settings.recognition_threshold}")
        print(f"  liveness   : {settings.liveness_timeout}s timeout")
        print(f"  storage    : {settings.storage_backend}")
        print(f"  enrolled   : {len(users)} user(s)"
              + (f" — {', '.join(users)}" if users else ""))
        return 0

    # ── logs ─────────────────────────────────────────────────────────────
    if args.command == "logs":
        return run_logs(settings.log_file, limit=getattr(args, "limit", 20))

    # ── delete ───────────────────────────────────────────────────────────
    if args.command == "delete":
        if store.delete(args.user):
            print(f"User '{args.user}' deleted.")
        else:
            print(f"User '{args.user}' is not enrolled.")

        return 0

    # ── Commands that need the AI model ──────────────────────────────────
    print("Starting camera…")

    if settings.insightface_model_name == "buffalo_l":
        print(
            "[MODEL] buffalo_l selected — higher accuracy, "
            "first run downloads ~500 MB …"
        )

    detector = FaceDetector(settings)

    try:
        detector.load()
    except Exception as exc:
        log.error("Failed to load face model: %s", exc)
        print(f"[ERROR] Model failed to load: {exc}")
        return 1

    # ── enroll ──────────────────────────────────────────────────────────
    if args.command == "enroll":
        # Guard against accidental overwrite
        if store.is_enrolled(args.user) and not args.re_enroll:
            print(
                f"User '{args.user}' is already enrolled.\n"
                f"Use --re-enroll to overwrite the existing data."
            )

            log.warning(
                "Enrollment blocked — '%s' already enrolled (use --re-enroll)",
                args.user,
            )

            return 1

        session = EnrollmentSession(
            settings,
            detector,
            store,
        )

        success = session.run(args.user)

        return 0 if success else 1

    # ── recognize ───────────────────────────────────────────────────────
    if args.command == "recognize":
        enrolled = store.list_users()

        if not enrolled:
            print(
                "[WARNING] No users are enrolled yet.\n"
                "Everyone will appear as UNKNOWN.\n"
                "Run:  python main.py enroll --user <name>"
            )

            log.warning(
                "Recognition started with no enrolled users"
            )

        runner = RecognitionRunner(
            settings,
            detector,
            store,
        )

        runner.run(
            debug=getattr(args, "debug", False),
            liveness=getattr(args, "liveness", False),
        )

        return 0

    # ── liveness ─────────────────────────────────────────────────────────
    if args.command == "liveness":
        session = LivenessSession(
            settings,
            detector,
        )

        state = session.run()

        return 0 if state == LivenessState.LIVE else 1

    # ── authenticate ────────────────────────────────────────────────────
    if args.command == "authenticate":
        headless = getattr(args, "headless", False)

        if headless:
            session = HeadlessAuthSession(
                settings,
                detector,
                store,
            )
        else:
            session = AuthSession(
                settings,
                detector,
                store,
            )

        result = session.run(args.user)

        # Print outcome clearly for both human and script consumers
        if result.success:
            print("\n✓  AUTHENTICATION SUCCESS")
            print(f"   User      : {result.username}")
            print(f"   Similarity: {result.similarity:.2f}")
        else:
            print("\n✗  AUTHENTICATION DENIED")
            print(f"   Reason: {result.message}")

        log.info(
            "Authentication outcome — user=%s  outcome=%s  similarity=%.3f",
            args.user,
            result.outcome.name,
            result.similarity,
        )

        return 0 if result.success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())