"""
Ubuntu FaceAuth API server entry point.

Usage (direct):
    python server.py
    python server.py --host 127.0.0.1 --port 8765

Usage (via CLI):
    python main.py api
    python main.py api --host 127.0.0.1 --port 8765

Binds to 127.0.0.1 (localhost only) by default.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when run directly
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the Uvicorn server with the FaceAuth FastAPI application."""
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is not installed.")
        print("Install it with:  pip install 'uvicorn[standard]'")
        sys.exit(1)

    from app.api import create_app

    app = create_app()

    print(f"Ubuntu FaceAuth API starting on http://{host}:{port}")
    print(f"  Docs : http://{host}:{port}/docs")
    print(f"  Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port, log_level="info")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="server.py",
        description="Ubuntu FaceAuth local API server",
    )
    p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="TCP port (default: 8765)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_server(host=args.host, port=args.port)
