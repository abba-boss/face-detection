"""
Ubuntu FaceAuth — Local FastAPI layer.

Exposes read-only and authentication endpoints over HTTP on
127.0.0.1:8765 (default).  All endpoint logic delegates to the
existing FaceAuth modules — no duplication of business logic.

Security contract
-----------------
- Raw face embeddings are NEVER returned in any response.
- Passwords are not handled or stored by this layer.
- The server binds to localhost only by default.
- POST /api/authenticate runs the same headless auth flow as the
  CLI `authenticate --headless` command.
"""

from .routes import create_app

__all__ = ["create_app"]
