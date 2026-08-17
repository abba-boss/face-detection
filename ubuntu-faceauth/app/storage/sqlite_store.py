"""
SQLite-backed face-embedding storage.

Drop-in replacement for the .npz FaceStore — identical public API.

Schema (single table):
    CREATE TABLE enrollments (
        username    TEXT PRIMARY KEY,
        embedding   BLOB NOT NULL,        -- float32 (512,) packed as bytes
        enrolled_at TEXT NOT NULL,        -- ISO-8601 UTC
        version     INTEGER NOT NULL      -- format version
    )

Security:
    - Database file set to mode 0o600 (owner read/write only).
    - No plaintext photos or raw embeddings in any log.
    - Embeddings are L2-normalised before storage.

Migration:
    Run  python scripts/migrate_to_sqlite.py  to import existing .npz files.
"""

from __future__ import annotations

import re
import sqlite3
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np

from app.config import Settings
from app.security import get_logger


_DB_VERSION  = 1
_EMBED_DTYPE = np.float32
_EMBED_DIM   = 512

# Re-export so callers can use `from app.storage.sqlite_store import EnrolledUser`
from app.storage.face_store import EnrolledUser   # noqa: E402


class SQLiteFaceStore:
    """
    Stores enrolled face embeddings in a local SQLite database.

    Public API is identical to FaceStore — callers never need to know
    which backend is in use.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._db_path  = settings.data_dir / "faceauth.db"
        self._log      = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )
        self._cache: Dict[str, EnrolledUser] = {}
        self._init_db()

    # ── Public API (mirrors FaceStore exactly) ────────────────────────────

    def save(self, username: str, embedding: np.ndarray) -> Path:
        """Persist *embedding* for *username*. Returns the database path."""
        username = _sanitise(username)
        normed   = _l2_normalise(embedding)
        now      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        blob     = normed.astype(_EMBED_DTYPE).tobytes()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrollments (username, embedding, enrolled_at, version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    embedding   = excluded.embedding,
                    enrolled_at = excluded.enrolled_at,
                    version     = excluded.version
                """,
                (username, blob, now, _DB_VERSION),
            )

        self._cache.pop(username, None)
        self._log.info(
            "Saved embedding for user '%s' → faceauth.db  (enrolled_at=%s)",
            username, now,
        )
        return self._db_path

    def load(self, username: str) -> Optional[EnrolledUser]:
        """Load a single user's embedding.  Returns None if not enrolled."""
        username = _sanitise(username)
        if username in self._cache:
            return self._cache[username]

        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, embedding, enrolled_at FROM enrollments "
                "WHERE username = ?",
                (username,),
            ).fetchone()

        if row is None:
            return None

        try:
            emb = np.frombuffer(row[1], dtype=_EMBED_DTYPE).copy()
            if emb.shape != (_EMBED_DIM,):
                self._log.error(
                    "Corrupt embedding for '%s': shape %s", username, emb.shape
                )
                return None
            user = EnrolledUser(
                username=row[0],
                embedding=emb,
                enrolled_at=row[2] or "",
            )
            self._cache[username] = user
            return user
        except Exception as exc:
            self._log.error(
                "Failed to load embedding for '%s': %s", username, exc
            )
            return None

    def load_all(self) -> List[EnrolledUser]:
        """Load every enrolled user."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username FROM enrollments ORDER BY username"
            ).fetchall()
        users = []
        for (uname,) in rows:
            user = self.load(uname)
            if user:
                users.append(user)
        self._log.debug("Loaded %d enrolled user(s)", len(users))
        return users

    def delete(self, username: str) -> bool:
        """Remove an enrolled user.  Returns True if the record existed."""
        username = _sanitise(username)
        self._cache.pop(username, None)
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM enrollments WHERE username = ?", (username,)
            )
            deleted = cur.rowcount > 0
        if deleted:
            self._log.info("Deleted enrollment for user '%s'", username)
        return deleted

    def list_users(self) -> List[str]:
        """Return sorted list of enrolled usernames."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username FROM enrollments ORDER BY username"
            ).fetchall()
        return [r[0] for r in rows]

    def is_enrolled(self, username: str) -> bool:
        return _sanitise(username) in self.list_users()

    def enrollment_info(self, username: str) -> Optional[dict]:
        """Return display metadata without exposing the embedding."""
        user = self.load(username)
        if user is None:
            return None
        return {
            "username":    user.username,
            "enrolled_at": user.enrolled_at or "unknown",
        }

    # ── Private ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the database and table if they don't already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrollments (
                    username    TEXT PRIMARY KEY,
                    embedding   BLOB NOT NULL,
                    enrolled_at TEXT NOT NULL DEFAULT '',
                    version     INTEGER NOT NULL DEFAULT 1
                )
                """
            )
        self._log.debug("SQLite store initialised: %s", self._db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a database connection with:
          - WAL journal mode for safe concurrent reads
          - strict foreign keys
          - file restricted to owner read/write (0o600) on first creation
        """
        is_new = not self._db_path.exists()
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

        # Restrict permissions after first creation
        if is_new:
            try:
                self._db_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass


# ── Helpers ──────────────────────────────────────────────────────────────

def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def _sanitise(username: str) -> str:
    """Allow only alphanumeric + underscores/hyphens (prevent injection)."""
    clean = re.sub(r"[^\w\-]", "_", username.strip())
    if not clean:
        raise ValueError(f"Invalid username: {username!r}")
    return clean.lower()
