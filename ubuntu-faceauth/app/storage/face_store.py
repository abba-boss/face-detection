"""
Local face-embedding storage.

Design:
  - Each enrolled user has their own .npz file: data/embeddings/<username>.npz
  - Only the L2-normalised mean embedding is stored — no raw photos.
  - The format is intentionally simple so it can be migrated to SQLite
    without changing the recognition engine.

File format (numpy .npz, version 2):
  embedding    → float32 (512,)   — L2-normalised ArcFace embedding
  username     → str              — identity label
  enrolled_at  → str              — ISO-8601 UTC timestamp
  version      → int              — format version (for future migration)
"""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import Settings
from app.security import get_logger


_FORMAT_VERSION = 2


@dataclass
class EnrolledUser:
    username: str
    embedding: np.ndarray        # L2-normalised float32 (512,)
    enrolled_at: str = ""        # ISO-8601 UTC, e.g. "2026-08-10T17:14:26Z"


class FaceStore:
    """
    Reads and writes enrolled face embeddings to local .npz files.
    One file per user.  Storage path is configured in Settings.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )
        self._cache: Dict[str, EnrolledUser] = {}

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, username: str, embedding: np.ndarray) -> Path:
        """
        Persist *embedding* for *username*.

        The embedding is L2-normalised before saving so cosine similarity
        is a plain dot product at query time.
        The file is set to mode 0o600 (owner read/write only).
        """
        username = _sanitise(username)
        normed = _l2_normalise(embedding)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        path = self._settings.embeddings_dir / f"{username}.npz"
        np.savez(
            path,
            embedding=normed,
            username=np.array(username),
            enrolled_at=np.array(now),
            version=np.array(_FORMAT_VERSION),
        )

        # Restrict file permissions to owner only (security best practice)
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass   # Non-fatal — may fail in some container environments

        self._log.info(
            "Saved embedding for user '%s' → %s  (enrolled_at=%s)",
            username, path.name, now,
        )

        # Invalidate in-memory cache
        self._cache.pop(username, None)
        return path

    def load(self, username: str) -> Optional[EnrolledUser]:
        """Load a single user's embedding.  Returns None if not enrolled."""
        username = _sanitise(username)
        if username in self._cache:
            return self._cache[username]

        path = self._settings.embeddings_dir / f"{username}.npz"
        if not path.exists():
            return None

        try:
            data = np.load(path, allow_pickle=False)
            enrolled_at = (
                str(data["enrolled_at"])
                if "enrolled_at" in data else ""
            )
            user = EnrolledUser(
                username=str(data["username"]),
                embedding=data["embedding"].astype(np.float32),
                enrolled_at=enrolled_at,
            )
            self._cache[username] = user
            return user
        except Exception as exc:
            self._log.error(
                "Failed to load embedding for '%s': %s", username, exc
            )
            return None

    def load_all(self) -> List[EnrolledUser]:
        """Load every enrolled user from the embeddings directory."""
        users: List[EnrolledUser] = []
        for path in sorted(self._settings.embeddings_dir.glob("*.npz")):
            user = self.load(path.stem)
            if user:
                users.append(user)
        self._log.debug("Loaded %d enrolled user(s)", len(users))
        return users

    def delete(self, username: str) -> bool:
        """Remove an enrolled user.  Returns True if the file existed."""
        username = _sanitise(username)
        path = self._settings.embeddings_dir / f"{username}.npz"
        self._cache.pop(username, None)
        if path.exists():
            path.unlink()
            self._log.info("Deleted enrollment for user '%s'", username)
            return True
        return False

    def list_users(self) -> List[str]:
        """Return sorted list of enrolled usernames."""
        return sorted(
            p.stem for p in self._settings.embeddings_dir.glob("*.npz")
        )

    def is_enrolled(self, username: str) -> bool:
        return _sanitise(username) in self.list_users()

    def enrollment_info(self, username: str) -> Optional[dict]:
        """
        Return display metadata for a user without exposing the embedding.

        Returns None if the user is not enrolled.
        """
        user = self.load(username)
        if user is None:
            return None
        return {
            "username": user.username,
            "enrolled_at": user.enrolled_at or "unknown",
        }


# ── Helpers ──────────────────────────────────────────────────────────────

def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def _sanitise(username: str) -> str:
    """Allow only alphanumeric + underscores/hyphens (prevent path traversal)."""
    clean = re.sub(r"[^\w\-]", "_", username.strip())
    if not clean:
        raise ValueError(f"Invalid username: {username!r}")
    return clean.lower()
