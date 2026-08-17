"""
Face recognition engine.

Compares a query embedding against every enrolled user and returns
the best match together with a cosine similarity score.

Cosine similarity of two L2-normalised vectors = their dot product.
Range: [-1, 1] where 1 = identical direction.

Typical ArcFace values:
  Same person      : 0.55 – 0.95
  Different person : 0.05 – 0.35
  Default threshold: 0.45  (configurable in Settings)

Performance note
----------------
Enrolled users are loaded from disk once and cached in memory.
Call `reload()` if you need to pick up a freshly enrolled user
during a running recognition session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.config import Settings
from app.security import get_logger
from app.storage import FaceStore, EnrolledUser


@dataclass
class RecognitionResult:
    authorized: bool
    username: Optional[str]      # None when UNKNOWN
    similarity: float            # best cosine similarity, clamped to [0, 1]
    all_scores: Dict[str, float] = field(default_factory=dict)
    # all_scores lets callers inspect every enrolled user's score —
    # useful for threshold tuning and debugging. Never log or display
    # this in production (it reveals enrolled identities).


class Recognizer:
    """Match a query embedding against all enrolled users."""

    def __init__(self, settings: Settings, store: FaceStore):
        self._settings = settings
        self._store = store
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )
        self._users: Optional[List[EnrolledUser]] = None  # lazy cache

    # ── Public API ───────────────────────────────────────────────────────

    def reload(self) -> None:
        """Force a fresh read from disk (call after enrolling a new user)."""
        self._users = None
        self._log.info("Recognition cache cleared — will reload on next frame")

    def identify(self, embedding: np.ndarray) -> RecognitionResult:
        """
        Compare *embedding* against all enrolled users.

        Returns the best match.  If the best score is below the configured
        threshold, the result is UNKNOWN.
        """
        users = self._get_users()

        if not users:
            self._log.warning("No enrolled users — cannot recognise anyone")
            return RecognitionResult(
                authorized=False, username=None, similarity=0.0
            )

        query = _l2_normalise(embedding.astype(np.float32))
        scores: Dict[str, float] = {}

        for user in users:
            scores[user.username] = float(np.dot(query, user.embedding))

        best_user = max(scores, key=lambda u: scores[u])
        best_score = scores[best_user]

        threshold = self._settings.recognition_threshold
        authorized = best_score >= threshold

        if authorized:
            self._log.info(
                "Known face — user: %s  similarity: %.3f",
                best_user, best_score,
            )
        else:
            self._log.debug(
                "Unknown face — best: %s  similarity: %.3f  threshold: %.2f",
                best_user, best_score, threshold,
            )

        return RecognitionResult(
            authorized=authorized,
            username=best_user if authorized else None,
            similarity=max(0.0, best_score),
            all_scores=scores,
        )

    # ── Private ──────────────────────────────────────────────────────────

    def _get_users(self) -> List[EnrolledUser]:
        """Return cached user list, loading from disk on first call."""
        if self._users is None:
            self._users = self._store.load_all()
            self._log.info(
                "Loaded %d enrolled user(s) into recognition cache",
                len(self._users),
            )
        return self._users


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v
