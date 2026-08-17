"""
Tests for face quality scoring (blur detection) and recognition smoothing.
No camera or AI model required.
"""

import numpy as np
import pytest

from app.recognition.recognizer import RecognitionResult
from app.recognition.runner import _smooth
from collections import deque


# ── Blur / quality scoring ─────────────────────────────────────────────

class TestBlurScore:
    """
    DetectedFace.blur_score and .is_sharp are computed from the frame crop
    inside FaceDetector.detect().  We test the underlying helper logic
    independently by replicating the Laplacian variance calculation.
    """

    @staticmethod
    def _laplacian_var(frame: np.ndarray, bbox) -> float:
        import cv2
        x1, y1, x2, y2 = (int(max(0, v)) for v in bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def test_sharp_image_has_high_score(self):
        """A synthetic checkerboard has very high Laplacian variance."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Checkerboard pattern
        for i in range(20):
            for j in range(20):
                if (i + j) % 2 == 0:
                    frame[i*10:(i+1)*10, j*10:(j+1)*10] = 255
        score = self._laplacian_var(frame, [0, 0, 200, 200])
        assert score > 40.0

    def test_blank_image_has_low_score(self):
        """A uniform grey frame has zero Laplacian variance — totally blurry."""
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        score = self._laplacian_var(frame, [0, 0, 200, 200])
        assert score < 1.0

    def test_empty_crop_returns_zero(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # bbox outside frame — crop will be empty
        score = self._laplacian_var(frame, [90, 90, 90, 90])
        assert score == 0.0


# ── Recognition smoothing ──────────────────────────────────────────────

def _res(authorized: bool, username: str | None, similarity: float) -> RecognitionResult:
    return RecognitionResult(authorized=authorized, username=username,
                             similarity=similarity)


class TestSmoothing:

    def test_empty_buffer_returns_unknown(self):
        buf = deque(maxlen=5)
        result = _smooth(buf)
        assert not result.authorized
        assert result.username is None
        assert result.similarity == 0.0

    def test_all_authorized_gives_authorized(self):
        buf = deque(
            [_res(True, "alice", 0.8)] * 5,
            maxlen=5,
        )
        result = _smooth(buf)
        assert result.authorized
        assert result.username == "alice"
        assert abs(result.similarity - 0.8) < 1e-5

    def test_all_unknown_gives_unknown(self):
        buf = deque(
            [_res(False, None, 0.2)] * 5,
            maxlen=5,
        )
        result = _smooth(buf)
        assert not result.authorized

    def test_majority_authorized_wins(self):
        """3 authorized + 2 unknown → authorized."""
        buf = deque([
            _res(True,  "alice", 0.7),
            _res(True,  "alice", 0.75),
            _res(True,  "alice", 0.72),
            _res(False, None,    0.3),
            _res(False, None,    0.28),
        ], maxlen=5)
        result = _smooth(buf)
        assert result.authorized
        assert result.username == "alice"

    def test_majority_unknown_wins(self):
        """2 authorized + 3 unknown → unknown."""
        buf = deque([
            _res(True,  "alice", 0.7),
            _res(True,  "alice", 0.7),
            _res(False, None,    0.3),
            _res(False, None,    0.3),
            _res(False, None,    0.3),
        ], maxlen=5)
        result = _smooth(buf)
        assert not result.authorized

    def test_similarity_is_ema_weighted(self):
        """
        Similarity must be the EMA-weighted average, not a plain mean.
        The most-recent entry (last in deque) should carry the highest weight.
        """
        from app.recognition.runner import _EMA_ALPHA
        sims = [0.4, 0.5, 0.6, 0.7, 0.9]   # last = most recent
        buf = deque([_res(True, "bob", s) for s in sims], maxlen=5)
        result = _smooth(buf)

        n = len(sims)
        alpha = _EMA_ALPHA
        weights = np.array(
            [alpha * (1.0 - alpha) ** (n - 1 - i) for i in range(n)]
        )
        weights /= weights.sum()
        expected = float(np.dot(weights, sims))
        assert abs(result.similarity - expected) < 1e-5

    def test_ema_weights_recent_higher_than_old(self):
        """
        When the most-recent frame has a higher similarity, EMA result must
        exceed the plain mean.
        """
        # Older frames low, newest frame high
        sims_low_then_high = [0.3, 0.3, 0.3, 0.3, 0.9]
        buf = deque([_res(True, "bob", s) for s in sims_low_then_high], maxlen=5)
        result = _smooth(buf)
        plain_mean = float(np.mean(sims_low_then_high))
        assert result.similarity > plain_mean, (
            f"EMA {result.similarity:.4f} should exceed plain mean {plain_mean:.4f}"
        )

    def test_ema_alpha_zero_equals_mean(self):
        """With alpha=0, EMA degenerates to a plain mean."""
        sims = [0.5, 0.6, 0.7, 0.8, 0.9]
        buf = deque([_res(True, "bob", s) for s in sims], maxlen=5)
        result = _smooth(buf, alpha=0.0)
        assert abs(result.similarity - float(np.mean(sims))) < 1e-5

    def test_ema_alpha_one_equals_latest(self):
        """With alpha=1, EMA equals only the latest frame's similarity."""
        sims = [0.3, 0.4, 0.5, 0.6, 0.99]
        buf = deque([_res(True, "bob", s) for s in sims], maxlen=5)
        result = _smooth(buf, alpha=1.0)
        assert abs(result.similarity - 0.99) < 1e-5

    def test_most_frequent_username_returned(self):
        """When two names appear, the majority name should win."""
        buf = deque([
            _res(True, "alice", 0.8),
            _res(True, "alice", 0.8),
            _res(True, "alice", 0.8),
            _res(True, "bob",   0.6),
            _res(True, "bob",   0.6),
        ], maxlen=5)
        result = _smooth(buf)
        assert result.username == "alice"
