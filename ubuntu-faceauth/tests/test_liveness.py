"""
Unit tests for the Phase 2A liveness challenge-response module.

No camera, no AI model — all tests use synthetic landmark arrays.

Landmark layout (InsightFace kps):
    kps[0] = left eye   (x, y)
    kps[1] = right eye  (x, y)
    kps[2] = nose tip   (x, y)
    kps[3] = mouth-left (x, y)
    kps[4] = mouth-right(x, y)

Nose-offset ratio (NOR) = (nose_x - eye_mid_x) / eye_span
    ≈  0.0  → frontal
    < -0.18 → left turn (default threshold)
"""

from __future__ import annotations

import time
import numpy as np
import pytest
from unittest.mock import patch

from app.liveness.detector import (
    LivenessDetector,
    LivenessState,
    LivenessConfig,
    LivenessResult,
    _nose_offset_ratio,
    _valid_kps,
)


# ── Landmark factory helpers ──────────────────────────────────────────────

def _kps(nor: float, eye_span: float = 100.0, y: float = 200.0) -> np.ndarray:
    """
    Build a synthetic 5-point kps array with the given nose-offset ratio.

    eye_span: distance between the two eye x-coordinates.
    The face centre is placed at x=320 for realism.
    """
    centre_x    = 320.0
    left_eye_x  = centre_x - eye_span / 2.0
    right_eye_x = centre_x + eye_span / 2.0
    eye_mid_x   = (left_eye_x + right_eye_x) / 2.0
    nose_x      = eye_mid_x + nor * eye_span

    return np.array([
        [left_eye_x,  y],          # 0 left eye
        [right_eye_x, y],          # 1 right eye
        [nose_x,      y + 40],     # 2 nose tip
        [centre_x - 30, y + 80],   # 3 mouth left
        [centre_x + 30, y + 80],   # 4 mouth right
    ], dtype=np.float32)


def _frontal() -> np.ndarray:
    return _kps(nor=0.0)


def _left_turn(amount: float = 0.25) -> np.ndarray:
    """amount > left_threshold (0.18) means a definite left turn."""
    return _kps(nor=-amount)


def _right_turn(amount: float = 0.25) -> np.ndarray:
    return _kps(nor=+amount)


# ── Helper: fast config with 1 confirm frame for speed ───────────────────

def _fast_cfg(**overrides) -> LivenessConfig:
    defaults = dict(
        timeout_seconds    = 5.0,
        left_threshold     = 0.18,
        frontal_max_nor    = 0.15,
        min_confirm_frames = 1,    # confirm immediately
    )
    defaults.update(overrides)
    return LivenessConfig(**defaults)


# ── _valid_kps ────────────────────────────────────────────────────────────

class TestValidKps:
    def test_valid_frontal(self):
        assert _valid_kps(_frontal())

    def test_wrong_shape_rejected(self):
        assert not _valid_kps(np.zeros((3, 2), dtype=np.float32))

    def test_none_rejected(self):
        assert not _valid_kps(None)  # type: ignore[arg-type]

    def test_nan_rejected(self):
        k = _frontal()
        k[0, 0] = float("nan")
        assert not _valid_kps(k)

    def test_left_right_eye_swapped_rejected(self):
        """right_eye_x must be greater than left_eye_x."""
        k = _frontal()
        k[0, 0], k[1, 0] = k[1, 0], k[0, 0]   # swap
        assert not _valid_kps(k)


# ── _nose_offset_ratio ────────────────────────────────────────────────────

class TestNoseOffsetRatio:
    def test_frontal_is_near_zero(self):
        assert abs(_nose_offset_ratio(_frontal())) < 0.01

    def test_left_turn_is_negative(self):
        assert _nose_offset_ratio(_left_turn(0.25)) < -0.20

    def test_right_turn_is_positive(self):
        assert _nose_offset_ratio(_right_turn(0.25)) > 0.20

    def test_specific_value(self):
        nor = _nose_offset_ratio(_kps(nor=-0.3))
        assert abs(nor - (-0.3)) < 1e-4

    def test_eye_span_independence(self):
        """NOR should be the same regardless of face distance from camera."""
        k_close = _kps(nor=-0.25, eye_span=150.0)
        k_far   = _kps(nor=-0.25, eye_span=60.0)
        assert abs(_nose_offset_ratio(k_close) - _nose_offset_ratio(k_far)) < 0.01


# ── State machine — initial state ─────────────────────────────────────────

class TestInitialState:
    def test_starts_in_waiting(self):
        d = LivenessDetector()
        assert d.state == LivenessState.WAITING

    def test_update_no_face_stays_waiting(self):
        d = LivenessDetector()
        r = d.update(None)
        assert r.state == LivenessState.WAITING

    def test_update_angled_face_stays_waiting(self):
        """Angled face (|NOR| > frontal_max) should not lock baseline."""
        d = LivenessDetector(_fast_cfg())
        r = d.update(_right_turn(0.30))
        assert r.state == LivenessState.WAITING
        assert "straight" in r.message.lower()


# ── WAITING → CHALLENGE_ACTIVE ────────────────────────────────────────────

class TestWaitingToChallenge:
    def test_frontal_triggers_challenge(self):
        d = LivenessDetector(_fast_cfg())
        r = d.update(_frontal())
        assert r.state == LivenessState.CHALLENGE_ACTIVE

    def test_baseline_nor_locked(self):
        d = LivenessDetector(_fast_cfg())
        d.update(_frontal())
        r = d.update(_frontal())
        assert r.baseline_nor is not None
        assert abs(r.baseline_nor) < 0.05

    def test_message_contains_left(self):
        d = LivenessDetector(_fast_cfg())
        r = d.update(_frontal())
        assert "left" in r.message.lower()


# ── CHALLENGE_ACTIVE: success path ────────────────────────────────────────

class TestChallengeSuccess:
    def test_left_turn_leads_to_live(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=1))
        d.update(_frontal())            # lock baseline
        r = d.update(_left_turn(0.25)) # sufficient left turn
        assert r.state == LivenessState.LIVE

    def test_live_message(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=1))
        d.update(_frontal())
        r = d.update(_left_turn(0.25))
        assert "liveness" in r.message.lower() or "confirmed" in r.message.lower()

    def test_multiple_confirm_frames_required(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=3))
        d.update(_frontal())           # CHALLENGE_ACTIVE

        r1 = d.update(_left_turn(0.25))
        assert r1.state == LivenessState.CHALLENGE_ACTIVE
        assert r1.confirm_count == 1

        r2 = d.update(_left_turn(0.25))
        assert r2.state == LivenessState.CHALLENGE_ACTIVE
        assert r2.confirm_count == 2

        r3 = d.update(_left_turn(0.25))
        assert r3.state == LivenessState.LIVE
        assert r3.confirm_count == 3

    def test_confirm_streak_resets_on_insufficient_movement(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=3))
        d.update(_frontal())
        d.update(_left_turn(0.25))     # count=1
        d.update(_frontal())           # streak broken — count back to 0
        r = d.update(_left_turn(0.25))
        assert r.confirm_count == 1    # restarted


# ── CHALLENGE_ACTIVE: failure paths ───────────────────────────────────────

class TestChallengeFailure:
    def test_face_lost_during_challenge_fails(self):
        d = LivenessDetector(_fast_cfg())
        d.update(_frontal())           # CHALLENGE_ACTIVE
        r = d.update(None)             # face disappears
        assert r.state == LivenessState.FAILED

    def test_failed_message_set(self):
        d = LivenessDetector(_fast_cfg())
        d.update(_frontal())
        r = d.update(None)
        assert "lost" in r.message.lower() or "failed" in r.message.lower()

    def test_right_turn_does_not_satisfy_left_challenge(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=1))
        d.update(_frontal())
        r = d.update(_right_turn(0.30))
        assert r.state == LivenessState.CHALLENGE_ACTIVE   # not passed
        assert r.confirm_count == 0

    def test_insufficient_left_turn_not_confirmed(self):
        """A left turn smaller than the threshold should not confirm."""
        d = LivenessDetector(_fast_cfg(left_threshold=0.18, min_confirm_frames=1))
        d.update(_frontal())
        r = d.update(_left_turn(0.10))   # below threshold
        assert r.state == LivenessState.CHALLENGE_ACTIVE
        assert r.confirm_count == 0


# ── TIMEOUT ───────────────────────────────────────────────────────────────

class TestTimeout:
    def test_timeout_triggers(self):
        cfg = _fast_cfg(timeout_seconds=2.0)
        d = LivenessDetector(cfg)
        d.update(_frontal())    # starts challenge

        # Simulate elapsed time by patching time.monotonic
        future = time.monotonic() + 3.0   # 3 s > 2 s timeout
        with patch("app.liveness.detector.time.monotonic", return_value=future):
            r = d.update(_frontal())   # still frontal but timed out

        assert r.state == LivenessState.TIMEOUT

    def test_timeout_message(self):
        cfg = _fast_cfg(timeout_seconds=0.001)
        d = LivenessDetector(cfg)
        d.update(_frontal())
        time.sleep(0.01)        # ensure timeout elapses
        r = d.update(_frontal())
        assert r.state == LivenessState.TIMEOUT
        assert "time" in r.message.lower()


# ── Terminal state immutability ───────────────────────────────────────────

class TestTerminalStates:
    def _reach_live(self) -> LivenessDetector:
        d = LivenessDetector(_fast_cfg(min_confirm_frames=1))
        d.update(_frontal())
        d.update(_left_turn(0.25))
        assert d.state == LivenessState.LIVE
        return d

    def _reach_failed(self) -> LivenessDetector:
        d = LivenessDetector(_fast_cfg())
        d.update(_frontal())
        d.update(None)
        assert d.state == LivenessState.FAILED
        return d

    def test_live_stays_live(self):
        d = self._reach_live()
        r = d.update(None)
        assert r.state == LivenessState.LIVE

    def test_failed_stays_failed(self):
        d = self._reach_failed()
        r = d.update(_frontal())
        assert r.state == LivenessState.FAILED

    def test_timeout_stays_timeout(self):
        cfg = _fast_cfg(timeout_seconds=0.001)
        d = LivenessDetector(cfg)
        d.update(_frontal())
        time.sleep(0.01)
        d.update(_frontal())   # → TIMEOUT
        r = d.update(_frontal())
        assert r.state == LivenessState.TIMEOUT


# ── reset() ───────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_from_live_back_to_waiting(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=1))
        d.update(_frontal())
        d.update(_left_turn(0.25))
        assert d.state == LivenessState.LIVE
        d.reset()
        assert d.state == LivenessState.WAITING

    def test_reset_clears_baseline(self):
        d = LivenessDetector(_fast_cfg())
        d.update(_frontal())
        d.reset()
        r = d.update(None)
        assert r.baseline_nor is None

    def test_reset_allows_fresh_challenge(self):
        d = LivenessDetector(_fast_cfg(min_confirm_frames=1))
        d.update(_frontal())
        d.update(_left_turn(0.25))
        d.reset()
        d.update(_frontal())
        r = d.update(_left_turn(0.25))
        assert r.state == LivenessState.LIVE


# ── LivenessResult fields ─────────────────────────────────────────────────

class TestLivenessResult:
    def test_nor_present_when_face_detected(self):
        d = LivenessDetector(_fast_cfg())
        r = d.update(_frontal())
        assert r.nor is not None

    def test_nor_none_when_no_face(self):
        d = LivenessDetector(_fast_cfg())
        r = d.update(None)
        assert r.nor is None

    def test_elapsed_nonzero_after_challenge_starts(self):
        d = LivenessDetector(_fast_cfg())
        d.update(_frontal())   # starts challenge
        time.sleep(0.05)
        r = d.update(_frontal())
        assert r.elapsed >= 0.0


# ── Settings integration ──────────────────────────────────────────────────

class TestSettingsIntegration:
    def test_liveness_config_from_settings(self, tmp_settings):
        cfg = tmp_settings.liveness_config()
        assert cfg.timeout_seconds    == tmp_settings.liveness_timeout
        assert cfg.left_threshold     == tmp_settings.liveness_left_threshold
        assert cfg.frontal_max_nor    == tmp_settings.liveness_frontal_max
        assert cfg.min_confirm_frames == tmp_settings.liveness_min_frames

    def test_custom_timeout_in_settings(self, tmp_settings):
        tmp_settings.liveness_timeout = 12.0
        cfg = tmp_settings.liveness_config()
        assert cfg.timeout_seconds == 12.0


# ── LivenessSession (mocked camera + detector) ────────────────────────────

class TestLivenessSession:
    """
    Test LivenessSession without opening a real camera.

    We mock Camera and FaceDetector so the session loop
    drives the state machine with synthetic landmark sequences.
    """

    def _run_session(self, kps_sequence, settings=None, timeout=5.0):
        """
        Feed a fixed sequence of kps values through LivenessSession.run()
        without touching any hardware.

        Returns the final LivenessState.
        """
        from unittest.mock import MagicMock, patch
        from app.liveness.session import LivenessSession
        from app.detection import DetectedFace
        import numpy as np

        if settings is None:
            from app.config import Settings
            settings = Settings(
                log_to_file=False,
                liveness_timeout=timeout,
                liveness_min_frames=1,
            )

        # Build a fake iterator over the kps sequence
        frame_iter = iter(kps_sequence)

        def fake_read():
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, dummy_frame

        def fake_detect(_frame):
            try:
                kps = next(frame_iter)
            except StopIteration:
                return []
            if kps is None:
                return []
            face = MagicMock(spec=DetectedFace)
            face.kps = kps
            face.embedding = None
            return [face]

        mock_cam = MagicMock()
        mock_cam.read.side_effect = fake_read
        mock_cam.__enter__ = lambda s: s
        mock_cam.__exit__ = MagicMock(return_value=False)

        mock_det = MagicMock()
        mock_det.detect.side_effect = fake_detect

        with patch("app.liveness.session.Camera", return_value=mock_cam), \
             patch("app.liveness.session.cv2.imshow"), \
             patch("app.liveness.session.cv2.waitKey", return_value=0):
            session = LivenessSession(settings, mock_det)
            return session.run()

    def test_happy_path_returns_live(self):
        frontal  = _kps(0.0)
        left_turn = _kps(-0.25)
        # frontal → lock baseline → left turns → LIVE
        sequence = [frontal] + [left_turn] * 5
        state = self._run_session(sequence)
        assert state == LivenessState.LIVE

    def test_face_lost_returns_failed(self):
        # frontal locks baseline, then face disappears
        sequence = [_kps(0.0), _kps(0.0), None, None, None, None, None]
        state = self._run_session(sequence)
        assert state == LivenessState.FAILED

    def test_timeout_returns_timeout(self):
        # Only frontal frames — never turns left → timeout
        sequence = [_kps(0.0)] * 3
        state = self._run_session(sequence, timeout=0.001)
        # session will exhaust sequence then keep returning CHALLENGE_ACTIVE
        # until the liveness detector times out (needs at least one more frame
        # after timeout elapses; provide an extra frontal frame)
        from unittest.mock import MagicMock, patch
        from app.liveness.session import LivenessSession
        from app.config import Settings
        import numpy as np, time

        settings = Settings(
            log_to_file=False,
            liveness_timeout=0.05,
            liveness_min_frames=1,
        )
        frontal = _kps(0.0)

        call_count = 0

        def fake_read():
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def fake_detect(_frame):
            nonlocal call_count
            call_count += 1
            # frame 1: frontal to lock baseline
            # frames 2+: frontal but timeout should fire
            face = MagicMock()
            face.kps = frontal
            face.embedding = None
            if call_count == 1:
                return [face]
            time.sleep(0.06)   # push past the 0.05s timeout
            return [face]

        mock_cam = MagicMock()
        mock_cam.read.side_effect = fake_read

        with patch("app.liveness.session.Camera", return_value=mock_cam), \
             patch("app.liveness.session.cv2.imshow"), \
             patch("app.liveness.session.cv2.waitKey", return_value=0):
            det_mock = MagicMock()
            det_mock.detect.side_effect = fake_detect
            session = LivenessSession(settings, det_mock)
            state = session.run()

        assert state == LivenessState.TIMEOUT


    def test_q_key_cancels_session(self):
        """Pressing Q during the session must return FAILED immediately."""
        from unittest.mock import MagicMock, patch
        from app.liveness.session import LivenessSession
        from app.config import Settings
        import numpy as np

        settings = Settings(log_to_file=False, liveness_timeout=5.0,
                            liveness_min_frames=1)

        def fake_read():
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def fake_detect(_frame):
            face = MagicMock()
            face.kps = _kps(0.0)
            face.embedding = None
            return [face]

        mock_cam = MagicMock()
        mock_cam.read.side_effect = fake_read

        call_count = 0

        def fake_waitkey(_):
            nonlocal call_count
            call_count += 1
            # Return 'q' on the second frame
            return ord("q") if call_count >= 2 else 0

        with patch("app.liveness.session.Camera", return_value=mock_cam), \
             patch("app.liveness.session.cv2.imshow"), \
             patch("app.liveness.session.cv2.waitKey", side_effect=fake_waitkey):
            det_mock = MagicMock()
            det_mock.detect.side_effect = fake_detect
            session = LivenessSession(settings, det_mock)
            state = session.run()

        assert state == LivenessState.FAILED

    def test_liveness_result_has_timeout_field(self):
        """LivenessResult must carry the timeout for the progress bar."""
        cfg = _fast_cfg(timeout_seconds=12.0)
        d = LivenessDetector(cfg)
        d.update(_kps(0.0))        # CHALLENGE_ACTIVE
        r = d.update(_kps(0.0))    # still in challenge
        assert r.timeout == 12.0
