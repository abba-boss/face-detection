"""
Tests for the liveness UI drawing helpers.

No camera required — all tests operate on synthetic numpy frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.liveness.detector import (
    LivenessState,
    LivenessResult,
    _DEFAULT_TIMEOUT,
)
from app.liveness.drawing import draw_liveness_overlay


# ── Helpers ───────────────────────────────────────────────────────────────

def _blank(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _result(state: LivenessState, nor: float | None = None,
            elapsed: float = 0.0, timeout: float = _DEFAULT_TIMEOUT,
            message: str = "Test message",
            confirm_count: int = 0) -> LivenessResult:
    return LivenessResult(
        state         = state,
        message       = message,
        nor           = nor,
        baseline_nor  = 0.0 if state != LivenessState.WAITING else None,
        elapsed       = elapsed,
        confirm_count = confirm_count,
        timeout       = timeout,
    )


# ── draw_liveness_overlay — smoke tests ──────────────────────────────────

class TestDrawLivenessOverlay:
    """
    These tests verify that draw_liveness_overlay:
      - runs without raising for every LivenessState
      - modifies the frame (i.e. actually draws something)
      - respects the timeout field for the progress bar
    """

    @pytest.mark.parametrize("state", list(LivenessState))
    def test_no_exception_for_any_state(self, state):
        frame = _blank()
        result = _result(state, nor=0.05 if state != LivenessState.WAITING else None)
        draw_liveness_overlay(frame, result)   # must not raise

    @pytest.mark.parametrize("state", list(LivenessState))
    def test_frame_is_modified(self, state):
        """The overlay must change at least some pixels."""
        original = _blank()
        frame = original.copy()
        result = _result(state, nor=0.0)
        draw_liveness_overlay(frame, result)
        assert not np.array_equal(frame, original), (
            f"Frame not modified for state={state.name}"
        )

    def test_nor_readout_only_when_nor_present(self):
        """
        When nor is None the NOR readout is skipped; when nor is provided
        the frame should differ from the no-nor case (an extra text line drawn).
        """
        frame_with_nor    = _blank()
        frame_without_nor = _blank()

        draw_liveness_overlay(frame_with_nor,    _result(LivenessState.WAITING, nor=0.05))
        draw_liveness_overlay(frame_without_nor, _result(LivenessState.WAITING, nor=None))

        # The two frames should differ (NOR text is present in one)
        assert not np.array_equal(frame_with_nor, frame_without_nor)

    def test_progress_bar_drawn_during_challenge(self):
        """
        CHALLENGE_ACTIVE should draw a progress bar region that
        differs from WAITING (no bar).
        """
        frame_challenge = _blank()
        frame_waiting   = _blank()

        draw_liveness_overlay(frame_challenge,
                              _result(LivenessState.CHALLENGE_ACTIVE,
                                      nor=0.0, elapsed=1.0, timeout=8.0))
        draw_liveness_overlay(frame_waiting,
                              _result(LivenessState.WAITING))

        # Frames must differ (challenge has a progress bar, waiting does not)
        assert not np.array_equal(frame_challenge, frame_waiting)

    def test_progress_bar_uses_result_timeout_not_default(self):
        """
        A custom timeout must produce a different progress bar than the
        default timeout at the same elapsed time.
        """
        frame_default = _blank()
        frame_custom  = _blank()

        draw_liveness_overlay(
            frame_default,
            _result(LivenessState.CHALLENGE_ACTIVE,
                    nor=0.0, elapsed=4.0, timeout=8.0),   # 50% elapsed
        )
        draw_liveness_overlay(
            frame_custom,
            _result(LivenessState.CHALLENGE_ACTIVE,
                    nor=0.0, elapsed=4.0, timeout=16.0),  # 25% elapsed
        )
        # Different fraction remaining → different bar width → different pixels
        assert not np.array_equal(frame_default, frame_custom)

    def test_live_state_draws_centred_label(self):
        """LIVE state should draw something near the centre of the frame."""
        frame = _blank()
        draw_liveness_overlay(frame, _result(LivenessState.LIVE))
        h, w = frame.shape[:2]
        centre_region = frame[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
        # Centre region should have non-zero pixels (the large LIVE label)
        assert centre_region.max() > 0

    def test_failed_state_draws_centred_label(self):
        frame = _blank()
        draw_liveness_overlay(frame, _result(LivenessState.FAILED))
        h, w = frame.shape[:2]
        centre_region = frame[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
        assert centre_region.max() > 0

    def test_font_scale_parameter_accepted(self):
        """font_scale kwarg must be accepted without error."""
        frame = _blank()
        draw_liveness_overlay(
            frame,
            _result(LivenessState.CHALLENGE_ACTIVE, nor=0.0),
            font_scale=1.2,
        )   # should not raise

    def test_different_frame_sizes_accepted(self):
        """Overlay should work on non-standard resolutions."""
        for h, w in [(240, 320), (720, 1280), (100, 200)]:
            frame = _blank(h, w)
            draw_liveness_overlay(frame, _result(LivenessState.WAITING))
