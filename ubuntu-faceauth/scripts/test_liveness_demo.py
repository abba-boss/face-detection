#!/usr/bin/env python3
"""
scripts/test_liveness_demo.py — Liveness state machine demo (no camera).

Simulates a full challenge-response session using synthetic landmark data.
Run with:
    conda activate face-detection
    python scripts/test_liveness_demo.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from app.liveness.detector import (
    LivenessDetector,
    LivenessConfig,
    LivenessState,
)

GREEN  = "\033[32m"
RED    = "\033[31m"
CYAN   = "\033[36m"
RESET  = "\033[0m"
PASS_S = f"{GREEN}PASS{RESET}"
FAIL_S = f"{RED}FAIL{RESET}"


# ── landmark helpers ─────────────────────────────────────────────────────

def _kps(nor: float, eye_span: float = 100.0) -> np.ndarray:
    cx = 320.0
    lx, rx = cx - eye_span / 2, cx + eye_span / 2
    nx = (lx + rx) / 2 + nor * eye_span
    return np.array([[lx,200],[rx,200],[nx,240],
                     [cx-30,280],[cx+30,280]], dtype=np.float32)

def _frontal():     return _kps(0.0)
def _left(a=0.25):  return _kps(-a)
def _right(a=0.25): return _kps(+a)


# ── scenario runner ──────────────────────────────────────────────────────

def run(name: str, expected: LivenessState, det: LivenessDetector,
        frames) -> bool:
    print(f"\n  {CYAN}Scenario:{RESET} {name}")
    result = None
    for i, kps in enumerate(frames):
        result = det.update(kps)
        nor_s  = f"{result.nor:+.3f}" if result.nor is not None else "  none"
        print(f"    frame {i+1:02d}  NOR={nor_s}  "
              f"state={result.state.name:<18}  msg={result.message}")
        if result.state in (LivenessState.LIVE, LivenessState.FAILED,
                            LivenessState.TIMEOUT):
            break
    ok = (result.state == expected)
    print(f"  → {PASS_S if ok else FAIL_S}  "
          f"final={result.state.name}  (expected={expected.name})")
    return ok


def main() -> int:
    print("\nUbuntu FaceAuth — Liveness State Machine Demo")
    print("=" * 50)
    results = []

    cfg = LivenessConfig(timeout_seconds=5.0, left_threshold=0.18,
                         frontal_max_nor=0.15, min_confirm_frames=2)

    # 1 — Happy path: frontal then left turn → LIVE
    det = LivenessDetector(cfg)
    results.append(run(
        "Happy path (frontal → left turn → LIVE)",
        LivenessState.LIVE,
        det,
        [_frontal(), _left(0.25), _left(0.25), _left(0.25)],
    ))

    # 2 — Timeout: challenge starts (frontal) then time jumps past limit
    det2 = LivenessDetector(cfg)
    # Lock baseline at t=0
    det2.update(_frontal())
    # Simulate clock jumping 10 s into the future
    future_t = __import__("time").monotonic() + 10.0
    with patch("app.liveness.detector.time.monotonic", return_value=future_t):
        r = det2.update(_frontal())
    ok2 = (r.state == LivenessState.TIMEOUT)
    print(f"\n  {CYAN}Scenario:{RESET} Timeout (clock jump past limit)")
    print(f"    final={r.state.name}  msg={r.message}")
    print(f"  → {PASS_S if ok2 else FAIL_S}  "
          f"final={r.state.name}  (expected=TIMEOUT)")
    results.append(ok2)

    # 3 — Face lost during challenge → FAILED
    det3 = LivenessDetector(cfg)
    results.append(run(
        "Face lost during challenge → FAILED",
        LivenessState.FAILED,
        det3,
        [_frontal(), _frontal(), None, _frontal()],
    ))

    # 4 — Wrong direction: right turn then timeout
    det4 = LivenessDetector(cfg)
    det4.update(_frontal())          # lock baseline
    future_t2 = __import__("time").monotonic() + 10.0
    with patch("app.liveness.detector.time.monotonic", return_value=future_t2):
        r4 = det4.update(_right(0.3))
    ok4 = (r4.state == LivenessState.TIMEOUT)
    print(f"\n  {CYAN}Scenario:{RESET} Wrong direction (right turn) → TIMEOUT")
    print(f"    final={r4.state.name}  msg={r4.message}")
    print(f"  → {PASS_S if ok4 else FAIL_S}  "
          f"final={r4.state.name}  (expected=TIMEOUT)")
    results.append(ok4)

    # 5 — Reset: LIVE → reset → fresh WAITING → LIVE again
    det5 = LivenessDetector(LivenessConfig(min_confirm_frames=1,
                                           timeout_seconds=5.0,
                                           left_threshold=0.18,
                                           frontal_max_nor=0.15))
    det5.update(_frontal())
    r5a = det5.update(_left(0.25))
    det5.reset()
    r5b = det5.update(None)
    det5.update(_frontal())
    r5c = det5.update(_left(0.25))
    ok5 = (r5a.state == LivenessState.LIVE and
           r5b.state == LivenessState.WAITING and
           r5c.state == LivenessState.LIVE)
    print(f"\n  {CYAN}Scenario:{RESET} Reset from LIVE → WAITING → LIVE again")
    print(f"    before_reset={r5a.state.name}  "
          f"after_reset={r5b.state.name}  "
          f"after_second_challenge={r5c.state.name}")
    print(f"  → {PASS_S if ok5 else FAIL_S}")
    results.append(ok5)

    passed, total = sum(results), len(results)
    colour = GREEN if passed == total else RED
    print(f"\n{'='*50}")
    print(f"{colour}{passed}/{total} scenarios passed{RESET}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
