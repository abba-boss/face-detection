#!/usr/bin/env python3
"""
scripts/benchmark.py — Measure InsightFace inference speed on this CPU.

Run with:
    conda activate face-detection
    python scripts/benchmark.py

This gives a realistic idea of how many frames per second the recognition
pipeline can sustain before optimising or switching to a heavier model.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.detection import FaceDetector


_WARMUP_FRAMES  = 5
_MEASURE_FRAMES = 30


def main() -> None:
    print("\nUbuntu FaceAuth — Inference Benchmark")
    print("=" * 42)

    settings = Settings()
    detector = FaceDetector(settings)
    print("Loading model…")
    detector.load()
    print("Model ready.\n")

    # Build a synthetic face-like frame (solid skin tone, no real face)
    # The model will find 0 detections but the pipeline overhead is measured.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some texture so the detector has something to process
    rng = np.random.default_rng(42)
    frame[:] = rng.integers(80, 160, (480, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up ({_WARMUP_FRAMES} frames)…")
    for _ in range(_WARMUP_FRAMES):
        detector.detect(frame)

    # Measure
    print(f"Measuring over {_MEASURE_FRAMES} frames…")
    times = []
    for _ in range(_MEASURE_FRAMES):
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append(time.perf_counter() - t0)

    avg_ms  = 1000 * sum(times) / len(times)
    min_ms  = 1000 * min(times)
    max_ms  = 1000 * max(times)
    fps     = 1000 / avg_ms

    print()
    print(f"  Average latency : {avg_ms:6.1f} ms")
    print(f"  Min latency     : {min_ms:6.1f} ms")
    print(f"  Max latency     : {max_ms:6.1f} ms")
    print(f"  Throughput      : {fps:6.1f} FPS  (pipeline only, no camera I/O)")
    print()

    if fps >= 15:
        print("✓  Fast enough for real-time use (≥ 15 FPS).")
    elif fps >= 8:
        print("⚠  Marginal — usable but may feel sluggish on complex frames.")
    else:
        print("✗  Too slow for real-time — consider buffalo_l only for enrollment.")


if __name__ == "__main__":
    main()
