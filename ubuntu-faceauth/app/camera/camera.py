"""
Camera abstraction layer.

Wraps OpenCV VideoCapture so the rest of the application
never touches cv2 directly for capture concerns.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from app.config import Settings
from app.security import get_logger


class Camera:
    """Manages the webcam lifecycle."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._cap: Optional[cv2.VideoCapture] = None
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Public API ───────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the camera device and discard warmup frames.

        Raises RuntimeError on failure.
        """
        device = self._settings.camera_device
        self._log.info("Opening camera device %s", device)

        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            self._log.error("Failed to open camera device %s", device)
            raise RuntimeError(f"Cannot open camera /dev/video{device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._settings.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._settings.camera_fps)

        # Discard warmup frames — webcam auto-exposure needs a moment to settle.
        warmup = self._settings.camera_warmup_frames
        for _ in range(warmup):
            self._cap.read()

        self._log.info(
            "Camera ready — %dx%d @ %d fps  (discarded %d warmup frame(s))",
            self._settings.camera_width,
            self._settings.camera_height,
            self._settings.camera_fps,
            warmup,
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read one frame.

        Returns (True, frame) on success, (False, None) on failure.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            self._log.warning("Camera read failed — no frame returned")
        return ok, frame if ok else None

    def release(self) -> None:
        """Release the camera and destroy any OpenCV windows."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._log.info("Camera released")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # opencv-python-headless has no GUI backend; destroyAllWindows
            # raises cv2.error in headless environments — safe to ignore.
            pass

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
