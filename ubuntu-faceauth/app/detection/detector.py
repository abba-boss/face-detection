"""
Face detection using InsightFace's bundled RetinaFace detector.

Responsibilities:
  - Initialise the InsightFace app once.
  - Detect faces in a BGR frame.
  - Filter out faces that are too small or have low confidence.
  - Return structured DetectedFace objects — no raw InsightFace internals
    leak beyond this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from app.config import Settings
from app.security import get_logger


@dataclass
class DetectedFace:
    """Everything we know about one detected face in a frame."""
    bbox: np.ndarray          # [x1, y1, x2, y2]  float32
    confidence: float
    embedding: Optional[np.ndarray] = None   # ArcFace 512-d vector
    kps: Optional[np.ndarray] = None         # 5-point landmarks
    blur_score: float = 0.0                  # Laplacian variance — higher = sharper

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_sharp(self) -> bool:
        """
        Convenience property using the built-in threshold of 40.0.

        In the enrollment loop, compare against
        settings.enrollment_blur_threshold directly for the
        calibrated threshold.  This property is kept for quick
        sanity checks and unit tests.
        """
        return self.blur_score > 40.0


class FaceDetector:
    """
    Wraps InsightFace FaceAnalysis for detection + embedding in one pass.

    InsightFace's FaceAnalysis runs detection and recognition together,
    so we initialise it here and expose the results cleanly.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._app = None
        self._log = get_logger(
            __name__,
            log_file=settings.log_file if settings.log_to_file else None,
            level=settings.log_level,
        )

    def load(self) -> None:
        """Download (first run) and initialise the InsightFace model."""
        import insightface
        from insightface.app import FaceAnalysis

        # One-time patch: insightface calls the deprecated
        # SimilarityTransform.estimate() from scikit-image >= 0.26.
        # Redirect it to the private _estimate() method which has
        # the same signature and no deprecation wrapper.
        _patch_skimage_transform()

        self._log.info(
            "Initialising InsightFace model '%s' (CPU) — "
            "first run downloads ~14 MB …",
            self._settings.insightface_model_name,
        )
        try:
            self._app = FaceAnalysis(
                name=self._settings.insightface_model_name,
                root=str(self._settings.insightface_root),
                providers=["CPUExecutionProvider"],
            )
            # ctx_id=0  → CPU;  det_size controls detector input resolution
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._log.info("InsightFace model ready")
        except Exception as exc:
            self._log.error("Model initialisation failed: %s", exc)
            raise

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Run detection + embedding on *frame* (BGR, uint8).

        Returns a list of DetectedFace objects that pass the minimum
        size filter.  The list is empty when nothing is found.

        Note: _patch_skimage_transform() is called in load() to silence
        the scikit-image FutureWarning before it is ever emitted.
        """
        if self._app is None:
            raise RuntimeError("FaceDetector.load() must be called first")

        faces_raw = self._app.get(frame)
        results: List[DetectedFace] = []

        for f in faces_raw:
            det_score = float(getattr(f, "det_score", 1.0))
            if det_score < self._settings.detection_threshold:
                continue

            bbox = np.array(f.bbox, dtype=np.float32)
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])

            if w < self._settings.min_face_size or h < self._settings.min_face_size:
                self._log.debug(
                    "Skipping small face %.0fx%.0f (min %d)",
                    w, h, self._settings.min_face_size,
                )
                continue

            embedding = None
            if hasattr(f, "embedding") and f.embedding is not None:
                raw_emb = np.array(f.embedding, dtype=np.float32)
                # L2-normalise here so every consumer (recogniser, store)
                # receives a unit vector.  Cosine similarity then equals
                # a plain dot product.
                norm = np.linalg.norm(raw_emb)
                embedding = raw_emb / norm if norm > 1e-10 else raw_emb

            kps = None
            if hasattr(f, "kps") and f.kps is not None:
                kps = np.array(f.kps, dtype=np.float32)

            blur = _blur_score(frame, bbox)

            results.append(DetectedFace(
                bbox=bbox,
                confidence=det_score,
                embedding=embedding,
                kps=kps,
                blur_score=blur,
            ))

        return results


# ── Helpers ──────────────────────────────────────────────────────────────

def _blur_score(frame: np.ndarray, bbox: np.ndarray) -> float:
    """
    Return the Laplacian variance of the face crop as a sharpness proxy.

    A value below ~40 indicates a blurry or low-quality face region.
    Higher is sharper.
    """
    x1, y1, x2, y2 = (int(max(0, v)) for v in bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_skimage_patched = False


def _patch_skimage_transform() -> None:
    """
    One-time monkey-patch: redirect SimilarityTransform.estimate() to
    SimilarityTransform._estimate() so insightface stops triggering the
    scikit-image >= 0.26 FutureWarning.

    The deprecated `estimate` method is just a thin wrapper around
    `_estimate` with no behavioural difference — it only adds the
    deprecation warning via a decorator.  We replace it with the
    private method directly on the class so no warning is ever issued.

    This patch is applied once at model-load time and is idempotent.
    """
    global _skimage_patched
    if _skimage_patched:
        return
    try:
        from skimage.transform import SimilarityTransform
        if hasattr(SimilarityTransform, "_estimate"):
            SimilarityTransform.estimate = SimilarityTransform._estimate
            _skimage_patched = True
    except Exception:
        # If skimage is not installed or the API changes, fail silently —
        # the warning suppression in detect() is the fallback.
        pass
