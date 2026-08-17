"""
Central configuration for Ubuntu FaceAuth.
All tuneable values live here — nothing is hardcoded in the modules.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # ── Camera ──────────────────────────────────────────────────────────
    camera_device: int = 0          # /dev/video0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ── Face detection ───────────────────────────────────────────────────
    min_face_size: int = 80         # pixels — reject faces smaller than this
    detection_threshold: float = 0.5

    # ── Enrollment ───────────────────────────────────────────────────────
    enrollment_samples: int = 10         # number of valid frames to capture
    enrollment_max_attempts: int = 300   # frames to try before giving up
    enrollment_blur_threshold: float = 20.0
    # Laplacian variance of the face crop — below this = too blurry.
    # Calibrated against a typical USB/built-in webcam (mean ~35, range 24–42).
    # Raise to 40+ for a high-quality camera or strict sharpness requirement.
    # Lower to 10–15 if the camera has soft optics and enrollment keeps stalling.

    # ── Camera ── warmup ─────────────────────────────────────────────────
    camera_warmup_frames: int = 10
    # Discard this many frames after opening the camera.
    # Webcam auto-exposure takes ~0.3 s to stabilise; skipping early frames
    # avoids collecting dark / over-exposed enrollment samples.

    # ── Recognition ──────────────────────────────────────────────────────
    recognition_threshold: float = 0.45   # cosine similarity; above → AUTHORIZED
    # InsightFace ArcFace embeddings use cosine distance.
    # Typical same-person similarity: 0.6–0.9
    # Typical different-person similarity: 0.1–0.35
    # 0.45 is a conservative default — tune upward for stricter matching.

    # ── Storage ──────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve()
                           .parents[3] / "data")

    # Storage backend: "sqlite" (default) or "npz" (legacy)
    # SQLite stores all embeddings in a single encrypted-permissions file.
    # "npz" is kept for backward compatibility and migration source.
    storage_backend: str = "sqlite"

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"         # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True

    # ── InsightFace model ────────────────────────────────────────────────
    insightface_model_name: str = "buffalo_sc"   # lightweight CPU model
    # buffalo_sc  → 500 kB detector + 1 MB ArcFace recogniser  (fast CPU)
    # buffalo_l   → full accuracy, larger download (~500 MB)
    insightface_root: Path = field(
        default_factory=lambda: Path.home() / ".insightface"
    )

    # ── Display ──────────────────────────────────────────────────────────
    show_landmarks: bool = False
    font_scale: float = 0.7

    # ── Liveness (Phase 2A) ──────────────────────────────────────────────
    liveness_timeout: float = 8.0
    # Seconds the user has to complete the head-turn challenge.
    liveness_left_threshold: float = 0.18
    # Nose-offset-ratio drop required to count as a LEFT turn.
    # Increase for stricter detection; decrease if users struggle.
    liveness_frontal_max: float = 0.15
    # |NOR| must be below this for the baseline to lock (face looks frontal).
    liveness_min_frames: int = 3
    # Consecutive frames the turn must hold to be confirmed.

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Restrict data directory to owner only
        try:
            import stat
            self.data_dir.chmod(
                stat.S_IRWXU  # 0o700 — owner rwx, no group/other
            )
        except OSError:
            pass

    @property
    def embeddings_dir(self) -> Path:
        p = self.data_dir / "embeddings"
        p.mkdir(parents=True, exist_ok=True)
        try:
            import stat
            p.chmod(stat.S_IRWXU)
        except OSError:
            pass
        return p

    @property
    def log_file(self) -> Path:
        return self.data_dir / "faceauth.log"

    def liveness_config(self):
        """Return a LivenessConfig built from the current Settings."""
        from app.liveness.detector import LivenessConfig
        return LivenessConfig(
            timeout_seconds    = self.liveness_timeout,
            left_threshold     = self.liveness_left_threshold,
            frontal_max_nor    = self.liveness_frontal_max,
            min_confirm_frames = self.liveness_min_frames,
        )
