"""Tests for the Settings configuration object."""

import tempfile
from pathlib import Path

import pytest

from app.config import Settings


def test_default_settings_instantiate():
    s = Settings()
    assert s.camera_device == 0
    assert s.enrollment_samples == 10
    assert 0 < s.recognition_threshold < 1
    assert s.insightface_model_name == "buffalo_sc"


def test_custom_threshold():
    s = Settings(recognition_threshold=0.6)
    assert s.recognition_threshold == 0.6


def test_data_dir_created(tmp_path):
    s = Settings(data_dir=tmp_path / "faceauth_data")
    assert s.data_dir.exists()


def test_embeddings_dir_created(tmp_path):
    s = Settings(data_dir=tmp_path / "faceauth_data")
    emb = s.embeddings_dir
    assert emb.exists()
    assert emb.name == "embeddings"


def test_log_file_path(tmp_path):
    s = Settings(data_dir=tmp_path / "faceauth_data")
    assert s.log_file.name == "faceauth.log"


def test_enrollment_blur_threshold_default():
    s = Settings()
    # Default calibrated for typical webcam (not the old over-strict 40.0)
    assert s.enrollment_blur_threshold == 20.0


def test_camera_warmup_frames_default():
    s = Settings()
    assert s.camera_warmup_frames == 10


def test_enrollment_max_attempts_default():
    s = Settings()
    assert s.enrollment_max_attempts == 300


def test_recognition_threshold_range():
    s = Settings()
    assert 0.0 < s.recognition_threshold < 1.0


def test_settings_override_blur_threshold():
    s = Settings(enrollment_blur_threshold=50.0)
    assert s.enrollment_blur_threshold == 50.0


def test_settings_override_recognition_threshold():
    s = Settings(recognition_threshold=0.60)
    assert s.recognition_threshold == 0.60


# ── Liveness settings ─────────────────────────────────────────────────────

def test_liveness_timeout_default():
    s = Settings()
    assert s.liveness_timeout == 8.0


def test_liveness_left_threshold_default():
    s = Settings()
    assert s.liveness_left_threshold == 0.18


def test_liveness_frontal_max_default():
    s = Settings()
    assert s.liveness_frontal_max == 0.15


def test_liveness_min_frames_default():
    s = Settings()
    assert s.liveness_min_frames == 3


def test_liveness_config_factory(tmp_path):
    from app.liveness.detector import LivenessConfig
    s = Settings(data_dir=tmp_path / "d",
                 liveness_timeout=12.0,
                 liveness_left_threshold=0.25,
                 liveness_frontal_max=0.10,
                 liveness_min_frames=5)
    cfg = s.liveness_config()
    assert isinstance(cfg, LivenessConfig)
    assert cfg.timeout_seconds    == 12.0
    assert cfg.left_threshold     == 0.25
    assert cfg.frontal_max_nor    == 0.10
    assert cfg.min_confirm_frames == 5
