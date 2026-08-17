"""
Shared pytest fixtures and path setup.

Adds the project root to sys.path so tests can do `import app.*`
without installing the package.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure `ubuntu-faceauth/` is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.storage import FaceStore
from app.security.logger import _loggers   # noqa: F401  (for reset fixture)


# ── Public helper ─────────────────────────────────────────────────────────

def make_unit(seed: int, dim: int = 512) -> np.ndarray:
    """Return a deterministic unit-length float32 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Shared fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_settings(tmp_path):
    """Settings pointing at a temp directory; file logging disabled."""
    return Settings(data_dir=tmp_path / "faceauth_data", log_to_file=False)


@pytest.fixture
def tmp_store(tmp_settings):
    """FaceStore backed by the temp settings."""
    return FaceStore(tmp_settings)


@pytest.fixture
def unit_embedding():
    """A reproducible unit-length 512-d embedding (seed 0)."""
    return make_unit(0)


@pytest.fixture(autouse=True)
def reset_logger_cache():
    """
    Clear the logger singleton cache between tests.

    Without this, a test that creates logger 'foo' without a file
    handler will poison the cache for a subsequent test that wants
    'foo' WITH a file handler.
    """
    _loggers.clear()
    yield
    _loggers.clear()
