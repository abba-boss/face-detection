"""
Tests for embedding serialization/deserialization round-trips.
"""

import numpy as np
import pytest

from app.config import Settings
from app.storage import FaceStore


@pytest.fixture
def store(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    return FaceStore(s)


def _random_emb(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(512).astype(np.float32)


def test_round_trip_preserves_direction(store):
    original = _random_emb(10)
    store.save("test_user", original)
    loaded = store.load("test_user")
    assert loaded is not None

    # Cosine similarity of saved vs loaded should be 1.0 (same direction)
    o_norm = original / np.linalg.norm(original)
    l_norm = loaded.embedding
    similarity = float(np.dot(o_norm, l_norm))
    assert abs(similarity - 1.0) < 1e-5


def test_round_trip_dtype_is_float32(store):
    store.save("dtype_user", _random_emb(20))
    user = store.load("dtype_user")
    assert user.embedding.dtype == np.float32


def test_round_trip_shape(store):
    store.save("shape_user", _random_emb(30))
    user = store.load("shape_user")
    assert user.embedding.shape == (512,)


def test_multiple_saves_overwrite(store):
    emb1 = _random_emb(1)
    emb2 = _random_emb(2)
    store.save("overlap", emb1)
    store.save("overlap", emb2)

    user = store.load("overlap")
    # Should reflect emb2
    emb2_norm = emb2 / np.linalg.norm(emb2)
    similarity = float(np.dot(emb2_norm, user.embedding))
    assert similarity > 0.99


def test_cache_invalidated_on_overwrite(store):
    """After overwriting an embedding, the old cached version must not be served."""
    emb1 = _random_emb(40)
    emb2 = _random_emb(41)

    store.save("cached_user", emb1)
    _ = store.load("cached_user")   # populate cache

    store.save("cached_user", emb2)
    user = store.load("cached_user")

    emb2_norm = emb2 / np.linalg.norm(emb2)
    sim = float(np.dot(emb2_norm, user.embedding))
    assert sim > 0.99
