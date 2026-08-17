"""
Tests for the Recognizer — no camera or real model required.
Uses synthetic embeddings to verify threshold logic and multi-user matching.
"""

import numpy as np
import pytest

from app.config import Settings
from app.recognition import Recognizer
from app.storage import FaceStore


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", recognition_threshold=0.45)


@pytest.fixture
def store(settings):
    return FaceStore(settings)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_no_enrolled_users_returns_unknown(settings, store):
    rec = Recognizer(settings, store)
    query = _unit(np.ones(512, dtype=np.float32))
    result = rec.identify(query)
    assert not result.authorized
    assert result.username is None


def test_identical_embedding_is_authorized(settings, store):
    emb = _unit(np.random.default_rng(1).standard_normal(512).astype(np.float32))
    store.save("alice", emb)

    rec = Recognizer(settings, store)
    result = rec.identify(emb)
    assert result.authorized
    assert result.username == "alice"
    assert result.similarity >= settings.recognition_threshold


def test_orthogonal_embedding_is_unknown(settings, store):
    rng = np.random.default_rng(2)
    emb = _unit(rng.standard_normal(512).astype(np.float32))
    store.save("alice", emb)

    # Orthogonal vector — cosine similarity ≈ 0
    query = _unit(rng.standard_normal(512).astype(np.float32))
    # Make it genuinely dissimilar by zeroing Alice's direction
    query = _unit(query - np.dot(query, emb) * emb)

    rec = Recognizer(settings, store)
    result = rec.identify(query)
    assert not result.authorized


def test_best_match_selected_among_multiple_users(settings, store):
    rng = np.random.default_rng(3)
    alice = _unit(rng.standard_normal(512).astype(np.float32))
    bob   = _unit(rng.standard_normal(512).astype(np.float32))

    store.save("alice", alice)
    store.save("bob",   bob)

    # Query very close to Alice
    noise = rng.standard_normal(512).astype(np.float32) * 0.05
    query = _unit(alice + noise)

    rec = Recognizer(settings, store)
    result = rec.identify(query)
    assert result.authorized
    assert result.username == "alice"


def test_threshold_boundary(settings, store):
    rng = np.random.default_rng(4)
    base = _unit(rng.standard_normal(512).astype(np.float32))
    store.save("user", base)
    rec = Recognizer(settings, store)

    # Craft a query with exact similarity = threshold via linear interpolation
    perp = rng.standard_normal(512).astype(np.float32)
    perp -= np.dot(perp, base) * base
    perp = _unit(perp)

    t = settings.recognition_threshold
    query = _unit(t * base + np.sqrt(max(0, 1 - t**2)) * perp)

    result = rec.identify(query)
    # At exactly the threshold we expect authorisation
    assert result.authorized


def test_similarity_is_non_negative(settings, store):
    emb = _unit(np.ones(512, dtype=np.float32))
    store.save("user", emb)
    rec = Recognizer(settings, store)

    anti = _unit(-np.ones(512, dtype=np.float32))
    result = rec.identify(anti)
    assert result.similarity >= 0.0
