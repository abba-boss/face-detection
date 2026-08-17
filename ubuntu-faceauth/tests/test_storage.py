"""
Tests for FaceStore — no camera or AI model required.
"""

import numpy as np
import pytest

from app.config import Settings
from app.storage import FaceStore


@pytest.fixture
def store(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    return FaceStore(s)


def _random_embedding(dim=512) -> np.ndarray:
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_save_and_load(store):
    emb = _random_embedding()
    store.save("alice", emb)
    user = store.load("alice")
    assert user is not None
    assert user.username == "alice"
    assert user.embedding.shape == (512,)


def test_embedding_is_normalised_on_save(store):
    raw = _random_embedding() * 5.0   # not unit-length
    store.save("bob", raw)
    user = store.load("bob")
    norm = float(np.linalg.norm(user.embedding))
    assert abs(norm - 1.0) < 1e-5


def test_load_nonexistent_returns_none(store):
    assert store.load("ghost") is None


def test_list_users(store):
    store.save("alice", _random_embedding())
    store.save("bob",   _random_embedding())
    users = store.list_users()
    assert "alice" in users
    assert "bob" in users


def test_delete_user(store):
    store.save("alice", _random_embedding())
    assert store.is_enrolled("alice")
    result = store.delete("alice")
    assert result is True
    assert not store.is_enrolled("alice")


def test_delete_nonexistent_returns_false(store):
    assert store.delete("nobody") is False


def test_load_all(store):
    store.save("u1", _random_embedding())
    store.save("u2", _random_embedding())
    users = store.load_all()
    usernames = {u.username for u in users}
    assert "u1" in usernames
    assert "u2" in usernames


def test_username_sanitised(store):
    """Path-traversal characters must be rejected/sanitised."""
    emb = _random_embedding()
    store.save("../evil", emb)
    users = store.list_users()
    assert "../evil" not in users   # sanitised to something safe


def test_invalid_username_raises(store):
    with pytest.raises(ValueError):
        store.save("", _random_embedding())


# ── New tests for version-2 features ─────────────────────────────────────

def test_enrolled_at_is_stored(store):
    """saved embedding must include a non-empty enrolled_at timestamp."""
    store.save("timestamped", _random_embedding())
    user = store.load("timestamped")
    assert user is not None
    assert user.enrolled_at != ""
    # Should look like an ISO-8601 UTC string
    assert "T" in user.enrolled_at
    assert user.enrolled_at.endswith("Z")


def test_enrollment_info_returns_metadata(store):
    store.save("alice", _random_embedding())
    info = store.enrollment_info("alice")
    assert info is not None
    assert info["username"] == "alice"
    assert "enrolled_at" in info
    # Does NOT expose the embedding vector
    assert "embedding" not in info


def test_enrollment_info_missing_user_returns_none(store):
    assert store.enrollment_info("nobody") is None


def test_file_permissions_are_owner_only(store, tmp_path):
    """Embedding file must be set to mode 0o600."""
    import stat as stat_module
    s = Settings(data_dir=tmp_path / "perm_data")
    fs = FaceStore(s)
    path = fs.save("secure_user", _random_embedding())
    mode = path.stat().st_mode
    # Only owner read (0o400) and owner write (0o200) should be set
    assert mode & stat_module.S_IRUSR, "owner read bit must be set"
    assert mode & stat_module.S_IWUSR, "owner write bit must be set"
    assert not (mode & stat_module.S_IRGRP), "group read must NOT be set"
    assert not (mode & stat_module.S_IROTH), "other read must NOT be set"


def test_load_sets_enrolled_at_empty_for_v1_files(tmp_path):
    """
    Version 1 files (no enrolled_at key) must load without error.
    enrolled_at defaults to empty string for backward compatibility.
    """
    from app.config import Settings
    import numpy as np

    s = Settings(data_dir=tmp_path / "v1_data")
    emb_dir = s.embeddings_dir
    rng = np.random.default_rng(99)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)

    # Write a v1 file manually (no enrolled_at, version=1)
    path = emb_dir / "legacy.npz"
    np.savez(path, embedding=v, username=np.array("legacy"),
             version=np.array(1))

    fs = FaceStore(s)
    user = fs.load("legacy")
    assert user is not None
    assert user.username == "legacy"
    assert user.enrolled_at == ""   # graceful default
