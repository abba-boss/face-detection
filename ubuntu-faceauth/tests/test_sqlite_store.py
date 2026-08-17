"""
Tests for SQLiteFaceStore — identical contract to FaceStore.

All tests use a tmp_path database; no camera or model required.
The test suite mirrors test_storage.py so the two backends stay in lockstep.
Additional tests cover SQLite-specific behaviour (binary blob, WAL mode,
permissions, migration path).
"""

from __future__ import annotations

import stat as stat_module
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.storage.sqlite_store import SQLiteFaceStore


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", log_to_file=False)


@pytest.fixture
def store(settings) -> SQLiteFaceStore:
    return SQLiteFaceStore(settings)


def _rng_emb(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ═══════════════════════════════════════════════════════════════════════════
# Core CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestCRUD:

    def test_save_and_load(self, store):
        emb = _rng_emb(0)
        store.save("alice", emb)
        user = store.load("alice")
        assert user is not None
        assert user.username == "alice"
        assert user.embedding.shape == (512,)

    def test_embedding_is_normalised_on_save(self, store):
        raw = _rng_emb(1) * 7.0    # not unit-length
        store.save("bob", raw)
        user = store.load("bob")
        norm = float(np.linalg.norm(user.embedding))
        assert abs(norm - 1.0) < 1e-5

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("ghost") is None

    def test_list_users(self, store):
        store.save("alice", _rng_emb(2))
        store.save("bob",   _rng_emb(3))
        users = store.list_users()
        assert "alice" in users
        assert "bob"   in users

    def test_list_users_sorted(self, store):
        store.save("zara", _rng_emb(4))
        store.save("anna", _rng_emb(5))
        users = store.list_users()
        assert users == sorted(users)

    def test_delete_user(self, store):
        store.save("alice", _rng_emb(6))
        assert store.is_enrolled("alice")
        result = store.delete("alice")
        assert result is True
        assert not store.is_enrolled("alice")

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("nobody") is False

    def test_load_all(self, store):
        store.save("u1", _rng_emb(7))
        store.save("u2", _rng_emb(8))
        users = store.load_all()
        usernames = {u.username for u in users}
        assert "u1" in usernames
        assert "u2" in usernames

    def test_overwrite_updates_record(self, store):
        emb1 = _rng_emb(9)
        emb2 = _rng_emb(10)
        store.save("overlap", emb1)
        store.save("overlap", emb2)
        user = store.load("overlap")
        emb2_norm = emb2 / np.linalg.norm(emb2)
        assert float(np.dot(emb2_norm, user.embedding)) > 0.999

    def test_overwrite_invalidates_cache(self, store):
        emb1 = _rng_emb(11)
        emb2 = _rng_emb(12)
        store.save("cached", emb1)
        _ = store.load("cached")            # populate cache
        store.save("cached", emb2)
        user = store.load("cached")         # must reflect emb2
        emb2_norm = emb2 / np.linalg.norm(emb2)
        assert float(np.dot(emb2_norm, user.embedding)) > 0.999


# ═══════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadata:

    def test_enrolled_at_is_stored(self, store):
        store.save("ts_user", _rng_emb(13))
        user = store.load("ts_user")
        assert user.enrolled_at != ""
        assert "T" in user.enrolled_at
        assert user.enrolled_at.endswith("Z")

    def test_enrollment_info_returns_metadata(self, store):
        store.save("alice", _rng_emb(14))
        info = store.enrollment_info("alice")
        assert info is not None
        assert info["username"] == "alice"
        assert "enrolled_at" in info
        assert "embedding" not in info   # never expose the vector

    def test_enrollment_info_missing_user_returns_none(self, store):
        assert store.enrollment_info("nobody") is None


# ═══════════════════════════════════════════════════════════════════════════
# Security
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurity:

    def test_db_permissions_are_owner_only(self, settings):
        store = SQLiteFaceStore(settings)
        store.save("sec_user", _rng_emb(15))
        mode = store._db_path.stat().st_mode
        assert mode & stat_module.S_IRUSR,      "owner read must be set"
        assert mode & stat_module.S_IWUSR,      "owner write must be set"
        assert not (mode & stat_module.S_IRGRP), "group read must NOT be set"
        assert not (mode & stat_module.S_IROTH), "other read must NOT be set"

    def test_username_sanitised(self, store):
        store.save("../evil", _rng_emb(16))
        users = store.list_users()
        assert "../evil" not in users

    def test_invalid_username_raises(self, store):
        with pytest.raises(ValueError):
            store.save("", _rng_emb(17))


# ═══════════════════════════════════════════════════════════════════════════
# Round-trip serialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestRoundTrip:

    def test_round_trip_preserves_direction(self, store):
        original = _rng_emb(18)
        store.save("rt_user", original)
        user = store.load("rt_user")
        o_norm = original / np.linalg.norm(original)
        sim = float(np.dot(o_norm, user.embedding))
        assert abs(sim - 1.0) < 1e-5

    def test_round_trip_dtype_is_float32(self, store):
        store.save("dtype_user", _rng_emb(19))
        user = store.load("dtype_user")
        assert user.embedding.dtype == np.float32

    def test_round_trip_shape(self, store):
        store.save("shape_user", _rng_emb(20))
        user = store.load("shape_user")
        assert user.embedding.shape == (512,)


# ═══════════════════════════════════════════════════════════════════════════
# SQLite-specific
# ═══════════════════════════════════════════════════════════════════════════

class TestSQLiteSpecific:

    def test_db_file_created(self, settings):
        store = SQLiteFaceStore(settings)
        assert store._db_path.exists()

    def test_db_file_in_data_dir(self, settings):
        store = SQLiteFaceStore(settings)
        assert store._db_path.parent == settings.data_dir

    def test_multiple_stores_share_same_db(self, settings):
        """Two store instances pointing at the same settings see the same data."""
        s1 = SQLiteFaceStore(settings)
        s2 = SQLiteFaceStore(settings)
        s1.save("shared_user", _rng_emb(21))
        assert s2.is_enrolled("shared_user")

    def test_wal_mode_enabled(self, settings):
        """WAL journal mode should be active for safe concurrent reads."""
        import sqlite3
        store = SQLiteFaceStore(settings)
        store.save("wal_user", _rng_emb(22))
        conn = sqlite3.connect(str(store._db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_empty_store_load_all_returns_empty(self, store):
        assert store.load_all() == []

    def test_is_enrolled_false_for_unknown(self, store):
        assert not store.is_enrolled("nobody")

    def test_is_enrolled_true_after_save(self, store):
        store.save("known", _rng_emb(23))
        assert store.is_enrolled("known")

    def test_is_enrolled_false_after_delete(self, store):
        store.save("gone", _rng_emb(24))
        store.delete("gone")
        assert not store.is_enrolled("gone")


# ═══════════════════════════════════════════════════════════════════════════
# get_store factory
# ═══════════════════════════════════════════════════════════════════════════

class TestGetStoreFactory:

    def test_sqlite_backend_returns_sqlite_store(self, settings):
        from app.storage import get_store
        settings.storage_backend = "sqlite"
        store = get_store(settings)
        assert isinstance(store, SQLiteFaceStore)

    def test_npz_backend_returns_face_store(self, settings):
        from app.storage import get_store
        from app.storage.face_store import FaceStore
        settings.storage_backend = "npz"
        store = get_store(settings)
        assert isinstance(store, FaceStore)

    def test_default_backend_is_sqlite(self, settings):
        from app.storage import get_store
        # settings.storage_backend defaults to "sqlite"
        store = get_store(settings)
        assert isinstance(store, SQLiteFaceStore)
