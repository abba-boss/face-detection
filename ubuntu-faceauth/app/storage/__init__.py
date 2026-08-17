from .face_store import FaceStore, EnrolledUser
from .sqlite_store import SQLiteFaceStore


def get_store(settings):
    """
    Return the appropriate store for *settings*.

    settings.storage_backend == "sqlite"  → SQLiteFaceStore  (default)
    settings.storage_backend == "npz"     → FaceStore  (legacy)
    """
    if getattr(settings, "storage_backend", "sqlite") == "npz":
        return FaceStore(settings)
    return SQLiteFaceStore(settings)


__all__ = ["FaceStore", "SQLiteFaceStore", "EnrolledUser", "get_store"]
