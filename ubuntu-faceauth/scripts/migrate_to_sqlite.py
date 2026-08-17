#!/usr/bin/env python3
"""
Migrate existing .npz face embeddings to the SQLite store.

Usage:
    conda activate face-detection
    cd ~/Desktop/face-detection/ubuntu-faceauth
    python scripts/migrate_to_sqlite.py

What it does:
    1. Scans data/embeddings/*.npz for enrolled users.
    2. Loads each embedding using the legacy FaceStore.
    3. Writes it to the SQLite database via SQLiteFaceStore.
    4. Leaves the original .npz files in place (safe to delete manually).

Run it once after upgrading to the SQLite backend.
Already-migrated users are silently overwritten (idempotent).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.storage.face_store import FaceStore
from app.storage.sqlite_store import SQLiteFaceStore


def migrate() -> None:
    settings = Settings()

    npz_store    = FaceStore(settings)
    sqlite_store = SQLiteFaceStore(settings)

    users = npz_store.list_users()
    if not users:
        print("No .npz enrollments found — nothing to migrate.")
        return

    print(f"Found {len(users)} enrolled user(s): {', '.join(users)}")
    migrated = 0
    failed   = 0

    for username in users:
        user = npz_store.load(username)
        if user is None:
            print(f"  [SKIP] {username} — could not load .npz")
            failed += 1
            continue
        try:
            sqlite_store.save(username, user.embedding)
            # Preserve the original enrolled_at timestamp
            # by updating the record directly
            from datetime import datetime, timezone
            if user.enrolled_at:
                import sqlite3
                conn = sqlite3.connect(str(sqlite_store._db_path))
                conn.execute(
                    "UPDATE enrollments SET enrolled_at = ? WHERE username = ?",
                    (user.enrolled_at, username),
                )
                conn.commit()
                conn.close()

            print(f"  [OK]   {username}  (enrolled: {user.enrolled_at or 'unknown'})")
            migrated += 1
        except Exception as exc:
            print(f"  [FAIL] {username} — {exc}")
            failed += 1

    print(f"\nMigration complete: {migrated} migrated, {failed} failed.")
    print(f"Database: {sqlite_store._db_path}")
    if migrated:
        print("\nOriginal .npz files are unchanged — delete manually when ready:")
        for username in users:
            npz_path = settings.embeddings_dir / f"{username}.npz"
            if npz_path.exists():
                print(f"  rm {npz_path}")


if __name__ == "__main__":
    migrate()
