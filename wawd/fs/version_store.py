"""VersionStore: file version history and snapshot management."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import aiosqlite

from wawd.fs.blob_store import BlobStore

log = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    blob_hash TEXT REFERENCES blobs(hash),
    operation TEXT NOT NULL CHECK(operation IN ('create', 'modify', 'delete')),
    agent_id TEXT,
    session_id TEXT,
    timestamp REAL NOT NULL DEFAULT (unixepoch('subsec')),
    intent TEXT,
    parent_version_id INTEGER REFERENCES versions(id)
);

CREATE INDEX IF NOT EXISTS idx_versions_path ON versions(path, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_versions_session ON versions(session_id);
CREATE INDEX IF NOT EXISTS idx_versions_agent ON versions(agent_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON versions(timestamp DESC);

CREATE TABLE IF NOT EXISTS snapshots (
    name TEXT PRIMARY KEY,
    description TEXT,
    timestamp REAL NOT NULL DEFAULT (unixepoch('subsec')),
    created_by TEXT
);

CREATE TABLE IF NOT EXISTS snapshot_files (
    snapshot_name TEXT REFERENCES snapshots(name) ON DELETE CASCADE,
    path TEXT NOT NULL,
    version_id INTEGER REFERENCES versions(id),
    PRIMARY KEY (snapshot_name, path)
);
"""


@dataclass
class VersionEntry:
    """A single file version record."""

    id: int
    path: str
    blob_hash: str | None
    operation: str
    agent_id: str | None
    session_id: str | None
    timestamp: float
    intent: str | None

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> VersionEntry:
        return cls(
            id=row[0],
            path=row[1],
            blob_hash=row[2],
            operation=row[3],
            agent_id=row[4],
            session_id=row[5],
            timestamp=row[6],
            intent=row[7],
        )


class VersionStore:
    """File version history and snapshot management backed by SQLite."""

    def __init__(self, db: aiosqlite.Connection, blob_store: BlobStore) -> None:
        self._db = db
        self._blobs = blob_store

    async def init_db(self) -> None:
        """Create tables and indices, enable WAL mode."""
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(SCHEMA)

    async def record_version(
        self,
        path: str,
        content: bytes,
        agent_id: str | None = None,
        session_id: str | None = None,
        intent: str | None = None,
    ) -> int | None:
        """Record a new file version. Returns version ID, or None if content unchanged."""
        blob_hash = await self._blobs.store(content)

        latest = await self.get_latest(path)

        if latest is not None:
            if latest.blob_hash == blob_hash:
                return None  # No-op: content identical
            operation = "modify"
            parent_id = latest.id
        else:
            operation = "create"
            parent_id = None

        ts = time.time()
        cursor = await self._db.execute(
            """INSERT INTO versions (path, blob_hash, operation, agent_id, session_id, timestamp, intent, parent_version_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (path, blob_hash, operation, agent_id, session_id, ts, intent, parent_id),
        )
        await self._db.commit()
        version_id = cursor.lastrowid
        log.debug("Recorded %s v%d for %s (%s)", operation, version_id, path, blob_hash[:12])
        
        # Prune to keep only the 3 most recent versions for this path
        await self._prune_old_versions(path)
        
        return version_id

    async def record_delete(
        self,
        path: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        intent: str | None = None,
    ) -> int:
        """Record a file deletion."""
        latest = await self.get_latest(path)
        parent_id = latest.id if latest else None

        ts = time.time()
        cursor = await self._db.execute(
            """INSERT INTO versions (path, blob_hash, operation, agent_id, session_id, timestamp, intent, parent_version_id)
               VALUES (?, NULL, 'delete', ?, ?, ?, ?, ?)""",
            (path, agent_id, session_id, ts, intent, parent_id),
        )
        await self._db.commit()
        log.debug("Recorded delete v%d for %s", cursor.lastrowid, path)
        
        # Prune old versions for this path
        await self._prune_old_versions(path)
        
        return cursor.lastrowid

    async def get_latest(self, path: str) -> VersionEntry | None:
        """Get the most recent version for a path."""
        cursor = await self._db.execute(
            """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
               FROM versions WHERE path = ? ORDER BY timestamp DESC LIMIT 1""",
            (path,),
        )
        row = await cursor.fetchone()
        return VersionEntry.from_row(row) if row else None

    async def get_version(self, version_id: int) -> VersionEntry | None:
        """Get a specific version by ID."""
        cursor = await self._db.execute(
            """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
               FROM versions WHERE id = ?""",
            (version_id,),
        )
        row = await cursor.fetchone()
        return VersionEntry.from_row(row) if row else None

    async def get_content(self, version_id: int) -> bytes:
        """Get the file content for a specific version."""
        ver = await self.get_version(version_id)
        if ver is None:
            raise KeyError(f"Version not found: {version_id}")
        if ver.blob_hash is None:
            raise ValueError(f"Version {version_id} is a delete operation (no content)")
        return await self._blobs.retrieve(ver.blob_hash)

    async def get_history(
        self,
        path: str,
        limit: int = 50,
        since_timestamp: float | None = None,
    ) -> list[VersionEntry]:
        """Get version history for a path, newest first."""
        if since_timestamp is not None:
            cursor = await self._db.execute(
                """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
                   FROM versions WHERE path = ? AND timestamp >= ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (path, since_timestamp, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
                   FROM versions WHERE path = ? ORDER BY timestamp DESC LIMIT ?""",
                (path, limit),
            )
        rows = await cursor.fetchall()
        return [VersionEntry.from_row(r) for r in rows]

    async def get_all_current_files(self) -> dict[str, VersionEntry]:
        """Get the latest non-delete version for each path."""
        cursor = await self._db.execute(
            """WITH ranked AS (
                 SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent,
                        ROW_NUMBER() OVER (PARTITION BY path ORDER BY timestamp DESC) AS rn
                 FROM versions
               )
               SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
               FROM ranked WHERE rn = 1 AND operation != 'delete'"""
        )
        rows = await cursor.fetchall()
        return {r[1]: VersionEntry.from_row(r) for r in rows}

    async def list_paths(self, prefix: str = "") -> list[str]:
        """List all current (non-deleted) file paths, optionally filtered by prefix."""
        files = await self.get_all_current_files()
        if prefix:
            return sorted(p for p in files if p.startswith(prefix))
        return sorted(files.keys())

    # --- Snapshots ---

    async def create_snapshot(
        self,
        name: str,
        description: str | None = None,
        created_by: str | None = None,
    ) -> None:
        """Create a snapshot of the current workspace state."""
        ts = time.time()
        await self._db.execute(
            "INSERT INTO snapshots (name, description, timestamp, created_by) VALUES (?, ?, ?, ?)",
            (name, description, ts, created_by),
        )

        current_files = await self.get_all_current_files()
        for path, entry in current_files.items():
            await self._db.execute(
                "INSERT INTO snapshot_files (snapshot_name, path, version_id) VALUES (?, ?, ?)",
                (name, path, entry.id),
            )
        await self._db.commit()
        log.info("Created snapshot '%s' with %d files", name, len(current_files))

    async def restore_snapshot(self, name: str) -> list[str]:
        """Restore all files from a named snapshot. Returns list of restored paths."""
        cursor = await self._db.execute(
            "SELECT path, version_id FROM snapshot_files WHERE snapshot_name = ?",
            (name,),
        )
        rows = await cursor.fetchall()
        if not rows:
            raise KeyError(f"Snapshot not found or empty: {name}")

        restored = []
        for path, version_id in rows:
            await self.restore_file_to_version(path, version_id)
            restored.append(path)
        return restored

    async def restore_file_to_version(self, path: str, version_id: int) -> None:
        """Restore a file to a specific version by creating a new version entry."""
        target = await self.get_version(version_id)
        if target is None:
            raise KeyError(f"Version not found: {version_id}")
        if target.blob_hash is None:
            raise ValueError(
                f"Cannot restore to version {version_id}: it is a delete operation (no content)"
            )

        content = await self._blobs.retrieve(target.blob_hash)
        ts = time.time()
        latest = await self.get_latest(path)
        parent_id = latest.id if latest else None

        await self._db.execute(
            """INSERT INTO versions (path, blob_hash, operation, agent_id, session_id, timestamp, intent, parent_version_id)
               VALUES (?, ?, 'modify', 'oracle', NULL, ?, ?, ?)""",
            (path, target.blob_hash, ts, f"Restored to version {version_id}", parent_id),
        )
        await self._db.commit()
        log.info("Restored %s to version %d", path, version_id)

    async def restore_files_to_time(
        self, paths: list[str], timestamp: float
    ) -> list[str]:
        """Restore files to their state at a given timestamp."""
        restored = []
        for path in paths:
            cursor = await self._db.execute(
                """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
                   FROM versions WHERE path = ? AND timestamp <= ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (path, timestamp),
            )
            row = await cursor.fetchone()
            if row and row[3] != "delete":
                entry = VersionEntry.from_row(row)
                await self.restore_file_to_version(path, entry.id)
                restored.append(path)
        return restored

    # --- Session-based queries ---

    async def get_changes_by_session(self, session_id: str) -> list[VersionEntry]:
        """Get all changes for a given session."""
        cursor = await self._db.execute(
            """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
               FROM versions WHERE session_id = ? ORDER BY timestamp""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [VersionEntry.from_row(r) for r in rows]

    async def get_changes_by_agent(
        self, agent_id: str, since: float | None = None
    ) -> list[VersionEntry]:
        """Get changes by a specific agent, optionally since a timestamp."""
        if since is not None:
            cursor = await self._db.execute(
                """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
                   FROM versions WHERE agent_id = ? AND timestamp >= ?
                   ORDER BY timestamp DESC""",
                (agent_id, since),
            )
        else:
            cursor = await self._db.execute(
                """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
                   FROM versions WHERE agent_id = ? ORDER BY timestamp DESC""",
                (agent_id,),
            )
        rows = await cursor.fetchall()
        return [VersionEntry.from_row(r) for r in rows]

    async def get_changes_since(self, timestamp: float) -> list[VersionEntry]:
        """Get all changes since a timestamp."""
        cursor = await self._db.execute(
            """SELECT id, path, blob_hash, operation, agent_id, session_id, timestamp, intent
               FROM versions WHERE timestamp >= ? ORDER BY timestamp DESC""",
            (timestamp,),
        )
        rows = await cursor.fetchall()
        return [VersionEntry.from_row(r) for r in rows]
    
    async def _prune_old_versions(self, path: str) -> None:
        """Keep only the 3 most recent versions for a given path."""
        # Get all version IDs for this path, newest first
        cursor = await self._db.execute(
            "SELECT id, blob_hash FROM versions WHERE path = ? ORDER BY timestamp DESC",
            (path,),
        )
        rows = await cursor.fetchall()
        
        if len(rows) <= 3:
            return
        
        # Delete versions beyond the 3 most recent
        to_delete = rows[3:]
        orphaned_hashes = [r[1] for r in to_delete if r[1] is not None]
        
        for row in to_delete:
            await self._db.execute("DELETE FROM versions WHERE id = ?", (row[0],))
        
        # Clean up orphaned blobs
        for blob_hash in orphaned_hashes:
            # Check if any other version still references this blob
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM versions WHERE blob_hash = ?",
                (blob_hash,),
            )
            count = (await cursor.fetchone())[0]
            if count == 0:
                await self._blobs.delete(blob_hash)
        
        await self._db.commit()
        log.debug("Pruned %d old versions for %s", len(to_delete), path)
