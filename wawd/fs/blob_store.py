"""BlobStore: content-addressed file storage with zstd compression in SQLite."""

from __future__ import annotations

import hashlib
import logging

import aiosqlite
import zstandard as zstd

log = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    compressed BLOB NOT NULL
);
"""


class BlobStore:
    """Content-addressed blob storage backed by SQLite with zstd compression."""

    def __init__(self, db: aiosqlite.Connection, compression_level: int = 3) -> None:
        self._db = db
        self._compressor = zstd.ZstdCompressor(level=compression_level)
        self._decompressor = zstd.ZstdDecompressor()

    async def init_db(self) -> None:
        """Create the blobs table if it doesn't exist."""
        await self._db.executescript(SCHEMA)

    async def store(self, content: bytes) -> str:
        """Store content and return its SHA-256 hex hash. Idempotent."""
        blob_hash = hashlib.sha256(content).hexdigest()

        existing = await self._db.execute(
            "SELECT 1 FROM blobs WHERE hash = ?", (blob_hash,)
        )
        if await existing.fetchone():
            return blob_hash

        compressed = self._compressor.compress(content)
        await self._db.execute(
            "INSERT OR IGNORE INTO blobs (hash, size, compressed) VALUES (?, ?, ?)",
            (blob_hash, len(content), compressed),
        )
        await self._db.commit()
        log.debug("Stored blob %s (%d bytes -> %d compressed)", blob_hash[:12], len(content), len(compressed))
        return blob_hash

    async def retrieve(self, blob_hash: str) -> bytes:
        """Retrieve and decompress content by hash."""
        cursor = await self._db.execute(
            "SELECT compressed FROM blobs WHERE hash = ?", (blob_hash,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Blob not found: {blob_hash}")
        return self._decompressor.decompress(row[0])

    async def exists(self, blob_hash: str) -> bool:
        """Check whether a blob exists."""
        cursor = await self._db.execute(
            "SELECT 1 FROM blobs WHERE hash = ?", (blob_hash,)
        )
        return await cursor.fetchone() is not None

    async def delete(self, blob_hash: str) -> None:
        """Delete a blob by hash."""
        await self._db.execute("DELETE FROM blobs WHERE hash = ?", (blob_hash,))
        await self._db.commit()

    async def size(self, blob_hash: str) -> int:
        """Return the original (uncompressed) size of a blob."""
        cursor = await self._db.execute(
            "SELECT size FROM blobs WHERE hash = ?", (blob_hash,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Blob not found: {blob_hash}")
        return row[0]
