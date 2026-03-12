"""End-to-end smoke test for the mfusepy FUSE layer with versioning.

Sequence:
  1. Create temp source dir + mount point
  2. Init SQLite DB with BlobStore / VersionStore
  3. Mount WAWDFuse via mount_fuse()
  4. Perform file operations through the mount point
  5. Verify versions appear in the database
  6. Unmount and clean up
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Ensure FUSE-T library path is set before any wawd imports
if sys.platform == "darwin":
    fuse_lib = "/Library/Frameworks/fuse_t.framework/fuse_t"
    if os.path.exists(fuse_lib):
        os.environ.setdefault("FUSE_LIBRARY_PATH", fuse_lib)

import aiosqlite

from wawd.fs.blob_store import BlobStore
from wawd.fs.version_store import VersionStore
from wawd.fs.fuse_mount import mount_fuse, stop_fuse


# ---------------------------------------------------------------------------
# Helpers — all mount-side file I/O must happen OFF the event loop thread to
# avoid deadlocking with _run_async() which schedules coroutines back onto
# the event loop.  In production this is fine because external processes
# (editors, agents) do the I/O, but in-process tests share the loop.
# ---------------------------------------------------------------------------

def _write_file(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)

def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

def _listdir(path: str) -> list[str]:
    return os.listdir(path)

def _unlink(path: str) -> None:
    os.unlink(path)

def _mkdir(path: str) -> None:
    os.mkdir(path)

def _rmdir(path: str) -> None:
    os.rmdir(path)


async def run_test() -> None:
    source_dir = tempfile.mkdtemp(prefix="wawd_src_")
    mount_point = tempfile.mkdtemp(prefix="wawd_mnt_")
    db_path = os.path.join(source_dir, ".wawd_test.db")

    print(f"Source:  {source_dir}")
    print(f"Mount:   {mount_point}")
    print(f"DB:      {db_path}")

    # Seed source with an existing file
    with open(os.path.join(source_dir, "existing.txt"), "w") as f:
        f.write("original content\n")

    db = await aiosqlite.connect(db_path)
    try:
        blob_store = BlobStore(db, compression_level=3)
        await blob_store.init_db()
        version_store = VersionStore(db, blob_store)
        await version_store.init_db()

        # Mount FUSE
        ops = await mount_fuse(
            source_dir, mount_point, version_store,
            exclude_patterns=[".wawd_test.db", ".wawd_test.db-*", "*.pyc", "._*"],
        )
        print("✓ FUSE mounted")

        # Give mount a moment to stabilise
        await asyncio.sleep(1.0)

        # --- Test 1: ls mount shows existing file ---
        entries = await asyncio.to_thread(_listdir, mount_point)
        assert "existing.txt" in entries, f"Expected existing.txt in {entries}"
        print("✓ Test 1: ls shows existing file")

        # --- Test 2: Create a new file through the mount ---
        new_file = os.path.join(mount_point, "hello.txt")
        await asyncio.to_thread(_write_file, new_file, "hello world\n")
        await asyncio.sleep(0.5)  # let versioning complete

        # Verify on source side
        src_hello = os.path.join(source_dir, "hello.txt")
        assert os.path.exists(src_hello), "File not passthrough'd to source"
        with open(src_hello) as f:
            assert f.read() == "hello world\n"
        print("✓ Test 2: Create file through mount, passthrough works")

        # --- Test 3: Check version was recorded ---
        latest = await version_store.get_latest("hello.txt")
        assert latest is not None, "No version recorded for hello.txt"
        assert latest.operation == "create", f"Expected 'create', got '{latest.operation}'"
        content = await version_store.get_content(latest.id)
        assert content == b"hello world\n"
        print(f"✓ Test 3: Version recorded (id={latest.id}, op={latest.operation})")

        # --- Test 4: Modify existing file through mount ---
        existing_mount = os.path.join(mount_point, "existing.txt")
        await asyncio.to_thread(_write_file, existing_mount, "modified content\n")
        await asyncio.sleep(0.5)

        latest_existing = await version_store.get_latest("existing.txt")
        assert latest_existing is not None, "No version for existing.txt"
        content2 = await version_store.get_content(latest_existing.id)
        assert content2 == b"modified content\n"
        print(f"✓ Test 4: Modify versioned (id={latest_existing.id}, op={latest_existing.operation})")

        # --- Test 5: Write again (second modify) ---
        await asyncio.to_thread(_write_file, existing_mount, "second edit\n")
        await asyncio.sleep(0.5)

        history = await version_store.get_history("existing.txt")
        assert len(history) >= 2, f"Expected >=2 versions, got {len(history)}"
        print(f"✓ Test 5: Multiple versions tracked ({len(history)} versions for existing.txt)")

        # --- Test 6: Delete a file ---
        await asyncio.to_thread(_unlink, os.path.join(mount_point, "hello.txt"))
        await asyncio.sleep(0.5)

        hello_latest = await version_store.get_latest("hello.txt")
        assert hello_latest is not None and hello_latest.operation == "delete", \
            f"Expected delete op, got {hello_latest}"
        print("✓ Test 6: Delete recorded in version store")

        # --- Test 7: Read file through mount ---
        data = await asyncio.to_thread(_read_file, existing_mount)
        assert data == "second edit\n", f"Read mismatch: {data!r}"
        print("✓ Test 7: Read through mount works")

        # --- Test 8: mkdir / rmdir ---
        subdir = os.path.join(mount_point, "subdir")
        await asyncio.to_thread(_mkdir, subdir)
        assert os.path.isdir(os.path.join(source_dir, "subdir"))
        await asyncio.to_thread(_rmdir, subdir)
        assert not os.path.exists(os.path.join(source_dir, "subdir"))
        print("✓ Test 8: mkdir/rmdir passthrough works")

        # --- Summary ---
        cursor = await db.execute("SELECT COUNT(*) FROM versions")
        total = (await cursor.fetchone())[0]
        cursor2 = await db.execute("SELECT COUNT(*) FROM blobs")
        blobs = (await cursor2.fetchone())[0]
        print(f"\n=== All tests passed ===")
        print(f"Total versions: {total}, Total blobs: {blobs}")

    finally:
        await stop_fuse()
        await asyncio.sleep(0.5)
        await db.close()

        # Clean up temp dirs
        shutil.rmtree(source_dir, ignore_errors=True)
        try:
            os.rmdir(mount_point)
        except OSError:
            subprocess.run(["umount", mount_point], capture_output=True)
            time.sleep(0.3)
            shutil.rmtree(mount_point, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_test())
