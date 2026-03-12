"""Inner script for restore integration tests.

Executed as a subprocess by test_restore_integration.py.  Uses os._exit(0)
after all assertions pass so FUSE-T teardown cannot SIGPIPE the process.
"""

import asyncio
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time

signal.signal(signal.SIGPIPE, signal.SIG_IGN)

if sys.platform == "darwin":
    fuse_lib = "/Library/Frameworks/fuse_t.framework/fuse_t"
    if os.path.exists(fuse_lib):
        os.environ.setdefault("FUSE_LIBRARY_PATH", fuse_lib)

import aiosqlite

from wawd.fs.blob_store import BlobStore
from wawd.fs.version_store import VersionStore
from wawd.fs.fuse_mount import mount_fuse, stop_fuse
from wawd.oracle.context import ContextBuilder
from wawd.oracle.restorer import Restorer, RestorationPlan, FileRestoration
from wawd.oracle.session_tracker import SessionTracker


def _write_file(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


class FakeBackend:
    async def generate(self, messages, max_tokens=2048):
        return '{"files_to_restore": [], "explanation": "stub"}'

    async def health_check(self):
        return True

    async def close(self):
        pass


async def run_tests() -> None:
    source_dir = tempfile.mkdtemp(prefix="wawd_rsrc_")
    mount_point = tempfile.mkdtemp(prefix="wawd_rmnt_")
    db_path = os.path.join(source_dir, ".wawd_test.db")

    print(f"Source:  {source_dir}", flush=True)
    print(f"Mount:   {mount_point}", flush=True)

    db = await aiosqlite.connect(db_path)
    try:
        blob_store = BlobStore(db, compression_level=3)
        await blob_store.init_db()
        vs = VersionStore(db, blob_store)
        await vs.init_db()
        st = SessionTracker(db, timeout_minutes=30)
        await st.init_db()

        ops = await mount_fuse(
            source_dir, mount_point, vs,
            exclude_patterns=[".wawd_test.db", ".wawd_test.db-*", "._*"],
        )
        print("✓ FUSE mounted", flush=True)
        await asyncio.sleep(1.0)

        f1 = os.path.join(mount_point, "target.txt")

        # --- Seed ---
        await asyncio.to_thread(_write_file, f1, "version one\n")
        await asyncio.sleep(0.5)
        v1 = await vs.get_latest("target.txt")
        assert v1 is not None and v1.operation == "create"
        print(f"✓ Seed: target.txt v1 (id={v1.id})", flush=True)

        # --- Test 1: Writes while paused are NOT versioned ---
        ops.pause()
        await asyncio.to_thread(_write_file, f1, "paused write\n")
        await asyncio.sleep(0.5)
        still_v1 = await vs.get_latest("target.txt")
        assert still_v1.id == v1.id
        ops.resume()
        await asyncio.sleep(0.5)
        print("✓ Test 1: Write during pause was not versioned", flush=True)

        # --- Test 2: Write after resume IS versioned ---
        await asyncio.to_thread(_write_file, f1, "version two\n")
        await asyncio.sleep(0.5)
        v2 = await vs.get_latest("target.txt")
        assert v2 is not None and v2.id > v1.id
        content = await vs.get_content(v2.id)
        assert content == b"version two\n", f"Got {repr(content)}"
        print(f"✓ Test 2: Write after resume versioned (id={v2.id})", flush=True)

        # --- Write v3 for restoration ---
        await asyncio.to_thread(_write_file, f1, "version three\n")
        await asyncio.sleep(0.5)
        v3 = await vs.get_latest("target.txt")
        assert v3.id > v2.id

        # --- Test 3: DB restoration with pause/resume ---
        await vs.create_snapshot(
            name=f"test-snap-{int(time.time())}",
            description="test",
            created_by="test",
        )
        ops.pause()
        await vs.restore_file_to_version("target.txt", v2.id)
        ops.resume()
        await asyncio.sleep(0.3)
        v_restored = await vs.get_latest("target.txt")
        assert v_restored.id > v3.id
        assert "Restored" in v_restored.intent
        print(f"✓ Test 3: DB restoration worked (id={v_restored.id})", flush=True)

        # --- Test 4: Write through mount after restore ---
        await asyncio.to_thread(_write_file, f1, "after restore\n")
        await asyncio.sleep(0.5)
        v_after = await vs.get_latest("target.txt")
        assert v_after.id > v_restored.id
        print(f"✓ Test 4: Post-restore write works (id={v_after.id})", flush=True)

        # --- Test 5: Fallback plan builder ---
        ctx = ContextBuilder(vs, st, context_budget_tokens=2000, history_depth=10)
        restorer = Restorer(vs, ctx, FakeBackend(), source_dir)
        fallback = await restorer._build_fallback_plan(scope=None, snapshot_name="fb")
        assert len(fallback.files) >= 1
        print(f"✓ Test 5: Fallback plan ({len(fallback.files)} files)", flush=True)

        # --- Test 6: Scoped fallback ---
        fb2 = await restorer._build_fallback_plan(scope="nonexistent/", snapshot_name="fb2")
        assert len(fb2.files) == 0
        print("✓ Test 6: Scoped fallback empty for non-matching prefix", flush=True)

        # --- Summary ---
        cursor = await db.execute("SELECT COUNT(*) FROM versions")
        total = (await cursor.fetchone())[0]
        print(f"\n=== ALL 6 TESTS PASSED === ({total} versions)", flush=True)

        # Exit immediately so FUSE-T teardown cannot SIGPIPE us.
        os._exit(0)

    finally:
        await stop_fuse()
        await asyncio.sleep(0.3)
        await db.close()
        shutil.rmtree(source_dir, ignore_errors=True)
        try:
            os.rmdir(mount_point)
        except OSError:
            subprocess.run(["umount", mount_point], capture_output=True)
            shutil.rmtree(mount_point, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_tests())
