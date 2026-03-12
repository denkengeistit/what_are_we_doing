"""WAWDFuse: FUSE filesystem that intercepts writes and versions them.

Uses mfusepy (ctypes-based) with FUSE-T on macOS (userland, no kext).
"""

from __future__ import annotations

import asyncio
import errno
import fnmatch
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from wawd.fs.version_store import VersionStore

log = logging.getLogger(__name__)

# 50 MB max buffer for versioning
MAX_VERSION_SIZE = 50 * 1024 * 1024

# Module-level state for the active FUSE mount
_fuse_ops: WAWDFuse | None = None
_fuse_thread: threading.Thread | None = None
_fuse_mount_point: str | None = None


def _ensure_fuse_library_path() -> None:
    """Ensure mfusepy can load the FUSE-T framework on macOS."""
    if sys.platform != "darwin":
        return
    if "FUSE_LIBRARY_PATH" in os.environ:
        return
    default = "/Library/Frameworks/fuse_t.framework/fuse_t"
    if Path(default).exists():
        os.environ["FUSE_LIBRARY_PATH"] = default


# Set env var *before* importing mfusepy so ctypes finds the lib.
_ensure_fuse_library_path()

from mfusepy import FUSE, FuseOSError, Operations  # noqa: E402


class WAWDFuse(Operations):
    """FUSE filesystem that transparently versions file writes.

    All methods are synchronous (called from mfusepy's FUSE thread).
    Async version-store calls are dispatched via ``_run_async``.
    """

    # Tell mfusepy to pass timestamps as nanoseconds (avoids deprecation warning).
    use_ns = True

    def __init__(
        self,
        source_dir: str,
        version_store: VersionStore,
        loop: asyncio.AbstractEventLoop,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._source = Path(source_dir).resolve()
        self._vs = version_store
        self._loop = loop
        self._exclude = exclude_patterns or []
        self._lock = threading.Lock()

        # Agent/session tracking (set by MCP server).
        # NOTE: these represent the *last known* active agent.  When multiple
        # agents write concurrently, attribution may be imprecise — the oracle
        # can refine authorship from session-tracker timing data.
        self.current_agent_id: str = "unknown"
        self.current_session_id: str | None = None

        # Restoration coordination — when True, writes are not versioned.
        self._paused: bool = False

        # Write buffers keyed by OS file handle
        self._buffers: dict[int, bytearray] = {}
        self._fh_to_path: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _full_path(self, path: str) -> str:
        """Resolve a FUSE relative path to an absolute source path."""
        if path.lstrip("/") == ".wawd" or path.lstrip("/").startswith(".wawd/"):
            raise FuseOSError(errno.ENOENT)
        return str(self._source / path.lstrip("/"))

    def _relative_path(self, full_path: str) -> str:
        """Convert an absolute path back to source-relative."""
        try:
            return str(Path(full_path).resolve().relative_to(self._source))
        except ValueError:
            return full_path

    def _is_excluded(self, rel_path: str) -> bool:
        """Check whether *rel_path* matches any exclude pattern."""
        for pattern in self._exclude:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path + "/", pattern):
                return True
            for part in Path(rel_path).parts:
                if fnmatch.fnmatch(part, pattern) or fnmatch.fnmatch(part + "/", pattern):
                    return True
        return False

    def _run_async(self, coro):
        """Run an async coroutine on the main event loop from the FUSE thread."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=30)

    # ------------------------------------------------------------------
    # Restoration coordination
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Suppress versioning. Called before the restorer writes to disk."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Re-enable versioning after restoration completes."""
        with self._lock:
            self._paused = False

    def invalidate(self, paths: list[str]) -> None:
        """Drop write buffers for *paths* after external (restorer) writes.

        Any file handle whose tracked path matches one of the given
        source-relative paths loses its buffer so a subsequent
        ``release()`` won't overwrite the restored content.
        """
        with self._lock:
            fhs_to_drop = [
                fh for fh, full in self._fh_to_path.items()
                if self._relative_path(full) in paths
            ]
            for fh in fhs_to_drop:
                self._buffers.pop(fh, None)

    # ------------------------------------------------------------------
    # Filesystem metadata
    # ------------------------------------------------------------------

    def access(self, path: str, mode: int) -> None:
        if not os.access(self._full_path(path), mode):
            raise FuseOSError(errno.EACCES)

    def chmod(self, path: str, mode: int) -> None:
        os.chmod(self._full_path(path), mode)

    def chown(self, path: str, uid: int, gid: int) -> None:
        os.chown(self._full_path(path), uid, gid)

    def getattr(self, path: str, fh: int | None = None) -> dict:
        full = self._full_path(path)
        try:
            st = os.lstat(full)
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT)
        return {
            "st_atime": st.st_atime,
            "st_ctime": st.st_ctime,
            "st_gid": st.st_gid,
            "st_mode": st.st_mode,
            "st_mtime": st.st_mtime,
            "st_nlink": st.st_nlink,
            "st_size": st.st_size,
            "st_uid": st.st_uid,
        }

    def readdir(self, path: str, fh: int) -> list[str]:
        full = self._full_path(path)
        entries = [".", ".."]
        try:
            for name in os.listdir(full):
                if name == ".wawd":
                    continue
                entries.append(name)
        except OSError:
            pass
        return entries

    def readlink(self, path: str) -> str:
        target = os.readlink(self._full_path(path))
        if target.startswith("/"):
            return os.path.relpath(target, str(self._source))
        return target

    def statfs(self, path: str) -> dict:
        stv = os.statvfs(self._full_path(path))
        return {
            "f_bavail": stv.f_bavail,
            "f_bfree": stv.f_bfree,
            "f_blocks": stv.f_blocks,
            "f_bsize": stv.f_bsize,
            "f_favail": stv.f_favail,
            "f_ffree": stv.f_ffree,
            "f_files": stv.f_files,
            "f_flag": stv.f_flag,
            "f_frsize": stv.f_frsize,
            "f_namemax": stv.f_namemax,
        }

    def utimens(self, path: str, times: tuple | None = None) -> None:
        os.utime(self._full_path(path), times)

    # ------------------------------------------------------------------
    # Directory operations
    # ------------------------------------------------------------------

    def mkdir(self, path: str, mode: int) -> None:
        os.mkdir(self._full_path(path), mode)

    def rmdir(self, path: str) -> None:
        os.rmdir(self._full_path(path))

    def mknod(self, path: str, mode: int, dev: int) -> None:
        os.mknod(self._full_path(path), mode, dev)

    def symlink(self, target: str, name: str) -> None:
        os.symlink(target, self._full_path(name))

    def link(self, target: str, name: str) -> None:
        os.link(self._full_path(target), self._full_path(name))

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def open(self, path: str, flags: int) -> int:
        full = self._full_path(path)
        fh = os.open(full, flags)

        with self._lock:
            self._fh_to_path[fh] = full

            if flags & (os.O_WRONLY | os.O_RDWR):
                try:
                    with open(full, "rb") as f:
                        content = f.read()
                except OSError:
                    content = b""
                if len(content) <= MAX_VERSION_SIZE:
                    self._buffers[fh] = bytearray(content)
                if flags & os.O_TRUNC:
                    self._buffers[fh] = bytearray()

        return fh

    def create(self, path: str, mode: int, fi=None) -> int:
        full = self._full_path(path)
        fh = os.open(full, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with self._lock:
            self._fh_to_path[fh] = full
            self._buffers[fh] = bytearray()
        return fh

    def read(self, path: str, length: int, offset: int, fh: int) -> bytes:
        os.lseek(fh, offset, os.SEEK_SET)
        return os.read(fh, length)

    def write(self, path: str, buf: bytes, offset: int, fh: int) -> int:
        os.lseek(fh, offset, os.SEEK_SET)
        written = os.write(fh, buf)

        with self._lock:
            if fh in self._buffers:
                b = self._buffers[fh]
                end = offset + len(buf)
                if end > len(b):
                    b.extend(b"\x00" * (end - len(b)))
                b[offset:end] = buf

        return written

    def truncate(self, path: str, length: int, fh: int | None = None) -> None:
        full = self._full_path(path)
        with open(full, "r+b") as f:
            f.truncate(length)
        with self._lock:
            if fh is not None and fh in self._buffers:
                self._buffers[fh] = bytearray(self._buffers[fh][:length])
            else:
                # FUSE-T (NFS-based) sends truncate without an fh after open.
                # Find all open buffers for this path and truncate them.
                for tracked_fh, tracked_path in self._fh_to_path.items():
                    if tracked_path == full and tracked_fh in self._buffers:
                        self._buffers[tracked_fh] = bytearray(
                            self._buffers[tracked_fh][:length]
                        )

    def flush(self, path: str, fh: int) -> None:
        os.fsync(fh)

    def release(self, path: str, fh: int) -> None:
        """File handle closed — version the content if it was buffered."""
        with self._lock:
            full = self._fh_to_path.pop(fh, None)
            buffer = self._buffers.pop(fh, None)
            paused = self._paused
        os.close(fh)

        if full is None or buffer is None or paused:
            return

        content = bytes(buffer)
        rel = self._relative_path(full)

        if self._is_excluded(rel) or len(content) > MAX_VERSION_SIZE:
            return

        try:
            version_id = self._run_async(
                self._vs.record_version(
                    path=rel,
                    content=content,
                    agent_id=self.current_agent_id,
                    session_id=self.current_session_id,
                )
            )
            if version_id is not None:
                log.debug("Versioned %s -> v%d", rel, version_id)
        except Exception:
            log.exception("Failed to version %s", rel)

    def fsync(self, path: str, fdatasync: bool, fh: int) -> None:
        self.flush(path, fh)

    # ------------------------------------------------------------------
    # Versioned destructive operations
    # ------------------------------------------------------------------

    def unlink(self, path: str) -> None:
        full = self._full_path(path)
        rel = self._relative_path(full)
        os.unlink(full)

        if self._paused or self._is_excluded(rel):
            return

        try:
            self._run_async(
                self._vs.record_delete(
                    path=rel,
                    agent_id=self.current_agent_id,
                    session_id=self.current_session_id,
                )
            )
        except Exception:
            log.exception("Failed to record delete for %s", rel)

    def rename(self, old: str, new: str) -> None:
        old_full = self._full_path(old)
        new_full = self._full_path(new)
        # Resolve relative paths BEFORE the rename so old_full still exists.
        old_rel = self._relative_path(old_full)
        new_rel = self._relative_path(new_full)

        content = None
        if os.path.isfile(old_full):
            try:
                with open(old_full, "rb") as f:
                    content = f.read()
            except OSError:
                pass

        os.rename(old_full, new_full)

        if self._paused:
            return

        if not self._is_excluded(old_rel):
            try:
                self._run_async(
                    self._vs.record_delete(
                        path=old_rel,
                        agent_id=self.current_agent_id,
                        session_id=self.current_session_id,
                        intent=f"Renamed to {new_rel}",
                    )
                )
            except Exception:
                log.exception("Failed to record delete during rename for %s", old_rel)

        if content is not None and len(content) <= MAX_VERSION_SIZE and not self._is_excluded(new_rel):
            try:
                self._run_async(
                    self._vs.record_version(
                        path=new_rel,
                        content=content,
                        agent_id=self.current_agent_id,
                        session_id=self.current_session_id,
                        intent=f"Renamed from {old_rel}",
                    )
                )
            except Exception:
                log.exception("Failed to record create during rename for %s", new_rel)


# ======================================================================
# Public helpers — called from cli.py
# ======================================================================

def _fuse_thread_main(ops: WAWDFuse, mount_point: str) -> None:
    """Target for the daemon thread that runs the blocking FUSE main loop."""
    try:
        FUSE(
            ops,
            mount_point,
            foreground=True,
            nothreads=False,
            fsname="wawd",
            allow_other=False,
        )
    except Exception:
        log.exception("FUSE thread crashed")


async def mount_fuse(
    source_dir: str,
    mount_point: str,
    version_store: VersionStore,
    exclude_patterns: list[str] | None = None,
) -> WAWDFuse:
    """Initialize and mount the FUSE filesystem. Returns the ops instance."""
    global _fuse_ops, _fuse_thread, _fuse_mount_point

    Path(mount_point).mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    ops = WAWDFuse(source_dir, version_store, loop, exclude_patterns)

    t = threading.Thread(
        target=_fuse_thread_main,
        args=(ops, mount_point),
        daemon=True,
        name="wawd-fuse",
    )
    t.start()

    _fuse_ops = ops
    _fuse_thread = t
    _fuse_mount_point = mount_point

    # Give FUSE a moment to initialise.
    await asyncio.sleep(0.5)
    log.info("FUSE thread started: %s -> %s", source_dir, mount_point)
    return ops


async def run_fuse_loop() -> None:
    """Keep-alive coroutine that watches the FUSE daemon thread."""
    while True:
        await asyncio.sleep(0.5)
        if _fuse_thread is not None and not _fuse_thread.is_alive():
            raise RuntimeError("FUSE thread exited unexpectedly")


async def stop_fuse() -> None:
    """Unmount the FUSE filesystem (best-effort)."""
    global _fuse_mount_point
    if not _fuse_mount_point:
        return
    subprocess.run(["umount", _fuse_mount_point], capture_output=True, text=True)
    await asyncio.sleep(0.2)
