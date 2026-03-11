"""WAWDFuse: FUSE filesystem that intercepts writes and versions them.

Uses pyfuse3 which is compatible with FUSE-T (userland, no kext on macOS).
"""

from __future__ import annotations

import asyncio
import errno
import fnmatch
import logging
import os
import stat
import time
from pathlib import Path

import pyfuse3

from wawd.fs.version_store import VersionStore

log = logging.getLogger(__name__)

# 50 MB max buffer for versioning
MAX_VERSION_SIZE = 50 * 1024 * 1024


class WAWDFuse(pyfuse3.Operations):
    """FUSE filesystem that transparently versions file writes."""

    def __init__(
        self,
        source_dir: str,
        version_store: VersionStore,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._source = Path(source_dir).resolve()
        self._vs = version_store
        self._exclude = exclude_patterns or []

        # Agent/session tracking (set by MCP server)
        self.current_agent_id: str = "unknown"
        self.current_session_id: str | None = None

        # Inode management
        self._inode_to_path: dict[int, Path] = {pyfuse3.ROOT_INODE: self._source}
        self._path_to_inode: dict[Path, int] = {self._source: pyfuse3.ROOT_INODE}
        self._next_inode = pyfuse3.ROOT_INODE + 1

        # Write buffers keyed by file handle
        self._buffers: dict[int, bytearray] = {}
        self._fh_to_path: dict[int, Path] = {}
        self._fh_flags: dict[int, int] = {}
        self._next_fh = 1

    def _get_inode(self, path: Path) -> int:
        """Get or assign an inode for a path."""
        path = path.resolve()
        if path in self._path_to_inode:
            return self._path_to_inode[path]
        inode = self._next_inode
        self._next_inode += 1
        self._inode_to_path[inode] = path
        self._path_to_inode[path] = inode
        return inode

    def _inode_path(self, inode: int) -> Path:
        """Resolve an inode to its filesystem path."""
        try:
            return self._inode_to_path[inode]
        except KeyError:
            raise pyfuse3.FUSEError(errno.ENOENT)

    def _relative_path(self, path: Path) -> str:
        """Get path relative to source directory."""
        try:
            return str(path.resolve().relative_to(self._source))
        except ValueError:
            return str(path)

    def _is_excluded(self, rel_path: str) -> bool:
        """Check if a path matches any exclude pattern."""
        for pattern in self._exclude:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(
                rel_path + "/", pattern
            ):
                return True
            # Check each path component
            for part in Path(rel_path).parts:
                if fnmatch.fnmatch(part, pattern) or fnmatch.fnmatch(
                    part + "/", pattern
                ):
                    return True
        return False

    def _make_attr(self, path: Path, inode: int) -> pyfuse3.EntryAttributes:
        """Build entry attributes from an on-disk path."""
        try:
            st = os.lstat(str(path))
        except FileNotFoundError:
            raise pyfuse3.FUSEError(errno.ENOENT)

        attr = pyfuse3.EntryAttributes()
        attr.st_ino = inode
        attr.st_mode = st.st_mode
        attr.st_nlink = st.st_nlink
        attr.st_uid = st.st_uid
        attr.st_gid = st.st_gid
        attr.st_size = st.st_size
        attr.st_atime_ns = st.st_atime_ns
        attr.st_mtime_ns = st.st_mtime_ns
        attr.st_ctime_ns = st.st_ctime_ns
        attr.st_blksize = getattr(st, "st_blksize", 512)
        attr.st_blocks = getattr(st, "st_blocks", 0)
        attr.generation = 0
        attr.entry_timeout = 1
        attr.attr_timeout = 1
        return attr

    # --- FUSE Operations ---

    async def getattr(self, inode: int, ctx=None) -> pyfuse3.EntryAttributes:
        path = self._inode_path(inode)
        return self._make_attr(path, inode)

    async def lookup(self, parent_inode: int, name: bytes, ctx=None) -> pyfuse3.EntryAttributes:
        name_str = name.decode()
        # Block /.wawd/ for now (Phase 3)
        if name_str == ".wawd":
            raise pyfuse3.FUSEError(errno.ENOENT)

        parent_path = self._inode_path(parent_inode)
        child_path = parent_path / name_str

        if not child_path.exists():
            raise pyfuse3.FUSEError(errno.ENOENT)

        inode = self._get_inode(child_path)
        return self._make_attr(child_path, inode)

    async def opendir(self, inode: int, ctx=None) -> int:
        self._inode_path(inode)  # Validate
        fh = self._next_fh
        self._next_fh += 1
        self._fh_to_path[fh] = self._inode_path(inode)
        return fh

    async def readdir(self, fh: int, start_id: int, token) -> None:
        path = self._fh_to_path[fh]
        entries = []
        try:
            for i, entry in enumerate(sorted(os.scandir(str(path)), key=lambda e: e.name)):
                if entry.name == ".wawd":
                    continue
                child_path = Path(entry.path)
                inode = self._get_inode(child_path)
                entries.append((inode, entry.name, child_path))
        except OSError:
            return

        for i, (inode, name, child_path) in enumerate(entries):
            if i < start_id:
                continue
            attr = self._make_attr(child_path, inode)
            if not pyfuse3.readdir_reply(token, name.encode(), attr, i + 1):
                break

    async def releasedir(self, fh: int) -> None:
        self._fh_to_path.pop(fh, None)

    async def open(self, inode: int, flags: int, ctx=None) -> pyfuse3.FileInfo:
        path = self._inode_path(inode)
        fh = self._next_fh
        self._next_fh += 1
        self._fh_to_path[fh] = path
        self._fh_flags[fh] = flags

        # If opened for writing, initialize buffer with current content
        if flags & (os.O_WRONLY | os.O_RDWR):
            try:
                if path.is_file():
                    content = path.read_bytes()
                    if len(content) <= MAX_VERSION_SIZE:
                        self._buffers[fh] = bytearray(content)
                    # Large files: skip versioning
                else:
                    self._buffers[fh] = bytearray()
            except OSError:
                self._buffers[fh] = bytearray()

            if flags & os.O_TRUNC and fh in self._buffers:
                self._buffers[fh] = bytearray()

        fi = pyfuse3.FileInfo()
        fi.fh = fh
        fi.keep_cache = False
        return fi

    async def create(
        self, parent_inode: int, name: bytes, mode: int, flags: int, ctx=None
    ) -> tuple[pyfuse3.FileInfo, pyfuse3.EntryAttributes]:
        parent_path = self._inode_path(parent_inode)
        child_path = parent_path / name.decode()

        # Create the file on disk
        fd = os.open(str(child_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
        os.close(fd)

        inode = self._get_inode(child_path)
        fh = self._next_fh
        self._next_fh += 1
        self._fh_to_path[fh] = child_path
        self._fh_flags[fh] = flags
        self._buffers[fh] = bytearray()

        fi = pyfuse3.FileInfo()
        fi.fh = fh
        fi.keep_cache = False

        attr = self._make_attr(child_path, inode)
        return (fi, attr)

    async def read(self, fh: int, offset: int, size: int) -> bytes:
        # If we have a write buffer, read from it
        if fh in self._buffers:
            buf = self._buffers[fh]
            return bytes(buf[offset : offset + size])

        path = self._fh_to_path.get(fh)
        if path is None:
            raise pyfuse3.FUSEError(errno.EBADF)

        try:
            with open(str(path), "rb") as f:
                f.seek(offset)
                return f.read(size)
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno or errno.EIO)

    async def write(self, fh: int, offset: int, buf: bytes) -> int:
        if fh not in self._buffers:
            # Large file or non-buffered: write through
            path = self._fh_to_path.get(fh)
            if path is None:
                raise pyfuse3.FUSEError(errno.EBADF)
            with open(str(path), "r+b") as f:
                f.seek(offset)
                return f.write(buf)

        buffer = self._buffers[fh]
        end = offset + len(buf)
        if end > len(buffer):
            buffer.extend(b"\x00" * (end - len(buffer)))
        buffer[offset:end] = buf
        return len(buf)

    async def release(self, fh: int) -> None:
        """Called when a file handle is closed. Version if content changed."""
        path = self._fh_to_path.pop(fh, None)
        self._fh_flags.pop(fh, None)
        buffer = self._buffers.pop(fh, None)

        if path is None or buffer is None:
            return

        content = bytes(buffer)
        rel_path = self._relative_path(path)

        # Write content to disk
        try:
            path.write_bytes(content)
        except OSError:
            log.exception("Failed to write %s to disk", rel_path)
            return

        # Skip versioning for excluded paths or oversized files
        if self._is_excluded(rel_path) or len(content) > MAX_VERSION_SIZE:
            return

        # Record version (async — run in the event loop)
        try:
            version_id = await self._vs.record_version(
                path=rel_path,
                content=content,
                agent_id=self.current_agent_id,
                session_id=self.current_session_id,
            )
            if version_id is not None:
                log.debug("Versioned %s -> v%d", rel_path, version_id)
        except Exception:
            log.exception("Failed to version %s", rel_path)

    async def unlink(self, parent_inode: int, name: bytes, ctx=None) -> None:
        parent_path = self._inode_path(parent_inode)
        child_path = parent_path / name.decode()
        rel_path = self._relative_path(child_path)

        os.unlink(str(child_path))

        # Remove from inode maps
        child_path_resolved = child_path.resolve()
        inode = self._path_to_inode.pop(child_path_resolved, None)
        if inode is not None:
            self._inode_to_path.pop(inode, None)

        # Record delete
        if not self._is_excluded(rel_path):
            try:
                await self._vs.record_delete(
                    path=rel_path,
                    agent_id=self.current_agent_id,
                    session_id=self.current_session_id,
                )
            except Exception:
                log.exception("Failed to record delete for %s", rel_path)

    async def rename(
        self,
        parent_inode_old: int,
        name_old: bytes,
        parent_inode_new: int,
        name_new: bytes,
        flags: int,
        ctx=None,
    ) -> None:
        old_parent = self._inode_path(parent_inode_old)
        new_parent = self._inode_path(parent_inode_new)
        old_path = old_parent / name_old.decode()
        new_path = new_parent / name_new.decode()

        # Read content before rename for versioning
        content = None
        if old_path.is_file():
            try:
                content = old_path.read_bytes()
            except OSError:
                pass

        os.rename(str(old_path), str(new_path))

        # Update inode maps
        old_resolved = old_path.resolve()
        inode = self._path_to_inode.pop(old_resolved, None)
        new_resolved = new_path.resolve()
        if inode is not None:
            self._inode_to_path[inode] = new_resolved
            self._path_to_inode[new_resolved] = inode

        old_rel = self._relative_path(old_path)
        new_rel = self._relative_path(new_path)

        if not self._is_excluded(old_rel):
            try:
                await self._vs.record_delete(
                    path=old_rel,
                    agent_id=self.current_agent_id,
                    session_id=self.current_session_id,
                    intent=f"Renamed to {new_rel}",
                )
            except Exception:
                log.exception("Failed to record delete for rename %s", old_rel)

        if content is not None and not self._is_excluded(new_rel) and len(content) <= MAX_VERSION_SIZE:
            try:
                await self._vs.record_version(
                    path=new_rel,
                    content=content,
                    agent_id=self.current_agent_id,
                    session_id=self.current_session_id,
                    intent=f"Renamed from {old_rel}",
                )
            except Exception:
                log.exception("Failed to record create for rename %s", new_rel)

    async def mkdir(self, parent_inode: int, name: bytes, mode: int, ctx=None) -> pyfuse3.EntryAttributes:
        parent_path = self._inode_path(parent_inode)
        child_path = parent_path / name.decode()
        os.mkdir(str(child_path), mode)
        inode = self._get_inode(child_path)
        return self._make_attr(child_path, inode)

    async def rmdir(self, parent_inode: int, name: bytes, ctx=None) -> None:
        parent_path = self._inode_path(parent_inode)
        child_path = parent_path / name.decode()
        os.rmdir(str(child_path))

        child_resolved = child_path.resolve()
        inode = self._path_to_inode.pop(child_resolved, None)
        if inode is not None:
            self._inode_to_path.pop(inode, None)

    async def setattr(self, inode: int, attr, fields, fh=None, ctx=None) -> pyfuse3.EntryAttributes:
        path = self._inode_path(inode)

        if fields.update_size:
            with open(str(path), "r+b") as f:
                f.truncate(attr.st_size)
            # Also truncate buffer if present
            if fh is not None and fh in self._buffers:
                self._buffers[fh] = bytearray(self._buffers[fh][: attr.st_size])

        if fields.update_mode:
            os.chmod(str(path), stat.S_IMODE(attr.st_mode))

        if fields.update_uid or fields.update_gid:
            uid = attr.st_uid if fields.update_uid else -1
            gid = attr.st_gid if fields.update_gid else -1
            os.chown(str(path), uid, gid)

        if fields.update_atime or fields.update_mtime:
            atime_ns = attr.st_atime_ns if fields.update_atime else None
            mtime_ns = attr.st_mtime_ns if fields.update_mtime else None
            if atime_ns is not None or mtime_ns is not None:
                st = os.lstat(str(path))
                a = atime_ns if atime_ns is not None else st.st_atime_ns
                m = mtime_ns if mtime_ns is not None else st.st_mtime_ns
                os.utime(str(path), ns=(a, m))

        return self._make_attr(path, inode)

    async def statfs(self, ctx=None) -> pyfuse3.StatvfsData:
        st = os.statvfs(str(self._source))
        out = pyfuse3.StatvfsData()
        out.f_bsize = st.f_bsize
        out.f_frsize = st.f_frsize
        out.f_blocks = st.f_blocks
        out.f_bfree = st.f_bfree
        out.f_bavail = st.f_bavail
        out.f_files = st.f_files
        out.f_ffree = st.f_ffree
        out.f_favail = st.f_favail
        out.f_namemax = st.f_namemax
        return out


async def mount_fuse(
    source_dir: str,
    mount_point: str,
    version_store: VersionStore,
    exclude_patterns: list[str] | None = None,
) -> WAWDFuse:
    """Initialize and mount the FUSE filesystem. Returns the operations instance."""
    mount_path = Path(mount_point)
    mount_path.mkdir(parents=True, exist_ok=True)

    ops = WAWDFuse(source_dir, version_store, exclude_patterns)

    fuse_options = set(pyfuse3.default_options)
    fuse_options.add("fsname=wawd")
    fuse_options.discard("default_permissions")

    pyfuse3.init(ops, str(mount_path), fuse_options)
    log.info("FUSE mounted: %s -> %s", source_dir, mount_point)
    return ops


async def run_fuse_loop() -> None:
    """Run the FUSE event loop (blocking)."""
    try:
        await pyfuse3.main()
    except Exception:
        log.exception("FUSE event loop error")
    finally:
        pyfuse3.close(unmount=True)
