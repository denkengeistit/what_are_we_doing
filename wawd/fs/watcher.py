"""Watchdog-based directory watcher for WAWD.

Uses OS-level filesystem events (FSEvents on macOS, inotify on Linux)
via the *watchdog* library to detect changes, debounces them, then
hands batches to the version store.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
from pathlib import Path
from typing import Sequence

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from wawd.fs.version_store import VersionStore

log = logging.getLogger(__name__)

# Avoid circular import — SessionTracker is only used for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wawd.oracle.session_tracker import SessionTracker

_DEBOUNCE_SECONDS = 0.5  # coalesce rapid-fire events


class _Handler(FileSystemEventHandler):
    """Collect raw events; the watcher will drain them periodically."""

    def __init__(self, watcher: "WAWDWatcher") -> None:
        self._w = watcher

    # — only care about files, not directories —
    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._w._enqueue(event.src_path, "create")

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory:
            self._w._enqueue(event.src_path, "write")

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory:
            self._w._enqueue(event.src_path, "delete")

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._w._enqueue(event.src_path, "delete")
            self._w._enqueue(event.dest_path, "create")


class WAWDWatcher:
    """Watch *root* for changes, version them via *version_store*.

    Attribution is resolved at version-time by querying the
    SessionTracker for the most recently active session, so
    concurrent agents are attributed correctly.
    """

    def __init__(
        self,
        root: str | Path,
        version_store: VersionStore,
        exclude: Sequence[str] = (),
        session_tracker: SessionTracker | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._vs = version_store
        self._exclude = list(exclude)
        self._session_tracker = session_tracker

        # Internal state
        self._observer: Observer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[str, str] = {}  # abs_path -> last op
        self._drain_task: asyncio.Task | None = None
        self._paused = False

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the observer thread and the async drain loop."""
        self._loop = asyncio.get_running_loop()
        handler = _Handler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._root), recursive=True)
        self._observer.start()
        self._drain_task = asyncio.create_task(self._drain_loop())
        log.info("Watcher started on %s", self._root)

        # initial scan — version every existing file once
        await self._initial_scan()

    async def stop(self) -> None:
        """Stop observer + drain loop."""
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        if self._observer:
            self._observer.stop()
            self._observer.join()
        log.info("Watcher stopped")

    # ── pause / resume (used by Restorer) ─────────────────────────

    def pause(self) -> None:
        self._paused = True
        log.debug("Watcher paused")

    def resume(self) -> None:
        self._paused = False
        log.debug("Watcher resumed")

    def invalidate(self, paths: Sequence[str]) -> None:
        """No-op kept for API compatibility with Restorer."""
        pass

    # ── internals ─────────────────────────────────────────────────

    def _enqueue(self, abs_path: str, op: str) -> None:
        """Thread-safe enqueue from watchdog handler thread."""
        if self._paused:
            return
        rel = self._relpath(abs_path)
        if rel is None or self._is_excluded(rel):
            return
        self._pending[abs_path] = op

    def _relpath(self, abs_path: str) -> str | None:
        try:
            return str(Path(abs_path).relative_to(self._root))
        except ValueError:
            return None

    def _is_excluded(self, rel_path: str) -> bool:
        parts = Path(rel_path).parts
        for pattern in self._exclude:
            # directory pattern — check each component
            if pattern.endswith("/"):
                if any(fnmatch.fnmatch(p, pattern.rstrip("/")) for p in parts):
                    return True
            else:
                if fnmatch.fnmatch(Path(rel_path).name, pattern):
                    return True
        return False

    async def _drain_loop(self) -> None:
        """Periodically flush pending events to the version store."""
        while True:
            await asyncio.sleep(_DEBOUNCE_SECONDS)
            if self._paused or not self._pending:
                continue
            batch = dict(self._pending)
            self._pending.clear()
            for abs_path, op in batch.items():
                rel = self._relpath(abs_path)
                if rel is None:
                    continue
                try:
                    await self._version_one(rel, abs_path, op)
                except Exception:
                    log.exception("Failed to version %s (%s)", rel, op)

    async def _resolve_attribution(self) -> tuple[str | None, str | None]:
        """Look up the most recently active session for attribution."""
        if self._session_tracker is None:
            return None, None
        try:
            sessions = await self._session_tracker.get_active_sessions()
            if sessions:
                # Most recently seen agent gets attribution
                latest = sessions[0]  # already ordered by last_seen_at DESC
                return latest.agent_name, latest.id
        except Exception:
            log.debug("Failed to resolve attribution", exc_info=True)
        return None, None

    async def _version_one(self, rel: str, abs_path: str, op: str) -> None:
        agent_id, session_id = await self._resolve_attribution()

        if op == "delete":
            await self._vs.record_delete(
                path=rel,
                agent_id=agent_id,
                session_id=session_id,
            )
            return

        path = Path(abs_path)
        if not path.exists() or not path.is_file():
            return
        try:
            content = path.read_bytes()
        except OSError:
            return
        await self._vs.record_version(
            path=rel,
            content=content,
            agent_id=agent_id,
            session_id=session_id,
        )

    async def _initial_scan(self) -> None:
        """Walk the tree once and version every file (skipping excluded)."""
        count = 0
        for dirpath, dirnames, filenames in os.walk(self._root):
            # prune excluded directories in-place
            dirnames[:] = [
                d for d in dirnames
                if not self._is_excluded(d + "/")
            ]
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel = self._relpath(abs_path)
                if rel and not self._is_excluded(rel):
                    try:
                        content = Path(abs_path).read_bytes()
                        await self._vs.record_version(
                            path=rel,
                            content=content,
                            agent_id="wawd",
                            session_id=None,
                        )
                        count += 1
                    except Exception:
                        log.debug("Skipped %s during initial scan", rel)
        log.info("Initial scan versioned %d files", count)
