"""SessionTracker: implicit session tracking without explicit session tools."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Coroutine

import aiosqlite

log = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    task TEXT,
    started_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'completed')),
    summary TEXT
);
"""


@dataclass
class Session:
    """An agent session."""

    id: str
    agent_name: str
    task: str | None
    started_at: float
    last_seen_at: float
    status: str = "active"
    summary: str | None = None

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Session:
        return cls(
            id=row[0],
            agent_name=row[1],
            task=row[2],
            started_at=row[3],
            last_seen_at=row[4],
            status=row[5],
            summary=row[6],
        )


class SessionTracker:
    """Tracks agent sessions implicitly via check-in calls."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        timeout_minutes: int = 30,
        on_session_complete: Callable[[Session], Coroutine] | None = None,
    ) -> None:
        self._db = db
        self._timeout_seconds = timeout_minutes * 60
        self._on_complete = on_session_complete

    async def init_db(self) -> None:
        """Create the sessions table."""
        await self._db.executescript(SCHEMA)

    async def check_in(self, agent_name: str, task: str | None = None) -> Session:
        """Register or update an agent session. Creates new session if task changed."""
        await self.cleanup_stale()

        now = time.time()

        # Look for existing active session
        cursor = await self._db.execute(
            "SELECT id, agent_name, task, started_at, last_seen_at, status, summary "
            "FROM sessions WHERE agent_name = ? AND status = 'active'",
            (agent_name,),
        )
        row = await cursor.fetchone()

        if row is not None:
            existing = Session.from_row(row)
            # If task changed, complete old session and start new one
            if task is not None and task != existing.task:
                await self._complete_session(existing.id)
            else:
                # Update last_seen
                await self._db.execute(
                    "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                    (now, existing.id),
                )
                await self._db.commit()
                existing.last_seen_at = now
                return existing

        # Create new session
        session_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO sessions (id, agent_name, task, started_at, last_seen_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'active')",
            (session_id, agent_name, task, now, now),
        )
        await self._db.commit()

        session = Session(
            id=session_id,
            agent_name=agent_name,
            task=task,
            started_at=now,
            last_seen_at=now,
        )
        log.info("New session %s for agent '%s'", session_id[:8], agent_name)
        return session

    async def get_active_sessions(self) -> list[Session]:
        """Get all active sessions."""
        cursor = await self._db.execute(
            "SELECT id, agent_name, task, started_at, last_seen_at, status, summary "
            "FROM sessions WHERE status = 'active' ORDER BY last_seen_at DESC"
        )
        rows = await cursor.fetchall()
        return [Session.from_row(r) for r in rows]

    async def get_session(self, agent_name: str) -> Session | None:
        """Get the active session for a given agent."""
        cursor = await self._db.execute(
            "SELECT id, agent_name, task, started_at, last_seen_at, status, summary "
            "FROM sessions WHERE agent_name = ? AND status = 'active'",
            (agent_name,),
        )
        row = await cursor.fetchone()
        return Session.from_row(row) if row else None

    async def get_recent_sessions(self, limit: int = 10) -> list[Session]:
        """Get recently completed sessions."""
        cursor = await self._db.execute(
            "SELECT id, agent_name, task, started_at, last_seen_at, status, summary "
            "FROM sessions WHERE status = 'completed' ORDER BY last_seen_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [Session.from_row(r) for r in rows]

    async def cleanup_stale(self) -> list[Session]:
        """Mark stale sessions as completed. Returns the sessions that were closed."""
        cutoff = time.time() - self._timeout_seconds
        cursor = await self._db.execute(
            "SELECT id, agent_name, task, started_at, last_seen_at, status, summary "
            "FROM sessions WHERE status = 'active' AND last_seen_at < ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        stale = [Session.from_row(r) for r in rows]

        for session in stale:
            await self._complete_session(session.id)
            log.info("Stale session %s for '%s' marked completed", session.id[:8], session.agent_name)

        return stale

    async def _complete_session(self, session_id: str) -> None:
        """Mark a session as completed and trigger callback."""
        await self._db.execute(
            "UPDATE sessions SET status = 'completed' WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()

        if self._on_complete:
            cursor = await self._db.execute(
                "SELECT id, agent_name, task, started_at, last_seen_at, status, summary "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row:
                session = Session.from_row(row)
                try:
                    await self._on_complete(session)
                except Exception:
                    log.exception("Error in session complete callback for %s", session_id[:8])

    async def set_summary(self, session_id: str, summary: str) -> None:
        """Set the oracle-generated summary for a completed session."""
        await self._db.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary, session_id),
        )
        await self._db.commit()
