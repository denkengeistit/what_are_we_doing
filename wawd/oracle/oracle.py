"""Oracle: main interface for all oracle operations."""

from __future__ import annotations

import logging
import time

from wawd.fs.version_store import VersionStore
from wawd.oracle.backends.base import OracleBackend
from wawd.oracle.context import ContextBuilder
from wawd.oracle.restorer import Restorer
from wawd.oracle.session_tracker import SessionTracker

log = logging.getLogger(__name__)


class Oracle:
    """Main oracle interface. The MCP server calls this."""

    def __init__(
        self,
        version_store: VersionStore,
        session_tracker: SessionTracker,
        context_builder: ContextBuilder,
        restorer: Restorer,
        backend: OracleBackend,
        workspace_path: str,
        watcher=None,
    ) -> None:
        self._vs = version_store
        self._st = session_tracker
        self._ctx = context_builder
        self._restorer = restorer
        self._backend = backend
        self._workspace = workspace_path
        self._watcher = watcher  # WAWDWatcher instance, set after start

    def set_watcher(self, watcher) -> None:
        """Set the watcher instance (for agent/session tracking)."""
        self._watcher = watcher

    async def briefing(
        self,
        agent_name: str,
        task: str | None = None,
        focus: str | None = None,
    ) -> dict:
        """Handle what_are_we_doing: register session + generate briefing."""
        # Register / update session
        session = await self._st.check_in(agent_name, task)

        # Update watcher with current agent info
        if self._watcher is not None:
            self._watcher.current_agent_id = agent_name
            self._watcher.current_session_id = session.id

        # Build context and query oracle
        messages = await self._ctx.build_briefing_context(agent_name, task, focus)
        response = await self._backend.generate(messages)

        return {
            "workspace_path": self._workspace,
            "briefing": response,
        }

    async def history(
        self,
        question: str | None = None,
        path: str | None = None,
        agent: str | None = None,
        since: float | None = None,
    ) -> dict:
        """Handle what_happened: query history."""
        messages = await self._ctx.build_history_context(question, path, agent, since)
        response = await self._backend.generate(messages)

        # Also return structured change data
        changes = []
        if path:
            entries = await self._vs.get_history(path, limit=50, since_timestamp=since)
        elif agent:
            entries = await self._vs.get_changes_by_agent(agent, since=since)
        elif since:
            entries = await self._vs.get_changes_since(since)
        else:
            entries = await self._vs.get_changes_since(time.time() - 86400)

        for e in entries[:50]:
            changes.append({
                "version_id": e.id,
                "path": e.path,
                "operation": e.operation,
                "agent_id": e.agent_id,
                "timestamp": e.timestamp,
                "intent": e.intent,
            })

        return {
            "answer": response,
            "changes": changes,
        }

    async def fix(
        self,
        problem: str,
        scope: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Handle fix_this: analyze and restore."""
        result = await self._restorer.analyze_and_restore(problem, scope, dry_run)

        return {
            "action_taken": result.action_taken,
            "files_restored": result.files_restored,
            "explanation": result.explanation,
        }
