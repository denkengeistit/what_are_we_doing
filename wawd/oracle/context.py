"""ContextBuilder: assembles tiered context for oracle queries."""

from __future__ import annotations

import difflib
import logging
import time
from pathlib import Path

from wawd.fs.version_store import VersionEntry, VersionStore
from wawd.oracle.session_tracker import Session, SessionTracker

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a system prompt from the prompts directory."""
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text().strip()
    log.warning("Prompt file not found: %s", path)
    return ""


class ContextBuilder:
    """Assembles tiered context for oracle queries."""

    def __init__(
        self,
        version_store: VersionStore,
        session_tracker: SessionTracker,
        history_depth: int = 50,
    ) -> None:
        self._vs = version_store
        self._st = session_tracker
        self._depth = history_depth

    async def build_briefing_context(
        self,
        agent_name: str,
        task: str | None = None,
        focus: str | None = None,
    ) -> list[dict]:
        """Assemble context for a what_are_we_doing briefing."""
        messages: list[dict] = []

        # Layer 1: System prompt
        system_prompt = _load_prompt("briefing.txt")
        messages.append({"role": "system", "content": system_prompt})

        # Layer 2: Project manifest (if exists)
        # Skipped for now — would read .wawd/manifest.yaml

        # Layer 3: Active sessions
        active = await self._st.get_active_sessions()
        if active:
            session_lines = []
            for s in active:
                line = f"- {s.agent_name}: {s.task or 'no stated task'} (active since {_fmt_time(s.started_at)})"
                session_lines.append(line)
            messages.append({
                "role": "system",
                "content": "Active agent sessions:\n" + "\n".join(session_lines),
            })

        # Layer 4: Recent changes with full diffs (last 10)
        recent = await self._vs.get_changes_since(time.time() - 86400)  # last 24h
        recent = recent[: self._depth]

        if recent:
            detailed = recent[:10]
            diff_sections = []
            for entry in detailed:
                diff_text = await self._make_diff_text(entry)
                diff_sections.append(diff_text)

            diff_content = "\n\n".join(diff_sections)
            messages.append({
                "role": "system",
                "content": f"Recent changes (detailed):\n{diff_content}",
            })

            # Layer 5: Older changes with one-line summaries
            older = recent[10:50]
            if older:
                summary_lines = []
                for entry in older:
                    summary_lines.append(
                        f"- [{_fmt_time(entry.timestamp)}] {entry.operation} {entry.path}"
                        f" by {entry.agent_id or 'unknown'}"
                    )
                summary_content = "\n".join(summary_lines)
                messages.append({
                    "role": "system",
                    "content": f"Older changes (summary):\n{summary_content}",
                })

        # Layer 6: Recent session summaries
        completed = await self._st.get_recent_sessions(limit=5)
        if completed:
            sess_lines = []
            for s in completed:
                summary = s.summary or "No summary available"
                sess_lines.append(
                    f"- {s.agent_name} ({_fmt_time(s.started_at)} - {_fmt_time(s.last_seen_at)}): {summary}"
                )
            messages.append({
                "role": "system",
                "content": "Recent completed sessions:\n" + "\n".join(sess_lines),
            })

        # Layer 7: The query
        user_content = f"Agent: {agent_name}"
        if task:
            user_content += f"\nTask: {task}"
        if focus:
            user_content += f"\nFocus: {focus}"
        user_content += "\n\nPlease provide a briefing."
        messages.append({"role": "user", "content": user_content})

        return messages

    async def build_history_context(
        self,
        question: str | None = None,
        path: str | None = None,
        agent: str | None = None,
        since: float | None = None,
    ) -> list[dict]:
        """Assemble context for a what_happened query."""
        messages: list[dict] = []

        system_prompt = _load_prompt("history.txt")
        messages.append({"role": "system", "content": system_prompt})

        # Gather relevant history
        changes: list[VersionEntry] = []
        if path:
            changes = await self._vs.get_history(path, limit=self._depth, since_timestamp=since)
        elif agent:
            changes = await self._vs.get_changes_by_agent(agent, since=since)
        elif since:
            changes = await self._vs.get_changes_since(since)
        else:
            changes = await self._vs.get_changes_since(time.time() - 86400)

        changes = changes[: self._depth]

        if changes:
            diff_sections = []
            for entry in changes[:20]:
                diff_text = await self._make_diff_text(entry)
                diff_sections.append(diff_text)

            # Summarize the rest
            if len(changes) > 20:
                for entry in changes[20:]:
                    diff_sections.append(
                        f"[{_fmt_time(entry.timestamp)}] {entry.operation} {entry.path}"
                        f" by {entry.agent_id or 'unknown'}"
                    )

            history_content = "\n\n".join(diff_sections)
            messages.append({
                "role": "system",
                "content": f"Version history:\n{history_content}",
            })

        # User query
        user_content = question or "What happened recently?"
        if path:
            user_content += f"\n(Scope: {path})"
        if agent:
            user_content += f"\n(Agent: {agent})"
        messages.append({"role": "user", "content": user_content})

        return messages

    async def build_restoration_context(
        self,
        problem: str,
        scope: str | None = None,
    ) -> list[dict]:
        """Assemble context for a fix_this restoration query."""
        messages: list[dict] = []

        system_prompt = _load_prompt("restoration.txt")
        messages.append({"role": "system", "content": system_prompt})

        # Get all changes in scope from last 24 hours (full detail for restoration)
        since = time.time() - 86400
        if scope:
            # scope could be a file path or directory prefix
            changes = await self._vs.get_history(scope, limit=200, since_timestamp=since)
            if not changes:
                # Try as a prefix — get all changes and filter
                all_changes = await self._vs.get_changes_since(since)
                changes = [c for c in all_changes if c.path.startswith(scope)]
        else:
            changes = await self._vs.get_changes_since(since)

        if changes:
            diff_sections = []
            for entry in changes:
                diff_text = await self._make_diff_text(entry)
                diff_sections.append(diff_text)

            history_content = "\n\n".join(diff_sections)
            messages.append({
                "role": "system",
                "content": f"Complete version history for scope:\n{history_content}",
            })

        # Current file states
        current_files = await self._vs.get_all_current_files()
        if scope:
            current_files = {p: v for p, v in current_files.items() if p.startswith(scope)}

        if current_files:
            file_list = "\n".join(f"- {p} (v{v.id}, {v.operation})" for p, v in sorted(current_files.items()))
            messages.append({
                "role": "system",
                "content": f"Current file states:\n{file_list}",
            })

        messages.append({
            "role": "user",
            "content": f"Problem: {problem}\n\nAnalyze the version history and respond with a JSON restoration plan.",
        })

        return messages

    async def _make_diff_text(self, entry: VersionEntry) -> str:
        """Generate a human-readable diff description for a version entry."""
        header = (
            f"[v{entry.id}] [{_fmt_time(entry.timestamp)}] "
            f"{entry.operation} {entry.path} by {entry.agent_id or 'unknown'}"
        )
        if entry.intent:
            header += f" — {entry.intent}"

        if entry.operation == "delete":
            return f"{header}\n  (file deleted)"

        if entry.operation == "create" and entry.blob_hash:
            try:
                content = await self._vs.get_content(entry.id)
                lines = content.decode(errors="replace").splitlines()[:50]
                preview = "\n".join(f"  + {line}" for line in lines)
                if len(lines) == 50:
                    preview += "\n  ... (truncated)"
                return f"{header}\n{preview}"
            except Exception:
                return f"{header}\n  (content unavailable)"

        if entry.operation == "modify" and entry.blob_hash:
            try:
                new_content = await self._vs.get_content(entry.id)
                # Get previous version
                history = await self._vs.get_history(entry.path, limit=2)
                if len(history) >= 2:
                    prev = history[1]
                    if prev.blob_hash:
                        old_content = await self._vs.get_content(prev.id)
                        diff = difflib.unified_diff(
                            old_content.decode(errors="replace").splitlines(),
                            new_content.decode(errors="replace").splitlines(),
                            fromfile=f"{entry.path} (v{prev.id})",
                            tofile=f"{entry.path} (v{entry.id})",
                            lineterm="",
                        )
                        diff_text = "\n".join(list(diff)[:100])
                        if diff_text:
                            return f"{header}\n{diff_text}"
                return f"{header}\n  (content changed, diff unavailable)"
            except Exception:
                return f"{header}\n  (diff generation failed)"

        return header


def _fmt_time(ts: float) -> str:
    """Format a Unix timestamp as a human-readable string."""
    import datetime

    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
