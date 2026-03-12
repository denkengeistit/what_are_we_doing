"""Streamlit UI for WAWD: version history, diffs, and oracle chat."""

from __future__ import annotations

import asyncio
import difflib
import datetime
from pathlib import Path

import aiosqlite
import streamlit as st

from wawd.config import load_config, WAWDConfig
from wawd.fs.blob_store import BlobStore
from wawd.fs.version_store import VersionStore, VersionEntry
from wawd.oracle.session_tracker import SessionTracker

# ---------------------------------------------------------------------------
# Async helpers — bridge Streamlit's sync world to our async data layer
# ---------------------------------------------------------------------------


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for running async code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def run_async(coro):
    """Run an async coroutine from sync Streamlit code."""
    return _get_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Database connection (cached per Streamlit session)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_config() -> WAWDConfig:
    return load_config()


async def _open_stores(config: WAWDConfig):
    db = await aiosqlite.connect(str(config.db_path))
    blob_store = BlobStore(db, config.versioning.compression_level)
    version_store = VersionStore(db, blob_store)
    session_tracker = SessionTracker(db, config.oracle.session_timeout_minutes)
    return db, version_store, session_tracker


async def _get_oracle(config: WAWDConfig):
    """Build an Oracle instance for chat."""
    from wawd.oracle.backends.ollama import OllamaBackend
    from wawd.oracle.backends.llamacpp import LlamaCppBackend
    from wawd.oracle.backends.openai_compat import OpenAICompatBackend
    from wawd.oracle.context import ContextBuilder
    from wawd.oracle.oracle import Oracle
    from wawd.oracle.restorer import Restorer

    db, version_store, session_tracker = await _open_stores(config)

    if config.oracle.backend == "llamacpp":
        backend = LlamaCppBackend(
            base_url=config.oracle.base_url,
            timeout=config.oracle.timeout_seconds,
        )
    elif config.oracle.backend == "openai_compat":
        backend = OpenAICompatBackend(
            base_url=config.oracle.base_url,
            model=config.oracle.model,
            api_key=config.oracle.api_key or None,
            timeout=config.oracle.timeout_seconds,
        )
    else:
        backend = OllamaBackend(
            model=config.oracle.model,
            base_url=config.oracle.base_url,
            timeout=config.oracle.timeout_seconds,
        )

    context_builder = ContextBuilder(
        version_store, session_tracker,
        config.oracle.context_budget_tokens, config.oracle.history_depth,
    )
    restorer = Restorer(version_store, context_builder, backend, config.workspace.path)
    oracle = Oracle(
        version_store, session_tracker, context_builder, restorer,
        backend, config.workspace.path,
    )
    return oracle, db, backend


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_time(ts: float) -> str:
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_ago(ts: float) -> str:
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(ts)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


OP_ICONS = {"create": "🟢", "modify": "🟡", "delete": "🔴"}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_status():
    """Workspace status dashboard."""
    config = get_config()

    st.header("Workspace Status")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Workspace", config.workspace.path)
        st.metric("Backend", f"{config.oracle.backend} ({config.oracle.model})")
    with col2:
        st.metric("Database", str(config.db_path))
        st.metric("Backend URL", config.oracle.base_url)

    async def _load_status():
        db, vs, st_tracker = await _open_stores(config)
        try:
            files = await vs.get_all_current_files()
            active = await st_tracker.get_active_sessions()
            recent = await st_tracker.get_recent_sessions(limit=10)

            cursor = await db.execute("SELECT COUNT(*) FROM versions")
            row = await cursor.fetchone()
            version_count = row[0]

            cursor = await db.execute("SELECT COUNT(*) FROM blobs")
            row = await cursor.fetchone()
            blob_count = row[0]

            return files, active, recent, version_count, blob_count
        finally:
            await db.close()

    files, active, recent, version_count, blob_count = run_async(_load_status())

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Files", len(files))
    col2.metric("Versions", version_count)
    col3.metric("Blobs", blob_count)

    # Active sessions
    st.subheader("Active Sessions")
    if active:
        for s in active:
            st.markdown(
                f"**{s.agent_name}** — {s.task or 'no task'} "
                f"(since {fmt_time(s.started_at)}, last seen {fmt_ago(s.last_seen_at)})"
            )
    else:
        st.info("No active sessions.")

    # Recent completed sessions
    st.subheader("Recent Completed Sessions")
    if recent:
        for s in recent:
            with st.expander(f"{s.agent_name} — {fmt_time(s.started_at)}"):
                st.write(f"**Task:** {s.task or 'N/A'}")
                st.write(f"**Duration:** {fmt_time(s.started_at)} → {fmt_time(s.last_seen_at)}")
                if s.summary:
                    st.write(f"**Summary:** {s.summary}")
    else:
        st.info("No completed sessions yet.")

    # Current files
    st.subheader("Current Files")
    if files:
        for path, entry in sorted(files.items()):
            st.markdown(
                f"`{path}` — v{entry.id} "
                f"({entry.operation} by {entry.agent_id or 'unknown'}, "
                f"{fmt_ago(entry.timestamp)})"
            )
    else:
        st.info("No files in workspace.")


def page_history():
    """Version history browser with diffs."""
    config = get_config()

    st.header("Version History")

    async def _load_data():
        db, vs, _ = await _open_stores(config)
        try:
            paths = await vs.list_paths()
            return db, vs, paths
        except Exception:
            await db.close()
            raise

    db, vs, paths = run_async(_load_data())

    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_path = st.selectbox(
            "File", ["(all files)"] + paths, index=0
        )
    with col2:
        time_range = st.selectbox(
            "Time range",
            ["Last hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All time"],
            index=2,
        )

    import time as _time

    range_map = {
        "Last hour": 3600,
        "Last 6 hours": 21600,
        "Last 24 hours": 86400,
        "Last 7 days": 604800,
        "All time": None,
    }
    since_seconds = range_map[time_range]
    since_ts = _time.time() - since_seconds if since_seconds else None

    async def _load_history():
        try:
            if selected_path != "(all files)":
                entries = await vs.get_history(
                    selected_path, limit=200, since_timestamp=since_ts
                )
            elif since_ts:
                entries = await vs.get_changes_since(since_ts)
            else:
                entries = await vs.get_changes_since(_time.time() - 86400 * 365)
            return entries
        finally:
            await db.close()

    entries = run_async(_load_history())

    if not entries:
        st.info("No changes found for the selected filters.")
        return

    st.write(f"**{len(entries)} version(s)**")

    for entry in entries:
        icon = OP_ICONS.get(entry.operation, "⚪")
        label = (
            f"{icon} v{entry.id} — `{entry.path}` — "
            f"{entry.operation} by **{entry.agent_id or 'unknown'}** "
            f"({fmt_ago(entry.timestamp)})"
        )

        with st.expander(label):
            st.write(f"**Timestamp:** {fmt_time(entry.timestamp)}")
            if entry.intent:
                st.write(f"**Intent:** {entry.intent}")

            if entry.operation == "delete":
                st.warning("File was deleted in this version.")
            elif entry.blob_hash:
                # Show content or diff
                async def _load_diff(e=entry):
                    db2, vs2, _ = await _open_stores(config)
                    try:
                        content = await vs2.get_content(e.id)
                        new_text = content.decode(errors="replace")

                        # Try to get previous version for diff
                        history = await vs2.get_history(e.path, limit=50)
                        prev = None
                        for h in history:
                            if h.id < e.id and h.blob_hash:
                                prev = h
                                break

                        if prev:
                            old_content = await vs2.get_content(prev.id)
                            old_text = old_content.decode(errors="replace")
                            diff = difflib.unified_diff(
                                old_text.splitlines(),
                                new_text.splitlines(),
                                fromfile=f"v{prev.id}",
                                tofile=f"v{e.id}",
                                lineterm="",
                            )
                            return "diff", "\n".join(diff)
                        else:
                            return "content", new_text
                    finally:
                        await db2.close()

                kind, text = run_async(_load_diff())
                if kind == "diff" and text:
                    st.code(text, language="diff")
                elif text:
                    st.code(text, language=_guess_language(entry.path))


def page_chat():
    """Chat with the oracle."""
    config = get_config()

    st.header("Ask the Oracle")

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the workspace..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Querying oracle..."):
                async def _ask():
                    oracle, db, backend = await _get_oracle(config)
                    try:
                        result = await oracle.history(question=prompt)
                        return result
                    finally:
                        await backend.close()
                        await db.close()

                result = run_async(_ask())
                answer = result["answer"]
                changes = result.get("changes", [])

                st.markdown(answer)
                if changes:
                    st.caption(f"{len(changes)} change(s) in scope")

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _guess_language(path: str) -> str:
    """Guess syntax highlighting language from file extension."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".sh": "bash",
        ".toml": "toml",
        ".sql": "sql",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".java": "java",
    }
    suffix = Path(path).suffix.lower()
    return ext_map.get(suffix, "text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="WAWD",
        page_icon="👁️",
        layout="wide",
    )

    st.sidebar.title("👁️ WAWD")
    st.sidebar.caption("What Are We Doing")

    page = st.sidebar.radio(
        "Navigate",
        ["Status", "History", "Chat"],
        index=0,
    )

    if page == "Status":
        page_status()
    elif page == "History":
        page_history()
    elif page == "Chat":
        page_chat()


if __name__ == "__main__":
    main()
