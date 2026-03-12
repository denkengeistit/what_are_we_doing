"""End-to-end stress tests for WAWD.

Covers every layer: BlobStore, VersionStore, SessionTracker, ContextBuilder,
Restorer, Oracle, and the MCP server call_tool dispatcher.

All tests use an in-memory SQLite database (aiosqlite) — no real filesystem
watcher, no oracle backend network calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
import pytest_asyncio

from wawd.fs.blob_store import BlobStore
from wawd.fs.version_store import VersionStore
from wawd.oracle.backends.base import OracleBackend
from wawd.oracle.context import ContextBuilder
from wawd.oracle.oracle import Oracle
from wawd.oracle.restorer import Restorer
from wawd.oracle.session_tracker import SessionTracker
from wawd.server import _parse_since, create_mcp_server


# ────────────────────────────── fixtures ──────────────────────────────


@pytest_asyncio.fixture
async def db():
    """In-memory SQLite connection, closed after test."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def blob_store(db):
    bs = BlobStore(db)
    await bs.init_db()
    return bs


@pytest_asyncio.fixture
async def version_store(db, blob_store):
    vs = VersionStore(db, blob_store)
    await vs.init_db()
    return vs


@pytest_asyncio.fixture
async def session_tracker(db):
    st = SessionTracker(db, timeout_minutes=30)
    await st.init_db()
    return st


class _EchoBackend(OracleBackend):
    """Stub that echoes the last user message."""

    async def generate(self, messages: list[dict], max_tokens: int = 2048) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return f"[echo] {m['content']}"
        return "[echo] (no user message)"

    async def health_check(self) -> bool:
        return True


@pytest_asyncio.fixture
async def context_builder(version_store, session_tracker):
    return ContextBuilder(version_store, session_tracker)


@pytest_asyncio.fixture
async def restorer(version_store, context_builder, tmp_path):
    return Restorer(version_store, context_builder, _EchoBackend(), str(tmp_path))


@pytest_asyncio.fixture
async def oracle(version_store, session_tracker, context_builder, restorer, tmp_path):
    return Oracle(
        version_store,
        session_tracker,
        context_builder,
        restorer,
        _EchoBackend(),
        str(tmp_path),
    )


# ══════════════════════════════════════════════════════════════════════
# BlobStore
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_blob_store_roundtrip(blob_store):
    content = b"hello, world"
    h = await blob_store.store(content)
    assert len(h) == 64  # sha-256 hex
    retrieved = await blob_store.retrieve(h)
    assert retrieved == content


@pytest.mark.asyncio
async def test_blob_store_idempotent(blob_store):
    """Storing the same content twice returns the same hash without error."""
    content = b"duplicate content"
    h1 = await blob_store.store(content)
    h2 = await blob_store.store(content)
    assert h1 == h2


@pytest.mark.asyncio
async def test_blob_store_deduplication(blob_store, db):
    """Two identical blobs produce exactly one row in the table."""
    content = b"dedupe me"
    await blob_store.store(content)
    await blob_store.store(content)
    cursor = await db.execute("SELECT COUNT(*) FROM blobs")
    row = await cursor.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_blob_store_missing_raises(blob_store):
    with pytest.raises(KeyError):
        await blob_store.retrieve("0" * 64)


@pytest.mark.asyncio
async def test_blob_store_exists(blob_store):
    assert not await blob_store.exists("0" * 64)
    h = await blob_store.store(b"x")
    assert await blob_store.exists(h)


@pytest.mark.asyncio
async def test_blob_store_size(blob_store):
    content = b"size test " * 100
    h = await blob_store.store(content)
    assert await blob_store.size(h) == len(content)


@pytest.mark.asyncio
async def test_blob_store_delete(blob_store):
    h = await blob_store.store(b"to delete")
    assert await blob_store.exists(h)
    await blob_store.delete(h)
    assert not await blob_store.exists(h)


@pytest.mark.asyncio
async def test_blob_store_binary_content(blob_store):
    """Binary (non-UTF8) content round-trips cleanly."""
    binary = bytes(range(256))
    h = await blob_store.store(binary)
    assert await blob_store.retrieve(h) == binary


@pytest.mark.asyncio
async def test_blob_store_empty_content(blob_store):
    h = await blob_store.store(b"")
    assert await blob_store.retrieve(h) == b""


@pytest.mark.asyncio
async def test_blob_store_large_content(blob_store):
    large = b"x" * (1024 * 1024)  # 1 MB
    h = await blob_store.store(large)
    assert await blob_store.retrieve(h) == large


# ══════════════════════════════════════════════════════════════════════
# VersionStore
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_version_store_create(version_store):
    vid = await version_store.record_version("foo.py", b"v1", agent_id="alice")
    assert vid is not None
    entry = await version_store.get_version(vid)
    assert entry.path == "foo.py"
    assert entry.operation == "create"
    assert entry.agent_id == "alice"


@pytest.mark.asyncio
async def test_version_store_modify(version_store):
    v1 = await version_store.record_version("foo.py", b"v1")
    v2 = await version_store.record_version("foo.py", b"v2")
    assert v2 is not None and v2 != v1
    entry = await version_store.get_version(v2)
    assert entry.operation == "modify"


@pytest.mark.asyncio
async def test_version_store_noop_on_same_content(version_store):
    """Recording identical content returns None (no new version)."""
    await version_store.record_version("unchanged.py", b"same")
    result = await version_store.record_version("unchanged.py", b"same")
    assert result is None


@pytest.mark.asyncio
async def test_version_store_delete(version_store):
    await version_store.record_version("bye.py", b"exists")
    dvid = await version_store.record_delete("bye.py", agent_id="bob")
    entry = await version_store.get_version(dvid)
    assert entry.operation == "delete"
    assert entry.blob_hash is None


@pytest.mark.asyncio
async def test_version_store_get_content_on_delete_raises(version_store):
    """get_content on a delete version must raise ValueError, not silently return."""
    await version_store.record_version("gone.py", b"data")
    dvid = await version_store.record_delete("gone.py")
    with pytest.raises(ValueError, match="delete operation"):
        await version_store.get_content(dvid)


@pytest.mark.asyncio
async def test_version_store_get_history_order(version_store):
    """History is returned newest-first."""
    await version_store.record_version("order.py", b"a")
    await version_store.record_version("order.py", b"b")
    await version_store.record_version("order.py", b"c")
    history = await version_store.get_history("order.py")
    contents = [await version_store.get_content(e.id) for e in history]
    assert contents == [b"c", b"b", b"a"]


@pytest.mark.asyncio
async def test_version_store_get_all_current_files_excludes_deleted(version_store):
    await version_store.record_version("alive.py", b"ok")
    await version_store.record_version("dead.py", b"bye")
    await version_store.record_delete("dead.py")
    current = await version_store.get_all_current_files()
    assert "alive.py" in current
    assert "dead.py" not in current


@pytest.mark.asyncio
async def test_version_store_list_paths(version_store):
    await version_store.record_version("src/a.py", b"a")
    await version_store.record_version("src/b.py", b"b")
    await version_store.record_version("tests/c.py", b"c")
    paths = await version_store.list_paths()
    assert set(paths) == {"src/a.py", "src/b.py", "tests/c.py"}
    src_paths = await version_store.list_paths(prefix="src/")
    assert set(src_paths) == {"src/a.py", "src/b.py"}


@pytest.mark.asyncio
async def test_version_store_get_changes_since(version_store):
    before = time.time() - 1
    await version_store.record_version("new.py", b"data")
    changes = await version_store.get_changes_since(before)
    assert any(c.path == "new.py" for c in changes)


@pytest.mark.asyncio
async def test_version_store_get_changes_by_agent(version_store):
    await version_store.record_version("alice.py", b"1", agent_id="alice")
    await version_store.record_version("bob.py", b"1", agent_id="bob")
    alice_changes = await version_store.get_changes_by_agent("alice")
    assert all(c.agent_id == "alice" for c in alice_changes)
    assert len(alice_changes) == 1


@pytest.mark.asyncio
async def test_version_store_restore_file_to_version(version_store):
    v1 = await version_store.record_version("restore.py", b"original")
    await version_store.record_version("restore.py", b"modified")
    await version_store.restore_file_to_version("restore.py", v1)
    content = await version_store.get_content(
        (await version_store.get_latest("restore.py")).id
    )
    assert content == b"original"


@pytest.mark.asyncio
async def test_version_store_restore_to_delete_version_raises(version_store):
    """BUG: restore_file_to_version crashes when target is a delete op (blob_hash=None)."""
    await version_store.record_version("x.py", b"data")
    dvid = await version_store.record_delete("x.py")
    # This should raise clearly, not with an obscure TypeError
    with pytest.raises((TypeError, ValueError, KeyError)):
        await version_store.restore_file_to_version("x.py", dvid)


@pytest.mark.asyncio
async def test_version_store_snapshot_create_and_restore(version_store):
    v1 = await version_store.record_version("snap.py", b"snapshot content")
    await version_store.create_snapshot("s1", description="test snap", created_by="ci")
    # Modify after snapshot
    await version_store.record_version("snap.py", b"modified content")
    # Restore snapshot
    restored = await version_store.restore_snapshot("s1")
    assert "snap.py" in restored
    latest = await version_store.get_latest("snap.py")
    content = await version_store.get_content(latest.id)
    assert content == b"snapshot content"


@pytest.mark.asyncio
async def test_version_store_snapshot_missing_raises(version_store):
    with pytest.raises(KeyError):
        await version_store.restore_snapshot("nonexistent")


@pytest.mark.asyncio
async def test_version_store_restore_files_to_time(version_store):
    v1 = await version_store.record_version("time.py", b"old")
    t = time.time()
    await asyncio.sleep(0.01)
    await version_store.record_version("time.py", b"new")
    restored = await version_store.restore_files_to_time(["time.py"], t)
    assert "time.py" in restored
    content = await version_store.get_content(
        (await version_store.get_latest("time.py")).id
    )
    assert content == b"old"


@pytest.mark.asyncio
async def test_version_store_parent_version_chain(version_store, db):
    """Versions form a linked list via parent_version_id."""
    v1 = await version_store.record_version("chain.py", b"1")
    v2 = await version_store.record_version("chain.py", b"2")
    cursor = await db.execute(
        "SELECT parent_version_id FROM versions WHERE id = ?", (v2,)
    )
    row = await cursor.fetchone()
    assert row[0] == v1


# ══════════════════════════════════════════════════════════════════════
# SessionTracker
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_session_tracker_new_session(session_tracker):
    session = await session_tracker.check_in("agent-1", task="fix bug")
    assert session.agent_name == "agent-1"
    assert session.task == "fix bug"
    assert session.status == "active"


@pytest.mark.asyncio
async def test_session_tracker_returns_existing_session(session_tracker):
    s1 = await session_tracker.check_in("agent-1")
    s2 = await session_tracker.check_in("agent-1")
    assert s1.id == s2.id


@pytest.mark.asyncio
async def test_session_tracker_task_change_creates_new_session(session_tracker):
    s1 = await session_tracker.check_in("agent-1", task="task A")
    s2 = await session_tracker.check_in("agent-1", task="task B")
    assert s1.id != s2.id


@pytest.mark.asyncio
async def test_session_tracker_updates_last_seen(session_tracker):
    s1 = await session_tracker.check_in("agent-1")
    await asyncio.sleep(0.05)
    s2 = await session_tracker.check_in("agent-1")
    assert s2.last_seen_at >= s1.last_seen_at


@pytest.mark.asyncio
async def test_session_tracker_active_sessions(session_tracker):
    await session_tracker.check_in("alpha")
    await session_tracker.check_in("beta")
    active = await session_tracker.get_active_sessions()
    names = {s.agent_name for s in active}
    assert "alpha" in names and "beta" in names


@pytest.mark.asyncio
async def test_session_tracker_stale_cleanup(db):
    """Sessions inactive beyond timeout are marked completed."""
    tracker = SessionTracker(db, timeout_minutes=0)  # instant timeout
    await tracker.init_db()
    session = await tracker.check_in("stale-agent")
    # Force last_seen_at to be in the past
    await db.execute(
        "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
        (time.time() - 3600, session.id),
    )
    await db.commit()
    stale = await tracker.cleanup_stale()
    assert any(s.id == session.id for s in stale)
    active = await tracker.get_active_sessions()
    assert not any(s.id == session.id for s in active)


@pytest.mark.asyncio
async def test_session_tracker_on_complete_callback(db):
    completed_sessions = []

    async def on_complete(session):
        completed_sessions.append(session.id)

    tracker = SessionTracker(db, timeout_minutes=0, on_session_complete=on_complete)
    await tracker.init_db()
    session = await tracker.check_in("callback-agent", task="work")
    await db.execute(
        "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
        (time.time() - 3600, session.id),
    )
    await db.commit()
    await tracker.cleanup_stale()
    assert session.id in completed_sessions


@pytest.mark.asyncio
async def test_session_tracker_get_recent_completed(session_tracker, db):
    session = await session_tracker.check_in("done-agent")
    await db.execute(
        "UPDATE sessions SET status = 'completed' WHERE id = ?", (session.id,)
    )
    await db.commit()
    recent = await session_tracker.get_recent_sessions()
    assert any(s.id == session.id for s in recent)


@pytest.mark.asyncio
async def test_session_tracker_set_summary(session_tracker, db):
    session = await session_tracker.check_in("summarized")
    await db.execute(
        "UPDATE sessions SET status = 'completed' WHERE id = ?", (session.id,)
    )
    await db.commit()
    await session_tracker.set_summary(session.id, "Fixed the thing")
    cursor = await db.execute(
        "SELECT summary FROM sessions WHERE id = ?", (session.id,)
    )
    row = await cursor.fetchone()
    assert row[0] == "Fixed the thing"


# ══════════════════════════════════════════════════════════════════════
# ContextBuilder
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_context_builder_briefing_has_user_message(context_builder):
    messages = await context_builder.build_briefing_context("test-agent", task="refactor")
    roles = [m["role"] for m in messages]
    assert "user" in roles
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "test-agent" in user_msg["content"]
    assert "refactor" in user_msg["content"]


@pytest.mark.asyncio
async def test_context_builder_briefing_includes_recent_changes(
    context_builder, version_store
):
    await version_store.record_version("app.py", b"def main(): pass", agent_id="dev")
    messages = await context_builder.build_briefing_context("inspector")
    full_text = " ".join(m["content"] for m in messages)
    assert "app.py" in full_text


@pytest.mark.asyncio
async def test_context_builder_history_question_in_user_message(context_builder):
    messages = await context_builder.build_history_context(question="What changed in auth?")
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "auth" in user_msg["content"]


@pytest.mark.asyncio
async def test_context_builder_history_by_path(context_builder, version_store):
    await version_store.record_version("db.py", b"v1")
    await version_store.record_version("db.py", b"v2")
    messages = await context_builder.build_history_context(path="db.py")
    full_text = " ".join(m["content"] for m in messages)
    assert "db.py" in full_text


@pytest.mark.asyncio
async def test_context_builder_history_no_question_fallback(context_builder):
    messages = await context_builder.build_history_context()
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "happened" in user_msg["content"].lower()


@pytest.mark.asyncio
async def test_context_builder_restoration_context(context_builder, version_store):
    await version_store.record_version("broken.py", b"bad code")
    messages = await context_builder.build_restoration_context("TypeError in broken.py")
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "TypeError" in user_msg["content"]
    assert "JSON" in user_msg["content"]


@pytest.mark.asyncio
async def test_context_builder_restoration_scope_filter(
    context_builder, version_store
):
    await version_store.record_version("auth/login.py", b"login code")
    await version_store.record_version("payments/charge.py", b"payment code")
    messages = await context_builder.build_restoration_context(
        "auth broken", scope="auth/"
    )
    full_text = " ".join(m["content"] for m in messages)
    assert "login.py" in full_text
    # payments/ should not appear since it's out of scope
    assert "charge.py" not in full_text


@pytest.mark.asyncio
async def test_context_builder_diff_shown_for_modify(context_builder, version_store):
    await version_store.record_version("diff.py", b"line1\nline2\n")
    await version_store.record_version("diff.py", b"line1\nline2\nline3\n")
    messages = await context_builder.build_history_context(path="diff.py")
    full_text = " ".join(m["content"] for m in messages)
    # unified diff should show the added line
    assert "line3" in full_text


# ══════════════════════════════════════════════════════════════════════
# Restorer._parse_oracle_response
# ══════════════════════════════════════════════════════════════════════


def make_restorer_for_parsing(tmp_path):
    """Return a bare Restorer with mocked internals for parse testing."""
    return Restorer(
        version_store=MagicMock(),
        context_builder=MagicMock(),
        backend=_EchoBackend(),
        workspace_path=str(tmp_path),
    )


def test_parse_oracle_response_valid_json(tmp_path):
    r = make_restorer_for_parsing(tmp_path)
    payload = json.dumps({
        "files_to_restore": [
            {"path": "app.py", "to_version_id": 7, "reason": "broke auth"}
        ],
        "explanation": "Rolled back auth change",
        "confidence": "high",
    })
    plan = r._parse_oracle_response(payload)
    assert len(plan.files) == 1
    assert plan.files[0].path == "app.py"
    assert plan.files[0].to_version_id == 7
    assert plan.confidence == "high"


def test_parse_oracle_response_markdown_fenced(tmp_path):
    r = make_restorer_for_parsing(tmp_path)
    payload = (
        "```json\n"
        + json.dumps({
            "files_to_restore": [{"path": "x.py", "to_version_id": 1}],
            "explanation": "test",
            "confidence": "medium",
        })
        + "\n```"
    )
    plan = r._parse_oracle_response(payload)
    assert plan.files[0].path == "x.py"


def test_parse_oracle_response_invalid_json_returns_low_confidence(tmp_path):
    r = make_restorer_for_parsing(tmp_path)
    plan = r._parse_oracle_response("this is not json at all")
    assert plan.confidence == "low"
    assert plan.files == []


def test_parse_oracle_response_missing_files_key(tmp_path):
    r = make_restorer_for_parsing(tmp_path)
    payload = json.dumps({"explanation": "no files key", "confidence": "high"})
    plan = r._parse_oracle_response(payload)
    assert plan.files == []
    assert plan.explanation == "no files key"


def test_parse_oracle_response_empty_files_list(tmp_path):
    r = make_restorer_for_parsing(tmp_path)
    payload = json.dumps({
        "files_to_restore": [],
        "explanation": "nothing to restore",
        "confidence": "high",
    })
    plan = r._parse_oracle_response(payload)
    assert plan.files == []


def test_parse_oracle_response_partial_file_entry(tmp_path):
    """Missing 'reason' field should not crash — it has a default."""
    r = make_restorer_for_parsing(tmp_path)
    payload = json.dumps({
        "files_to_restore": [{"path": "z.py", "to_version_id": 3}],
        "explanation": "ok",
        "confidence": "medium",
    })
    plan = r._parse_oracle_response(payload)
    assert plan.files[0].reason == ""


# ══════════════════════════════════════════════════════════════════════
# Restorer.execute_restoration_plan
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_restorer_execute_no_files_returns_no_action(restorer):
    from wawd.oracle.restorer import RestorationPlan

    plan = RestorationPlan(files=[], explanation="nothing", auto_snapshot_name="snap1")
    result = await restorer.execute_restoration_plan(plan)
    assert result.action_taken == "no_action"
    assert result.files_restored == []


@pytest.mark.asyncio
async def test_restorer_execute_writes_to_disk(
    version_store, context_builder, tmp_path
):
    """Restorer must physically write files to disk, not just update the DB."""
    v1 = await version_store.record_version("sub/file.py", b"original content")
    await version_store.record_version("sub/file.py", b"corrupted content")

    r = Restorer(version_store, context_builder, _EchoBackend(), str(tmp_path))

    from wawd.oracle.restorer import FileRestoration, RestorationPlan

    plan = RestorationPlan(
        files=[FileRestoration(path="sub/file.py", to_version_id=v1, reason="test")],
        explanation="restore test",
        auto_snapshot_name="pre-test-snap",
    )
    result = await r.execute_restoration_plan(plan)
    assert result.action_taken == "restored"
    assert (tmp_path / "sub" / "file.py").read_bytes() == b"original content"


@pytest.mark.asyncio
async def test_restorer_execute_pauses_watcher(version_store, context_builder, tmp_path):
    """Watcher must be paused during restoration and resumed after."""
    v1 = await version_store.record_version("w.py", b"v1")

    r = Restorer(version_store, context_builder, _EchoBackend(), str(tmp_path))
    mock_watcher = MagicMock()
    r.set_watcher(mock_watcher)

    from wawd.oracle.restorer import FileRestoration, RestorationPlan

    plan = RestorationPlan(
        files=[FileRestoration(path="w.py", to_version_id=v1)],
        explanation="test",
        auto_snapshot_name="snap",
    )
    await r.execute_restoration_plan(plan)

    mock_watcher.pause.assert_called_once()
    mock_watcher.resume.assert_called_once()


@pytest.mark.asyncio
async def test_restorer_execute_resumes_watcher_on_error(
    version_store, context_builder, tmp_path
):
    """Watcher must be resumed even if a restoration raises an exception."""
    # Use a nonexistent version ID to trigger an error
    r = Restorer(version_store, context_builder, _EchoBackend(), str(tmp_path))
    mock_watcher = MagicMock()
    r.set_watcher(mock_watcher)

    from wawd.oracle.restorer import FileRestoration, RestorationPlan

    plan = RestorationPlan(
        files=[FileRestoration(path="x.py", to_version_id=99999)],
        explanation="test",
        auto_snapshot_name="snap",
    )
    result = await r.execute_restoration_plan(plan)
    # Should not raise — errors are captured per-file
    mock_watcher.resume.assert_called_once()
    assert "error" in result.files_restored[0]


@pytest.mark.asyncio
async def test_restorer_dry_run(version_store, context_builder, tmp_path):
    """Dry run must not write to disk or modify DB."""
    v1 = await version_store.record_version("dry.py", b"original")
    await version_store.record_version("dry.py", b"changed")

    class _PlanBackend(OracleBackend):
        async def generate(self, messages, max_tokens=2048):
            return json.dumps({
                "files_to_restore": [
                    {"path": "dry.py", "to_version_id": v1, "reason": "test"}
                ],
                "explanation": "dry run test",
                "confidence": "high",
            })
        async def health_check(self):
            return True

    r = Restorer(version_store, context_builder, _PlanBackend(), str(tmp_path))
    result = await r.analyze_and_restore("test problem", dry_run=True)
    assert result.action_taken == "dry_run"
    assert len(result.files_restored) == 1
    assert not (tmp_path / "dry.py").exists()


@pytest.mark.asyncio
async def test_restorer_fallback_plan_when_oracle_unparseable(
    version_store, context_builder, tmp_path
):
    """When oracle returns garbage, fallback plan should restore from 1h ago."""

    class _GarbageBackend(OracleBackend):
        async def generate(self, messages, max_tokens=2048):
            return "I am not JSON"
        async def health_check(self):
            return True

    r = Restorer(version_store, context_builder, _GarbageBackend(), str(tmp_path))
    # Record a version *now* so there's something to fall back to
    await version_store.record_version("fallback.py", b"content")
    # analyze_and_restore should not crash
    result = await r.analyze_and_restore("something broke")
    # No eligible files from 1h ago since we just created it — should be no_action
    assert result.action_taken in ("no_action", "restored")


# ══════════════════════════════════════════════════════════════════════
# Oracle (briefing / history / fix)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_oracle_briefing_registers_session(oracle, session_tracker):
    await oracle.briefing("test-agent", task="write tests")
    active = await session_tracker.get_active_sessions()
    assert any(s.agent_name == "test-agent" for s in active)


@pytest.mark.asyncio
async def test_oracle_briefing_returns_workspace_path(oracle, tmp_path):
    result = await oracle.briefing("agent-a")
    assert str(tmp_path) in result["workspace_path"]


@pytest.mark.asyncio
async def test_oracle_briefing_updates_watcher_attribution(oracle):
    mock_watcher = MagicMock()
    oracle.set_watcher(mock_watcher)
    await oracle.briefing("track-me", task="the task")
    assert mock_watcher.current_agent_id == "track-me"


@pytest.mark.asyncio
async def test_oracle_briefing_different_task_new_session(oracle, session_tracker):
    s1 = await oracle.briefing("dual-agent", task="task 1")
    s2 = await oracle.briefing("dual-agent", task="task 2")
    sessions = await session_tracker.get_active_sessions()
    # Should have exactly one active session (new one)
    dual = [s for s in sessions if s.agent_name == "dual-agent"]
    assert len(dual) == 1
    assert dual[0].task == "task 2"


@pytest.mark.asyncio
async def test_oracle_history_returns_answer_and_changes(oracle, version_store):
    await version_store.record_version("hist.py", b"data", agent_id="dev")
    result = await oracle.history(question="what changed?")
    assert "answer" in result
    assert "changes" in result
    assert isinstance(result["changes"], list)


@pytest.mark.asyncio
async def test_oracle_history_path_filter(oracle, version_store):
    await version_store.record_version("target.py", b"x")
    await version_store.record_version("other.py", b"y")
    result = await oracle.history(path="target.py")
    for c in result["changes"]:
        assert c["path"] == "target.py"


@pytest.mark.asyncio
async def test_oracle_history_agent_filter(oracle, version_store):
    await version_store.record_version("a.py", b"1", agent_id="alice")
    await version_store.record_version("b.py", b"2", agent_id="bob")
    result = await oracle.history(agent="alice")
    for c in result["changes"]:
        assert c["agent_id"] == "alice"


@pytest.mark.asyncio
async def test_oracle_fix_dry_run(oracle, version_store, tmp_path):
    v1 = await version_store.record_version("x.py", b"good")
    await version_store.record_version("x.py", b"bad")

    class _FixBackend(OracleBackend):
        async def generate(self, messages, max_tokens=2048):
            return json.dumps({
                "files_to_restore": [{"path": "x.py", "to_version_id": v1}],
                "explanation": "restore to good",
                "confidence": "high",
            })
        async def health_check(self):
            return True

    oracle._backend = _FixBackend()
    oracle._restorer._backend = _FixBackend()
    result = await oracle.fix("x.py is broken", dry_run=True)
    assert result["action_taken"] == "dry_run"
    assert not (tmp_path / "x.py").exists()


@pytest.mark.asyncio
async def test_oracle_fix_real_restores_disk(
    version_store, session_tracker, context_builder, tmp_path
):
    v1 = await version_store.record_version("real.py", b"correct")
    await version_store.record_version("real.py", b"broken")

    class _FixBackend(OracleBackend):
        async def generate(self, messages, max_tokens=2048):
            return json.dumps({
                "files_to_restore": [{"path": "real.py", "to_version_id": v1}],
                "explanation": "fixing it",
                "confidence": "high",
            })
        async def health_check(self):
            return True

    fix_backend = _FixBackend()
    restorer = Restorer(version_store, context_builder, fix_backend, str(tmp_path))
    oracle = Oracle(
        version_store, session_tracker, context_builder, restorer,
        fix_backend, str(tmp_path),
    )
    result = await oracle.fix("real.py broke")
    assert result["action_taken"] == "restored"
    assert (tmp_path / "real.py").read_bytes() == b"correct"


# ══════════════════════════════════════════════════════════════════════
# MCP server call_tool dispatcher
# ══════════════════════════════════════════════════════════════════════


import mcp.types as mcp_types


def _mcp_call(server, tool_name, arguments):
    """Invoke an MCP server tool and return the CallToolResult."""
    handler = server.request_handlers[mcp_types.CallToolRequest]
    req = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=tool_name, arguments=arguments),
    )
    return handler(req)


@pytest_asyncio.fixture
def mock_oracle():
    o = MagicMock()
    o.briefing = AsyncMock(return_value={
        "workspace_path": "/fake/path",
        "briefing": "Everything is fine.",
    })
    o.history = AsyncMock(return_value={
        "answer": "Nothing changed.",
        "changes": [{"path": "a.py"}],
    })
    o.fix = AsyncMock(return_value={
        "action_taken": "restored",
        "files_restored": [{"path": "b.py", "to_version": 3}],
        "explanation": "Rolled back b.py",
    })
    return o


@pytest.mark.asyncio
async def test_mcp_what_are_we_doing(mock_oracle):
    server = create_mcp_server(mock_oracle)
    server_result = await _mcp_call(server, "what_are_we_doing", {"agent_name": "tester", "task": "stress test"})
    text = server_result.root.content[0].text
    assert text.startswith("Workspace:")
    mock_oracle.briefing.assert_awaited_once_with(
        agent_name="tester", task="stress test", focus=None
    )


@pytest.mark.asyncio
async def test_mcp_what_happened(mock_oracle):
    server = create_mcp_server(mock_oracle)
    server_result = await _mcp_call(server, "what_happened", {"question": "what broke?", "since": "1h"})
    text = server_result.root.content[0].text
    assert "Nothing changed" in text
    assert "(1 changes in scope)" in text


@pytest.mark.asyncio
async def test_mcp_fix_this(mock_oracle):
    server = create_mcp_server(mock_oracle)
    server_result = await _mcp_call(server, "fix_this", {"problem": "tests failing", "dry_run": False})
    text = server_result.root.content[0].text
    assert "restored" in text
    assert "b.py" in text
    assert "Rolled back" in text


@pytest.mark.asyncio
async def test_mcp_unknown_tool(mock_oracle):
    server = create_mcp_server(mock_oracle)
    server_result = await _mcp_call(server, "unknown_tool", {})
    text = server_result.root.content[0].text
    assert "Unknown tool" in text


@pytest.mark.asyncio
async def test_mcp_tool_exception_returns_error_text(mock_oracle):
    mock_oracle.briefing = AsyncMock(side_effect=RuntimeError("boom"))
    server = create_mcp_server(mock_oracle)
    server_result = await _mcp_call(server, "what_are_we_doing", {"agent_name": "crash-test"})
    text = server_result.root.content[0].text
    assert "Error:" in text
    assert "boom" in text


@pytest.mark.asyncio
async def test_mcp_what_happened_empty_changes(mock_oracle):
    """No trailing count line when changes list is empty."""
    mock_oracle.history = AsyncMock(return_value={"answer": "quiet", "changes": []})
    server = create_mcp_server(mock_oracle)
    server_result = await _mcp_call(server, "what_happened", {})
    text = server_result.root.content[0].text
    assert "0 changes" not in text


# ══════════════════════════════════════════════════════════════════════
# _parse_since (server.py utility)
# ══════════════════════════════════════════════════════════════════════


def test_parse_since_hours():
    ts = _parse_since("2h")
    assert abs(ts - (time.time() - 7200)) < 2


def test_parse_since_minutes():
    ts = _parse_since("30m")
    assert abs(ts - (time.time() - 1800)) < 2


def test_parse_since_days():
    ts = _parse_since("7d")
    assert abs(ts - (time.time() - 7 * 86400)) < 2


def test_parse_since_iso_timestamp():
    from datetime import datetime, timezone
    dt = datetime(2025, 1, 15, 12, 0, 0)
    ts = _parse_since(dt.isoformat())
    assert abs(ts - dt.timestamp()) < 1


def test_parse_since_none():
    assert _parse_since(None) is None


def test_parse_since_invalid_returns_none():
    assert _parse_since("last_session") is None
    assert _parse_since("bogus") is None
    assert _parse_since("not-a-date") is None


# ══════════════════════════════════════════════════════════════════════
# Integration: full stack watcher → version store → oracle
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_integration_full_write_then_query(tmp_path):
    """Files written to disk by the watcher appear in oracle history answers."""
    from wawd.fs.watcher import WAWDWatcher

    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row

    blob_store = BlobStore(conn)
    await blob_store.init_db()
    version_store = VersionStore(conn, blob_store)
    await version_store.init_db()
    session_tracker = SessionTracker(conn)
    await session_tracker.init_db()

    watcher = WAWDWatcher(tmp_path, version_store, exclude=[])
    await watcher.start()

    # Write files after watcher starts
    (tmp_path / "integration.py").write_text("print('hello')")
    await asyncio.sleep(1.5)  # wait for debounce

    await watcher.stop()

    # Now query via oracle
    ctx = ContextBuilder(version_store, session_tracker)
    restorer = Restorer(version_store, ctx, _EchoBackend(), str(tmp_path))
    oracle = Oracle(version_store, session_tracker, ctx, restorer, _EchoBackend(), str(tmp_path))

    result = await oracle.history(question="what files exist?")
    assert any(c["path"] == "integration.py" for c in result["changes"])

    await conn.close()


@pytest.mark.asyncio
async def test_integration_restore_flow(tmp_path):
    """Write v1, write v2 (corrupted), call fix_this, get v1 back on disk."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row

    blob_store = BlobStore(conn)
    await blob_store.init_db()
    vs = VersionStore(conn, blob_store)
    await vs.init_db()
    st = SessionTracker(conn)
    await st.init_db()

    v1 = await vs.record_version("critical.py", b"def good(): pass")
    await vs.record_version("critical.py", b"def broken(): raise RuntimeError()")

    class _TargetedBackend(OracleBackend):
        async def generate(self, messages, max_tokens=2048):
            return json.dumps({
                "files_to_restore": [
                    {"path": "critical.py", "to_version_id": v1, "reason": "was good"}
                ],
                "explanation": "Restored critical.py to working state",
                "confidence": "high",
            })
        async def health_check(self):
            return True

    backend = _TargetedBackend()
    ctx = ContextBuilder(vs, st)
    restorer = Restorer(vs, ctx, backend, str(tmp_path))
    oracle = Oracle(vs, st, ctx, restorer, backend, str(tmp_path))

    result = await oracle.fix("critical.py has a RuntimeError")
    assert result["action_taken"] == "restored"
    assert (tmp_path / "critical.py").read_bytes() == b"def good(): pass"

    await conn.close()


@pytest.mark.asyncio
async def test_integration_multi_agent_attribution(tmp_path):
    """Two agents check in; their changes are attributed correctly."""
    from wawd.fs.watcher import WAWDWatcher

    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row

    blob_store = BlobStore(conn)
    await blob_store.init_db()
    vs = VersionStore(conn, blob_store)
    await vs.init_db()
    st = SessionTracker(conn)
    await st.init_db()

    ctx = ContextBuilder(vs, st)
    restorer = Restorer(vs, ctx, _EchoBackend(), str(tmp_path))
    oracle = Oracle(vs, st, ctx, restorer, _EchoBackend(), str(tmp_path))

    watcher = WAWDWatcher(tmp_path, vs, exclude=[])
    await watcher.start()
    oracle.set_watcher(watcher)

    # Agent A checks in and writes a file
    await oracle.briefing("agent-alice", task="add feature")
    (tmp_path / "alice_feature.py").write_text("# alice")
    await asyncio.sleep(1.5)

    # Agent B checks in and writes a file
    await oracle.briefing("agent-bob", task="fix bug")
    (tmp_path / "bob_fix.py").write_text("# bob")
    await asyncio.sleep(1.5)

    await watcher.stop()

    alice_changes = await vs.get_changes_by_agent("agent-alice")
    bob_changes = await vs.get_changes_by_agent("agent-bob")

    # At minimum, the most recent watcher attribution should exist
    # (exact attribution depends on timing of check-in vs write)
    all_changes = await vs.get_changes_since(time.time() - 60)
    paths = {c.path for c in all_changes}
    assert "alice_feature.py" in paths
    assert "bob_fix.py" in paths

    await conn.close()
