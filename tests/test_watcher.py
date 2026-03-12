"""Tests for the watchdog-based WAWDWatcher."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from wawd.fs.watcher import WAWDWatcher


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def mock_version_store() -> MagicMock:
    vs = MagicMock()
    vs.record_version = AsyncMock(return_value=1)
    vs.record_delete = AsyncMock(return_value=1)
    return vs


@pytest_asyncio.fixture
async def watcher(tmp_workspace: Path, mock_version_store: MagicMock):
    """Start and yield a watcher, stop it on teardown."""
    w = WAWDWatcher(
        root=str(tmp_workspace),
        version_store=mock_version_store,
        exclude=["__pycache__/", "*.pyc", ".git/"],
    )
    await w.start()
    yield w
    await w.stop()


@pytest.mark.asyncio
async def test_initial_scan_versions_existing_files(
    tmp_workspace: Path, mock_version_store: MagicMock,
):
    """Files already present when the watcher starts are versioned."""
    (tmp_workspace / "hello.txt").write_text("world")
    (tmp_workspace / "sub").mkdir()
    (tmp_workspace / "sub" / "nested.txt").write_text("deep")

    w = WAWDWatcher(str(tmp_workspace), mock_version_store, exclude=[])
    await w.start()
    await w.stop()

    paths = {c.kwargs["path"] for c in mock_version_store.record_version.call_args_list}
    assert "hello.txt" in paths
    assert os.path.join("sub", "nested.txt") in paths


@pytest.mark.asyncio
async def test_file_create_detected(
    tmp_workspace: Path, mock_version_store: MagicMock, watcher: WAWDWatcher,
):
    """Creating a new file triggers a version recording."""
    (tmp_workspace / "new.txt").write_text("content")
    # Wait for debounce + a margin
    await asyncio.sleep(1.5)

    calls = mock_version_store.record_version.call_args_list
    paths = {c.kwargs["path"] for c in calls}
    assert "new.txt" in paths


@pytest.mark.asyncio
async def test_file_modify_detected(
    tmp_workspace: Path, mock_version_store: MagicMock, watcher: WAWDWatcher,
):
    """Modifying a file triggers version recording."""
    f = tmp_workspace / "edit_me.txt"
    f.write_text("v1")
    await asyncio.sleep(1.5)
    mock_version_store.record_version.reset_mock()

    f.write_text("v2")
    await asyncio.sleep(1.5)

    calls = mock_version_store.record_version.call_args_list
    paths = {c.kwargs["path"] for c in calls}
    assert "edit_me.txt" in paths


@pytest.mark.asyncio
async def test_file_delete_detected(
    tmp_workspace: Path, mock_version_store: MagicMock, watcher: WAWDWatcher,
):
    """Deleting a file calls record_delete."""
    f = tmp_workspace / "doomed.txt"
    f.write_text("bye")
    await asyncio.sleep(1.5)
    mock_version_store.record_delete.reset_mock()

    f.unlink()
    await asyncio.sleep(1.5)

    calls = mock_version_store.record_delete.call_args_list
    paths = {c.kwargs["path"] for c in calls}
    assert "doomed.txt" in paths


@pytest.mark.asyncio
async def test_excluded_files_ignored(
    tmp_workspace: Path, mock_version_store: MagicMock, watcher: WAWDWatcher,
):
    """Files matching exclude patterns are not versioned."""
    (tmp_workspace / "__pycache__").mkdir()
    (tmp_workspace / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"bytecode")
    (tmp_workspace / "regular.txt").write_text("ok")
    await asyncio.sleep(1.5)

    paths = {c.kwargs["path"] for c in mock_version_store.record_version.call_args_list}
    assert "regular.txt" in paths
    # __pycache__ contents must not appear
    assert not any("__pycache__" in p for p in paths)


@pytest.mark.asyncio
async def test_pause_prevents_versioning(
    tmp_workspace: Path, mock_version_store: MagicMock, watcher: WAWDWatcher,
):
    """While paused, new file events are dropped."""
    watcher.pause()
    (tmp_workspace / "paused.txt").write_text("ignored")
    await asyncio.sleep(1.5)

    paths = {c.kwargs["path"] for c in mock_version_store.record_version.call_args_list}
    assert "paused.txt" not in paths

    watcher.resume()


@pytest.mark.asyncio
async def test_resume_resumes_versioning(
    tmp_workspace: Path, mock_version_store: MagicMock, watcher: WAWDWatcher,
):
    """After resume, events flow again."""
    watcher.pause()
    await asyncio.sleep(0.2)
    watcher.resume()

    (tmp_workspace / "after_resume.txt").write_text("hi")
    await asyncio.sleep(1.5)

    paths = {c.kwargs["path"] for c in mock_version_store.record_version.call_args_list}
    assert "after_resume.txt" in paths


@pytest.mark.asyncio
async def test_agent_attribution(
    tmp_workspace: Path, mock_version_store: MagicMock,
):
    """Agent/session IDs are resolved from SessionTracker at version-time."""
    mock_tracker = MagicMock()
    mock_session = MagicMock()
    mock_session.agent_name = "claude"
    mock_session.id = "session-42"
    mock_tracker.get_active_sessions = AsyncMock(return_value=[mock_session])

    w = WAWDWatcher(
        str(tmp_workspace), mock_version_store,
        exclude=[], session_tracker=mock_tracker,
    )
    await w.start()

    (tmp_workspace / "attr.txt").write_text("test")
    await asyncio.sleep(1.5)

    await w.stop()

    for call in mock_version_store.record_version.call_args_list:
        if call.kwargs.get("path") == "attr.txt":
            assert call.kwargs["agent_id"] == "claude"
            assert call.kwargs["session_id"] == "session-42"
            break
    else:
        pytest.fail("attr.txt not found in record_version calls")
