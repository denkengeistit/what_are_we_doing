"""Tests for TaskStore."""

from pathlib import Path

import pytest

from wawd.tasks import TaskStore


@pytest.fixture
def tasks_md(tmp_path: Path) -> Path:
    """Create a sample TASKS.md file."""
    tasks_file = tmp_path / "TASKS.md"
    tasks_file.write_text(
        """# My Tasks

- [ ] Fix restore backup race condition 📅 2026-03-13 [assignee:: claude-code] [status:: in-progress]
- [ ] Add who_is_working tool 📅 2026-03-14 [assignee:: oz-agent-1]
- [ ] Write watcher tests 📅 2026-03-15
- [x] Initial wawd setup ✅ 2026-03-12

Some notes here.

- [ ] Unassigned task with no date
"""
    )
    return tmp_path


def test_parse_tasks(tasks_md: Path):
    """Test basic parsing of TASKS.md."""
    store = TaskStore(tasks_md)
    tasks = store.get_tasks(include_completed=True)
    
    assert len(tasks) == 5
    
    # Line 3: assigned to claude-code, in-progress
    t = tasks[0]
    assert t.line_num == 3
    assert not t.checked
    assert "Fix restore backup" in t.text
    assert t.due_date == "2026-03-13"
    assert t.assignee == "claude-code"
    assert t.status == "in-progress"
    
    # Line 4: assigned to oz-agent-1, no status
    t = tasks[1]
    assert t.assignee == "oz-agent-1"
    assert t.status is None
    
    # Line 5: unassigned
    t = tasks[2]
    assert t.assignee is None
    
    # Line 6: completed
    t = tasks[3]
    assert t.checked
    assert t.done_date == "2026-03-12"


def test_filter_by_assignee(tasks_md: Path):
    """Test filtering tasks by assignee."""
    store = TaskStore(tasks_md)
    tasks = store.get_tasks(assignee="claude-code")
    
    assert len(tasks) == 1
    assert tasks[0].line_num == 3


def test_filter_by_status(tasks_md: Path):
    """Test filtering by status."""
    store = TaskStore(tasks_md)
    tasks = store.get_tasks(status="in-progress")
    
    assert len(tasks) == 1
    assert tasks[0].assignee == "claude-code"


def test_exclude_completed_by_default(tasks_md: Path):
    """Test that completed tasks are excluded by default."""
    store = TaskStore(tasks_md)
    tasks = store.get_tasks()
    
    # Should get 4 incomplete tasks
    assert len(tasks) == 4
    assert all(not t.checked for t in tasks)


def test_claim_task(tasks_md: Path):
    """Test claiming a task."""
    store = TaskStore(tasks_md)
    
    # Claim line 5 (unassigned task)
    store.claim_task(5, "new-agent")
    
    # Verify it's now assigned
    tasks = store.get_tasks(assignee="new-agent")
    assert len(tasks) == 1
    assert tasks[0].line_num == 5
    assert tasks[0].status == "in-progress"
    
    # Check the file directly
    content = (tasks_md / "TASKS.md").read_text()
    assert "[assignee:: new-agent]" in content
    assert "[status:: in-progress]" in content


def test_reclaim_task(tasks_md: Path):
    """Test reclaiming an already-assigned task (replaces assignee)."""
    store = TaskStore(tasks_md)
    
    # Reclaim line 3 (currently claude-code)
    store.claim_task(3, "different-agent")
    
    tasks = store.get_tasks(assignee="different-agent")
    assert len(tasks) == 1
    assert tasks[0].line_num == 3
    
    # Original assignee should no longer see it
    tasks = store.get_tasks(assignee="claude-code")
    assert len(tasks) == 0


def test_complete_task(tasks_md: Path):
    """Test completing a task."""
    store = TaskStore(tasks_md)
    
    # Complete line 3
    store.claim_task(3, "test-agent")
    store.complete_task(3)
    
    # Verify it's checked and has done date
    tasks = store.get_tasks(assignee="test-agent", include_completed=True)
    assert len(tasks) == 1
    assert tasks[0].checked
    assert tasks[0].done_date is not None  # Today's date
    
    # Should not appear in incomplete tasks
    tasks = store.get_tasks(assignee="test-agent")
    assert len(tasks) == 0


def test_add_task(tasks_md: Path):
    """Test adding a new task."""
    store = TaskStore(tasks_md)
    
    store.add_task("New task", due_date="2026-03-20", assignee="bot")
    
    tasks = store.get_tasks(assignee="bot")
    assert len(tasks) == 1
    assert "New task" in tasks[0].text
    assert tasks[0].due_date == "2026-03-20"


def test_nonexistent_file(tmp_path: Path):
    """Test behavior when TASKS.md doesn't exist."""
    store = TaskStore(tmp_path)
    
    # Should return empty list, not crash
    tasks = store.get_tasks()
    assert tasks == []


def test_claim_nonexistent_line(tasks_md: Path):
    """Test claiming a line that doesn't exist."""
    store = TaskStore(tasks_md)
    
    with pytest.raises(ValueError, match="out of range"):
        store.claim_task(999, "agent")


def test_claim_completed_task_fails(tasks_md: Path):
    """Test that claiming a completed task raises an error."""
    store = TaskStore(tasks_md)
    
    with pytest.raises(ValueError, match="already completed"):
        store.claim_task(6, "agent")  # Line 6 is completed
