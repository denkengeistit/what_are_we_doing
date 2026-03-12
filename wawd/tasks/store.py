"""TaskStore: parse and manipulate TASKS.md in Obsidian Tasks format.

Supports:
- Obsidian Tasks plugin syntax: - [ ] / - [x] with 📅 YYYY-MM-DD and ✅ YYYY-MM-DD
- Dataview inline metadata: [assignee:: name] [status:: state]
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Regex patterns
TASK_LINE = re.compile(
    r"^(?P<indent>\s*)-\s+\[(?P<checked>[ xX])\]\s+"
    r"(?P<text>.+?)$"
)
DUE_DATE = re.compile(r"📅\s*(\d{4}-\d{2}-\d{2})")
DONE_DATE = re.compile(r"✅\s*(\d{4}-\d{2}-\d{2})")
ASSIGNEE = re.compile(r"\[assignee::\s*([^\]]+)\]")
STATUS = re.compile(r"\[status::\s*([^\]]+)\]")


@dataclass
class Task:
    """A single task from TASKS.md."""
    
    line_num: int  # 1-indexed line number in file
    checked: bool
    text: str  # Full text including metadata
    due_date: str | None = None  # YYYY-MM-DD
    done_date: str | None = None  # YYYY-MM-DD
    assignee: str | None = None
    status: str | None = None
    indent: str = ""  # Preserved for hierarchical tasks


def _parse_task_line(line: str, line_num: int) -> Task | None:
    """Parse a single line into a Task, or None if not a task."""
    match = TASK_LINE.match(line)
    if not match:
        return None
    
    indent = match.group("indent")
    checked = match.group("checked").lower() == "x"
    text = match.group("text")
    
    # Extract metadata
    due_match = DUE_DATE.search(text)
    done_match = DONE_DATE.search(text)
    assignee_match = ASSIGNEE.search(text)
    status_match = STATUS.search(text)
    
    return Task(
        line_num=line_num,
        checked=checked,
        text=text,
        due_date=due_match.group(1) if due_match else None,
        done_date=done_match.group(1) if done_match else None,
        assignee=assignee_match.group(1).strip() if assignee_match else None,
        status=status_match.group(1).strip() if status_match else None,
        indent=indent,
    )


class TaskStore:
    """Parse and manipulate TASKS.md in Obsidian Tasks format.
    
    Compatible with Obsidian Tasks plugin and Obsync for Apple Reminders sync.
    """
    
    def __init__(self, workspace_path: str | Path):
        self._workspace = Path(workspace_path)
        self._tasks_file = self._workspace / "TASKS.md"
    
    def get_tasks(
        self,
        assignee: str | None = None,
        status: str | None = None,
        due_before: str | None = None,  # YYYY-MM-DD
        include_completed: bool = False,
    ) -> list[Task]:
        """Query tasks with filters."""
        if not self._tasks_file.exists():
            return []
        
        lines = self._tasks_file.read_text().splitlines()
        tasks = []
        
        for i, line in enumerate(lines, start=1):
            task = _parse_task_line(line, i)
            if task is None:
                continue
            
            # Apply filters
            if not include_completed and task.checked:
                continue
            if assignee and task.assignee != assignee:
                continue
            if status and task.status != status:
                continue
            if due_before and task.due_date:
                if task.due_date > due_before:
                    continue
            
            tasks.append(task)
        
        return tasks
    
    def claim_task(self, line_num: int, agent_name: str) -> None:
        """Claim a task: set [assignee::] and [status::in-progress]."""
        if not self._tasks_file.exists():
            raise FileNotFoundError(f"TASKS.md not found at {self._tasks_file}")
        
        lines = self._tasks_file.read_text().splitlines()
        if line_num < 1 or line_num > len(lines):
            raise ValueError(f"Line {line_num} out of range (file has {len(lines)} lines)")
        
        line = lines[line_num - 1]
        task = _parse_task_line(line, line_num)
        if task is None:
            raise ValueError(f"Line {line_num} is not a valid task")
        
        if task.checked:
            raise ValueError(f"Task on line {line_num} is already completed")
        
        # Remove existing [assignee::] and [status::] if present
        text = task.text
        text = ASSIGNEE.sub("", text)
        text = STATUS.sub("", text)
        
        # Append new metadata at the end
        text = text.strip()
        text += f" [assignee:: {agent_name}] [status:: in-progress]"
        
        # Reconstruct the line
        new_line = f"{task.indent}- [ ] {text}"
        lines[line_num - 1] = new_line
        
        self._tasks_file.write_text("\n".join(lines) + "\n")
        log.info("Task line %d claimed by %s", line_num, agent_name)
    
    def complete_task(self, line_num: int) -> None:
        """Complete a task: check the box and stamp ✅ YYYY-MM-DD."""
        if not self._tasks_file.exists():
            raise FileNotFoundError(f"TASKS.md not found at {self._tasks_file}")
        
        lines = self._tasks_file.read_text().splitlines()
        if line_num < 1 or line_num > len(lines):
            raise ValueError(f"Line {line_num} out of range (file has {len(lines)} lines)")
        
        line = lines[line_num - 1]
        task = _parse_task_line(line, line_num)
        if task is None:
            raise ValueError(f"Line {line_num} is not a valid task")
        
        if task.checked:
            log.warning("Task on line %d is already completed", line_num)
            return
        
        # Check the box
        text = task.text
        
        # Remove [status::] if present
        text = STATUS.sub("", text).strip()
        
        # Add ✅ today if not already present
        today = datetime.date.today().isoformat()
        if not DONE_DATE.search(text):
            text += f" ✅ {today}"
        
        new_line = f"{task.indent}- [x] {text}"
        lines[line_num - 1] = new_line
        
        self._tasks_file.write_text("\n".join(lines) + "\n")
        log.info("Task line %d completed", line_num)
    
    def add_task(
        self,
        text: str,
        due_date: str | None = None,
        assignee: str | None = None,
    ) -> None:
        """Append a new task to TASKS.md."""
        task_line = f"- [ ] {text}"
        if due_date:
            task_line += f" 📅 {due_date}"
        if assignee:
            task_line += f" [assignee:: {assignee}]"
        
        # Append to file
        if self._tasks_file.exists():
            content = self._tasks_file.read_text()
            if not content.endswith("\n"):
                content += "\n"
            content += task_line + "\n"
        else:
            content = task_line + "\n"
        
        self._tasks_file.write_text(content)
        log.info("Added task: %s", text[:50])
