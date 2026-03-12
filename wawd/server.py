"""MCP Server for WAWD: versioning + task management."""

from __future__ import annotations

import logging
import re
import time

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from wawd.oracle.oracle import Oracle

log = logging.getLogger(__name__)


def _parse_since(since: str | None) -> float | None:
    """Parse a 'since' parameter into a Unix timestamp.

    Formats:
        "2h", "30m", "7d" — relative to now
        ISO 8601 string — parsed directly
        "last_session" — placeholder (handled by caller)
        None — returns None
    """
    if since is None:
        return None

    # Relative: e.g. "2h", "30m", "7d"
    match = re.match(r"^(\d+)([hmd])$", since)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        seconds = {"h": 3600, "m": 60, "d": 86400}[unit]
        return time.time() - (value * seconds)

    # ISO format
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(since)
        return dt.timestamp()
    except ValueError:
        pass

    return None


def create_mcp_server(oracle: Oracle) -> Server:
    """Create and configure the MCP server."""
    server = Server("wawd")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="what_are_we_doing",
                description=(
                    "Get a briefing on the current workspace state. "
                    "Call this when starting or resuming work to understand "
                    "what's happening in the project."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Your agent name/identifier.",
                        },
                        "task": {
                            "type": "string",
                            "description": "What you plan to work on (optional).",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Specific area of interest (optional).",
                        },
                    },
                    "required": ["agent_name"],
                },
            ),
            Tool(
                name="what_happened",
                description=(
                    "Ask about workspace history. Query what changed, "
                    "who changed it, and when."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Your question about the workspace history.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Filter to a specific file or directory path.",
                        },
                        "agent": {
                            "type": "string",
                            "description": "Filter to changes by a specific agent.",
                        },
                        "since": {
                            "type": "string",
                            "description": "Time filter: '2h', '30m', '7d', ISO timestamp, or 'last_session'.",
                        },
                    },
                },
            ),
            Tool(
                name="fix_this",
                description=(
                    "Report a problem and let the oracle analyze version history "
                    "to restore files to a working state."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "Describe the problem in natural language.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "Limit restoration to a file or directory path.",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, show what would be restored without actually doing it.",
                            "default": False,
                        },
                    },
                    "required": ["problem"],
                },
            ),
            Tool(
                name="get_tasks",
                description=(
                    "List tasks from TASKS.md, filtered by assignee, status, or due date. "
                    "Returns line numbers for claiming/completing."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "assignee": {
                            "type": "string",
                            "description": "Filter to tasks assigned to this agent.",
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by status (e.g. 'in-progress').",
                        },
                        "due_before": {
                            "type": "string",
                            "description": "Filter to tasks due before YYYY-MM-DD.",
                        },
                        "include_completed": {
                            "type": "boolean",
                            "description": "Include completed tasks.",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="claim_task",
                description=(
                    "Claim a task by line number: sets [assignee:: <your name>] and "
                    "[status:: in-progress]. Use get_tasks to find available tasks first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "line_num": {
                            "type": "integer",
                            "description": "Line number of the task in TASKS.md.",
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Your agent name/identifier.",
                        },
                    },
                    "required": ["line_num", "agent_name"],
                },
            ),
            Tool(
                name="complete_task",
                description=(
                    "Mark a task as complete: checks the box and stamps ✅ YYYY-MM-DD. "
                    "Obsync will sync this to Apple Reminders."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "line_num": {
                            "type": "integer",
                            "description": "Line number of the task in TASKS.md.",
                        },
                    },
                    "required": ["line_num"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "what_are_we_doing":
                result = await oracle.briefing(
                    agent_name=arguments["agent_name"],
                    task=arguments.get("task"),
                    focus=arguments.get("focus"),
                )
                text = f"Workspace: {result['workspace_path']}\n\n{result['briefing']}"
                return [TextContent(type="text", text=text)]

            elif name == "what_happened":
                since = _parse_since(arguments.get("since"))
                result = await oracle.history(
                    question=arguments.get("question"),
                    path=arguments.get("path"),
                    agent=arguments.get("agent"),
                    since=since,
                )
                text = result["answer"]
                if result["changes"]:
                    text += f"\n\n({len(result['changes'])} changes in scope)"
                return [TextContent(type="text", text=text)]

            elif name == "fix_this":
                result = await oracle.fix(
                    problem=arguments["problem"],
                    scope=arguments.get("scope"),
                    dry_run=arguments.get("dry_run", False),
                )
                parts = [f"Action: {result['action_taken']}"]
                if result["files_restored"]:
                    parts.append(f"Files restored: {len(result['files_restored'])}")
                    for f in result["files_restored"]:
                        parts.append(f"  - {f['path']} -> v{f.get('to_version', '?')}")
                parts.append(f"\n{result['explanation']}")
                return [TextContent(type="text", text="\n".join(parts))]
            
            elif name == "get_tasks":
                from wawd.tasks import TaskStore
                task_store = TaskStore(oracle._workspace)
                tasks = task_store.get_tasks(
                    assignee=arguments.get("assignee"),
                    status=arguments.get("status"),
                    due_before=arguments.get("due_before"),
                    include_completed=arguments.get("include_completed", False),
                )
                if not tasks:
                    return [TextContent(type="text", text="No tasks found matching filters.")]
                
                lines = []
                for task in tasks:
                    line = f"Line {task.line_num}: {'[x]' if task.checked else '[ ]'} {task.text}"
                    lines.append(line)
                return [TextContent(type="text", text="\n".join(lines))]
            
            elif name == "claim_task":
                from wawd.tasks import TaskStore
                task_store = TaskStore(oracle._workspace)
                task_store.claim_task(
                    line_num=arguments["line_num"],
                    agent_name=arguments["agent_name"],
                )
                return [TextContent(type="text", text=f"Task on line {arguments['line_num']} claimed by {arguments['agent_name']}.")]
            
            elif name == "complete_task":
                from wawd.tasks import TaskStore
                task_store = TaskStore(oracle._workspace)
                task_store.complete_task(line_num=arguments["line_num"])
                return [TextContent(type="text", text=f"Task on line {arguments['line_num']} marked complete.")]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            log.exception("Tool %s failed", name)
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


async def run_stdio_server(oracle: Oracle) -> None:
    """Run the MCP server over stdio transport."""
    server = create_mcp_server(oracle)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
