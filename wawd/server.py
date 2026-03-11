"""MCP Server for WAWD. Exactly 3 tools."""

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
    """Create and configure the MCP server with exactly 3 tools."""
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
