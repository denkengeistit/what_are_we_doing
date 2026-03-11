"""Restorer: parses oracle restoration output and executes file restorations."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from wawd.exceptions import RestorationError
from wawd.fs.version_store import VersionStore
from wawd.oracle.backends.base import OracleBackend
from wawd.oracle.context import ContextBuilder

log = logging.getLogger(__name__)


@dataclass
class FileRestoration:
    """A single file restoration instruction."""

    path: str
    to_version_id: int
    reason: str = ""


@dataclass
class RestorationPlan:
    """A complete restoration plan from the oracle."""

    files: list[FileRestoration]
    explanation: str
    auto_snapshot_name: str
    confidence: str = "medium"


@dataclass
class RestorationResult:
    """Result of executing a restoration."""

    action_taken: str
    files_restored: list[dict]
    explanation: str
    snapshot_created: str


class Restorer:
    """Executes restoration decisions made by the oracle."""

    def __init__(
        self,
        version_store: VersionStore,
        context_builder: ContextBuilder,
        backend: OracleBackend,
        workspace_path: str,
    ) -> None:
        self._vs = version_store
        self._ctx = context_builder
        self._backend = backend
        self._workspace = Path(workspace_path)

    async def analyze_and_restore(
        self,
        problem: str,
        scope: str | None = None,
        dry_run: bool = False,
    ) -> RestorationResult:
        """Full restoration workflow: query oracle, parse plan, execute."""
        # Build context
        messages = await self._ctx.build_restoration_context(problem, scope)

        # Query oracle
        response = await self._backend.generate(messages, max_tokens=2048)

        # Parse oracle response
        plan = self._parse_oracle_response(response)

        if dry_run:
            return RestorationResult(
                action_taken="dry_run",
                files_restored=[
                    {"path": f.path, "to_version": f.to_version_id, "reason": f.reason}
                    for f in plan.files
                ],
                explanation=plan.explanation,
                snapshot_created="(dry run — no snapshot)",
            )

        return await self.execute_restoration_plan(plan)

    async def execute_restoration_plan(self, plan: RestorationPlan) -> RestorationResult:
        """Execute a restoration plan: snapshot, restore, update disk."""
        if not plan.files:
            return RestorationResult(
                action_taken="no_action",
                files_restored=[],
                explanation=plan.explanation,
                snapshot_created="",
            )

        # Create pre-restoration snapshot
        try:
            await self._vs.create_snapshot(
                name=plan.auto_snapshot_name,
                description=f"Auto-snapshot before restoration",
                created_by="oracle",
            )
        except Exception as e:
            log.warning("Failed to create pre-restore snapshot: %s", e)

        # Execute restorations
        restored = []
        for fr in plan.files:
            try:
                await self._vs.restore_file_to_version(fr.path, fr.to_version_id)

                # Update on-disk file
                content = await self._vs.get_content(fr.to_version_id)
                disk_path = self._workspace / fr.path
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                disk_path.write_bytes(content)

                restored.append({
                    "path": fr.path,
                    "to_version": fr.to_version_id,
                    "reason": fr.reason,
                })
                log.info("Restored %s to version %d", fr.path, fr.to_version_id)
            except Exception as e:
                log.error("Failed to restore %s: %s", fr.path, e)
                restored.append({
                    "path": fr.path,
                    "to_version": fr.to_version_id,
                    "error": str(e),
                })

        return RestorationResult(
            action_taken="restored",
            files_restored=restored,
            explanation=plan.explanation,
            snapshot_created=plan.auto_snapshot_name,
        )

    def _parse_oracle_response(self, response: str) -> RestorationPlan:
        """Parse the oracle's JSON response into a RestorationPlan."""
        snapshot_name = f"pre-restore-{int(time.time())}"

        # Try to extract JSON from response
        try:
            # Strip markdown code fences if present
            text = response.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            data = json.loads(text)

            files = []
            for f in data.get("files_to_restore", []):
                files.append(FileRestoration(
                    path=f["path"],
                    to_version_id=f["to_version_id"],
                    reason=f.get("reason", ""),
                ))

            return RestorationPlan(
                files=files,
                explanation=data.get("explanation", ""),
                auto_snapshot_name=snapshot_name,
                confidence=data.get("confidence", "medium"),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("Failed to parse oracle response as JSON: %s", e)
            log.warning("Oracle response was: %s", response[:500])
            # Fallback: empty plan (caller can decide to restore from snapshot)
            return RestorationPlan(
                files=[],
                explanation=f"Oracle response was unparseable. Original response: {response[:200]}",
                auto_snapshot_name=snapshot_name,
                confidence="low",
            )
