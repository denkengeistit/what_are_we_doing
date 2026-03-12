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
        self._watcher = None  # WAWDWatcher, set via set_watcher()

    def set_watcher(self, watcher) -> None:
        """Attach the watcher instance for pause/resume during restoration."""
        self._watcher = watcher

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

        # If the oracle failed to produce a usable plan, fall back to
        # restoring changed files to their state from 1 hour ago.
        if not plan.files and plan.confidence == "low":
            log.warning("Oracle produced no restoration plan; falling back to time-based restore")
            plan = await self._build_fallback_plan(scope, plan.auto_snapshot_name)

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
        """Execute a restoration plan: snapshot, restore, update disk.

        The watcher is paused during restoration so that disk writes
        are not re-versioned.
        """
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
                description="Auto-snapshot before restoration",
                created_by="oracle",
            )
        except Exception as e:
            log.warning("Failed to create pre-restore snapshot: %s", e)

        # Pause watcher so restoration writes aren't re-versioned
        if self._watcher:
            self._watcher.pause()

        # Execute restorations
        restored = []
        try:
            for fr in plan.files:
                try:
                    await self._vs.restore_file_to_version(fr.path, fr.to_version_id)

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
        finally:
            if self._watcher:
                self._watcher.resume()

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
            # Return empty low-confidence plan; analyze_and_restore() will
            # invoke _build_fallback_plan() when it sees this.
            return RestorationPlan(
                files=[],
                explanation=f"Oracle response was unparseable. Original response: {response[:200]}",
                auto_snapshot_name=snapshot_name,
                confidence="low",
            )

    async def _build_fallback_plan(
        self, scope: str | None, snapshot_name: str,
    ) -> RestorationPlan:
        """Build a fallback plan: restore changed files to their state 1 hour ago."""
        target_time = time.time() - 3600

        changes = await self._vs.get_changes_since(target_time)
        if scope:
            changes = [c for c in changes if c.path.startswith(scope)]

        files: list[FileRestoration] = []
        seen: set[str] = set()
        for change in changes:
            if change.path in seen:
                continue
            seen.add(change.path)

            history = await self._vs.get_history(change.path, limit=50)
            for entry in history:
                if entry.timestamp <= target_time and entry.blob_hash:
                    files.append(FileRestoration(
                        path=change.path,
                        to_version_id=entry.id,
                        reason="Fallback: oracle response unparseable, restoring to pre-change state",
                    ))
                    break

        explanation = (
            "Oracle could not produce a valid restoration plan. "
            "Falling back to restoring changed files to their state from 1 hour ago."
        )
        if not files:
            explanation += " No eligible files found in the target window."

        return RestorationPlan(
            files=files,
            explanation=explanation,
            auto_snapshot_name=snapshot_name,
            confidence="low",
        )
