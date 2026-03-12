"""Click CLI for WAWD: init, start, stop, status, ask."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from wawd.config import (
    DEFAULT_CONFIG_DIR,
    WAWDConfig,
    create_default_config,
    load_config,
)

console = Console()
log = logging.getLogger(__name__)


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=False), default=None, help="Path to config file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """WAWD — What Are We Doing: versioned workspace for multi-agent collaboration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config_path) if config_path else None


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def init(directory: str) -> None:
    """Initialize a WAWD workspace for DIRECTORY."""
    directory = str(Path(directory).resolve())

    config_path = create_default_config(directory)
    config = load_config(config_path)

    # Initialize database
    async def _init_db():
        import aiosqlite
        from wawd.fs.blob_store import BlobStore
        from wawd.fs.version_store import VersionStore
        from wawd.oracle.session_tracker import SessionTracker

        config.config_dir.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(config.db_path))
        try:
            blob_store = BlobStore(db, config.versioning.compression_level)
            await blob_store.init_db()
            version_store = VersionStore(db, blob_store)
            await version_store.init_db()
            tracker = SessionTracker(db, config.oracle.session_timeout_minutes)
            await tracker.init_db()
        finally:
            await db.close()

    asyncio.run(_init_db())

    console.print(f"[green]✓[/green] Workspace initialized")
    console.print(f"  Config:  {config_path}")
    console.print(f"  Source:  {config.workspace.path}")
    console.print(f"  DB:      {config.db_path}")
    console.print()
    console.print("Next steps:")
    console.print("  1. Start the SLM backend: [bold]docker compose up -d[/bold]")
    console.print("  2. Start WAWD:            [bold]wawd start[/bold]")


@main.command()
@click.pass_context
def start(ctx: click.Context) -> None:
    """Start the WAWD daemon (watcher + MCP server)."""
    config_path = ctx.obj.get("config_path")
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]wawd init <directory>[/bold] first.")
        sys.exit(1)

    asyncio.run(_start_daemon(config))


async def _start_daemon(config: WAWDConfig) -> None:
    """Start all components and run until interrupted."""
    import aiosqlite

    from wawd.fs.blob_store import BlobStore
    from wawd.fs.version_store import VersionStore
    from wawd.fs.watcher import WAWDWatcher
    from wawd.oracle.backends.ollama import OllamaBackend
    from wawd.oracle.backends.llamacpp import LlamaCppBackend
    from wawd.oracle.backends.openai_compat import OpenAICompatBackend
    from wawd.oracle.context import ContextBuilder
    from wawd.oracle.oracle import Oracle
    from wawd.oracle.restorer import Restorer
    from wawd.oracle.session_tracker import SessionTracker
    from wawd.server import run_stdio_server

    # Open database
    config.config_dir.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(config.db_path))

    watcher: WAWDWatcher | None = None
    ui_proc: subprocess.Popen | None = None
    try:
        # Initialize stores
        blob_store = BlobStore(db, config.versioning.compression_level)
        await blob_store.init_db()
        version_store = VersionStore(db, blob_store)
        await version_store.init_db()
        session_tracker = SessionTracker(db, config.oracle.session_timeout_minutes)
        await session_tracker.init_db()

        # Initialize backend
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

        # Check backend health
        healthy = await backend.health_check()
        if healthy:
            console.print(f"[green]✓[/green] Oracle backend ({config.oracle.backend}) is healthy")
        else:
            console.print(f"[yellow]⚠[/yellow] Oracle backend ({config.oracle.backend}) is not responding")
            console.print("  The oracle will return fallback responses until the backend is available.")

        # Initialize oracle components
        context_builder = ContextBuilder(
            version_store, session_tracker,
            config.oracle.history_depth,
        )
        restorer = Restorer(version_store, context_builder, backend, config.workspace.path)
        oracle = Oracle(
            version_store, session_tracker, context_builder, restorer,
            backend, config.workspace.path,
        )

        # Start watcher
        console.print(f"Watching: {config.workspace.path}")
        watcher = WAWDWatcher(
            config.workspace.path,
            version_store,
            config.workspace.exclude,
            session_tracker=session_tracker,
        )
        await watcher.start()
        oracle.set_watcher(watcher)
        restorer.set_watcher(watcher)

        # Write PID file
        config.pid_path.write_text(str(os.getpid()))

        # Start Streamlit UI in background
        ui_path = Path(__file__).parent / "ui.py"
        ui_port = config.mcp.port or 8501
        ui_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ui_path),
             "--server.port", str(ui_port), "--server.headless", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        console.print(f"[green]✓[/green] WAWD started (PID {os.getpid()})")
        console.print(f"  Work in: [bold]{config.workspace.path}[/bold]")
        console.print(f"  UI:      [bold]http://localhost:{ui_port}[/bold]")

        # Run MCP server (watcher runs in background via its drain loop)
        await run_stdio_server(oracle)

    finally:
        if watcher:
            await watcher.stop()
        # Stop Streamlit UI
        if ui_proc and ui_proc.poll() is None:
            ui_proc.terminate()
            ui_proc.wait(timeout=5)
        await backend.close()
        await db.close()
        config.pid_path.unlink(missing_ok=True)
        console.print("[green]✓[/green] WAWD stopped")


@main.command()
def stop() -> None:
    """Stop the running WAWD daemon."""
    pid_path = DEFAULT_CONFIG_DIR / "wawd.pid"
    if not pid_path.exists():
        console.print("[yellow]No running WAWD instance found.[/yellow]")
        return

    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]✓[/green] Sent stop signal to WAWD (PID {pid})")
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid} not found. Cleaning up PID file.[/yellow]")
        pid_path.unlink(missing_ok=True)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show WAWD workspace status."""
    config_path = ctx.obj.get("config_path")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        console.print("[yellow]WAWD not initialized. Run wawd init first.[/yellow]")
        return

    # Check if running
    pid_path = config.pid_path
    running = False
    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            pass

    table = Table(title="WAWD Status")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Status", "[green]Running[/green]" if running else "[red]Stopped[/red]")
    table.add_row("Source", config.workspace.path)
    table.add_row("Database", str(config.db_path))
    table.add_row("Backend", f"{config.oracle.backend} ({config.oracle.model})")
    table.add_row("Backend URL", config.oracle.base_url)

    console.print(table)

    # Show session/change counts if DB exists
    if config.db_path.exists():
        async def _show_stats():
            import aiosqlite
            from wawd.fs.blob_store import BlobStore
            from wawd.fs.version_store import VersionStore
            from wawd.oracle.session_tracker import SessionTracker

            db = await aiosqlite.connect(str(config.db_path))
            try:
                cursor = await db.execute("SELECT COUNT(*) FROM versions")
                row = await cursor.fetchone()
                console.print(f"\n  Versions: {row[0]}")

                cursor = await db.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
                row = await cursor.fetchone()
                console.print(f"  Active sessions: {row[0]}")

                cursor = await db.execute("SELECT COUNT(*) FROM blobs")
                row = await cursor.fetchone()
                console.print(f"  Blobs: {row[0]}")
            except Exception:
                pass
            finally:
                await db.close()

        asyncio.run(_show_stats())


@main.command()
@click.argument("question")
@click.pass_context
def ask(ctx: click.Context, question: str) -> None:
    """Ask the oracle a question about the workspace."""
    config_path = ctx.obj.get("config_path")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        console.print("[red]WAWD not initialized. Run wawd init first.[/red]")
        sys.exit(1)

    asyncio.run(_ask(config, question))


async def _ask(config: WAWDConfig, question: str) -> None:
    """Query the oracle directly."""
    import aiosqlite

    from wawd.fs.blob_store import BlobStore
    from wawd.fs.version_store import VersionStore
    from wawd.oracle.backends.ollama import OllamaBackend
    from wawd.oracle.backends.llamacpp import LlamaCppBackend
    from wawd.oracle.backends.openai_compat import OpenAICompatBackend
    from wawd.oracle.context import ContextBuilder
    from wawd.oracle.oracle import Oracle
    from wawd.oracle.restorer import Restorer
    from wawd.oracle.session_tracker import SessionTracker

    db = await aiosqlite.connect(str(config.db_path))
    try:
        blob_store = BlobStore(db, config.versioning.compression_level)
        version_store = VersionStore(db, blob_store)
        session_tracker = SessionTracker(db, config.oracle.session_timeout_minutes)

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
            config.oracle.history_depth,
        )
        restorer = Restorer(version_store, context_builder, backend, config.workspace.path)
        oracle = Oracle(
            version_store, session_tracker, context_builder, restorer,
            backend, config.workspace.path,
        )

        result = await oracle.history(question=question)
        console.print(result["answer"])

        if result["changes"]:
            console.print(f"\n[dim]({len(result['changes'])} changes in scope)[/dim]")

        await backend.close()
    finally:
        await db.close()


@main.command()
@click.option("--port", default=8501, help="Port to run the UI on.")
def ui(port: int) -> None:
    """Launch the WAWD web UI (Streamlit)."""
    import subprocess

    ui_path = Path(__file__).parent / "ui.py"
    console.print(f"[green]✓[/green] Starting WAWD UI on http://localhost:{port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path),
         "--server.port", str(port), "--server.headless", "true"],
    )


if __name__ == "__main__":
    main()
