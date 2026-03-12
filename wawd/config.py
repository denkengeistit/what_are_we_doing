"""WAWD configuration: Pydantic settings loaded from YAML."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

log = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".wawd"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"


class WorkspaceConfig(BaseModel):
    """Source directory and watcher settings."""

    path: str
    exclude: list[str] = [
        "node_modules/",
        ".git/",
        "__pycache__/",
        "*.pyc",
        "build/",
        "dist/",
        ".DS_Store",
        "*.swp",
        "*~",
        "*.tmp",
        ".#*",
    ]


class VersioningConfig(BaseModel):
    """Versioning behavior knobs."""

    max_versions_per_file: int = 500
    skip_binary_over_mb: int = 10
    compression_level: int = 3


class OracleConfig(BaseModel):
    """SLM backend settings (Ollama in Docker by default)."""

    backend: str = "ollama"
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 30.0
    context_budget_tokens: int = 32000
    history_depth: int = 50
    cache_briefings_seconds: int = 60
    session_timeout_minutes: int = 30


class MCPConfig(BaseModel):
    """MCP server transport settings."""

    transport: str = "stdio"
    port: int = 8765


class WAWDConfig(BaseModel):
    """Top-level configuration."""

    workspace: WorkspaceConfig
    versioning: VersioningConfig = VersioningConfig()
    oracle: OracleConfig = OracleConfig()
    mcp: MCPConfig = MCPConfig()

    @property
    def config_dir(self) -> Path:
        return DEFAULT_CONFIG_DIR

    @property
    def db_path(self) -> Path:
        return DEFAULT_CONFIG_DIR / "wawd.db"

    @property
    def pid_path(self) -> Path:
        return DEFAULT_CONFIG_DIR / "wawd.pid"


def load_config(path: Path | None = None) -> WAWDConfig:
    """Load configuration from a YAML file."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    log.debug("Loaded config from %s", config_path)
    return WAWDConfig(**raw)


def create_default_config(workspace_path: str) -> Path:
    """Create a default config.yaml for a workspace and return its path."""
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "workspace": {
            "path": str(Path(workspace_path).resolve()),
        },
    }

    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log.info("Created config at %s", DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG_PATH
