# what_are_we_doing (wawd)

A shared versioned workspace with SLM oracle for multi-agent collaboration.

## Components

1. **Versioned filesystem (FUSE-T + SQLite)** — Transparent file versioning via userland FUSE mount.
2. **SLM oracle** — Local small language model (Ollama in Docker) that answers questions about project state.
3. **MCP server (3 tools)** — `what_are_we_doing`, `what_happened`, `fix_this`.

## Quick Start

```bash
# Start the SLM backend
docker compose up -d

# Install
uv pip install -e ".[dev]"

# Initialize a workspace
wawd init /path/to/your/project

# Start the daemon
wawd start
```

## Requirements

- Python 3.11+
- FUSE-T (macOS) or libfuse3 (Linux)
- Docker (for the SLM backend)
