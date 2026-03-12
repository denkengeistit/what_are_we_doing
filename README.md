# what_are_we_doing (wawd)

A shared versioned workspace with SLM oracle for multi-agent collaboration.

WAWD watches a directory for file changes, versions every modification in SQLite, and exposes an LLM-powered oracle via MCP tools so that multiple AI agents (and humans) can understand what's happening in a project.

## Components

1. **File watcher (watchdog + SQLite)** — Monitors a directory using OS-level filesystem events (FSEvents on macOS, inotify on Linux). Every create, modify, and delete is recorded with content-addressed blob storage and full version history.
2. **SLM oracle** — A language model that answers questions about project state, generates briefings for agents, and can analyze version history to restore files to working states.
3. **MCP server (3 tools)** — Exposes the oracle to AI agents via the Model Context Protocol:
   - `what_are_we_doing` — Get a briefing on the current workspace state, active agents, and recent changes.
   - `what_happened` — Query version history with filters by file, agent, or time range.
   - `fix_this` — Report a problem and let the oracle restore files to a working state.
4. **Streamlit UI** — Web dashboard for browsing version history, viewing diffs, and chatting with the oracle directly.

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Initialize a workspace
wawd init /path/to/your/project

# Start the daemon (watcher + MCP server + web UI)
wawd start
```

The web UI launches automatically at `http://localhost:8765` when the daemon starts. You can also run it standalone:

```bash
wawd ui --port 8080
```

## Oracle Backends

WAWD supports three oracle backends, configured in `~/.wawd/config.yaml`:

### Ollama (default)

Runs a local model via Ollama in Docker.

```bash
docker compose up -d   # starts Ollama with qwen2.5:3b
```

```yaml
oracle:
  backend: ollama
  model: qwen2.5:3b
  base_url: http://localhost:11434
```

### OpenAI-compatible (LM Studio, vLLM, etc.)

Connects to any server that implements the `/v1/chat/completions` endpoint.

```yaml
oracle:
  backend: openai_compat
  model: your-model-name
  base_url: http://your-server:1234
  api_key: ""  # optional; also reads WAWD_OPENAI_API_KEY env var
```

### llama.cpp

Connects directly to a llama.cpp server.

```yaml
oracle:
  backend: llamacpp
  base_url: http://localhost:8080
```

## Configuration

All configuration lives in `~/.wawd/config.yaml`. Running `wawd init` updates the workspace path while preserving existing settings.

Key settings:

- `workspace.path` — Directory to watch.
- `workspace.exclude` — Glob patterns to ignore (defaults: `node_modules/`, `.git/`, `__pycache__/`, etc.).
- `oracle.backend` — `ollama`, `openai_compat`, or `llamacpp`.
- `oracle.timeout_seconds` — Request timeout (default: 300s).
- `oracle.context_budget_tokens` — Max tokens for oracle context (default: 32000).
- `versioning.max_versions_per_file` — Version cap per file (default: 500).
- `versioning.compression_level` — Zstandard compression level for blobs (default: 3).

## CLI Commands

| Command | Description |
|---------|-------------|
| `wawd init <dir>` | Initialize a workspace (creates config + database) |
| `wawd start` | Start the daemon (watcher + MCP server + UI) |
| `wawd stop` | Stop the running daemon |
| `wawd status` | Show workspace status, file/version counts, active sessions |
| `wawd ask "question"` | Query the oracle directly from the command line |
| `wawd ui` | Launch the web UI standalone |

## Architecture

```
workspace directory
  │
  ├─ watchdog observer ──▶ VersionStore (SQLite)
  │                            │
  │                            ▼
  │                        BlobStore (zstd-compressed content)
  │
  ├─ MCP server (stdio) ──▶ Oracle
  │                            ├─ ContextBuilder (assembles tiered prompts)
  │                            ├─ SessionTracker (implicit agent sessions)
  │                            └─ Restorer (version-based file recovery)
  │                                   │
  │                                   ▼
  │                              SLM Backend (Ollama / OpenAI-compat / llama.cpp)
  │
  └─ Streamlit UI ──▶ reads SQLite directly + queries Oracle
```

## Requirements

- Python 3.11+
- Docker (optional, for Ollama backend)

## Development

```bash
uv pip install -e ".[dev]"
pytest
```
