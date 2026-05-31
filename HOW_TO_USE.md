# How to Use Twin

This guide walks through installing Twin, ingesting your notes, and running each of the four CLI commands.

---

## Prerequisites

- **Python 3.11+** — check with `python --version`
- **Rust toolchain** — required to build the native chunking extension. Install from [rustup.rs](https://rustup.rs).
- **uv** — fast Python package manager. Install with `pip install uv` or follow [the uv docs](https://docs.astral.sh/uv/getting-started/installation/).
- **An Anthropic API key** — required for the `rag` and `agent` commands only. The `ingest` and `query` commands run fully offline.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/BarrelDev/twin-repo
cd twin-repo
```

### 2. Build the Rust extension

The chunking hot path is implemented in Rust (`twin-core/`). Build it before installing Python dependencies:

```bash
cd twin_core
pip install maturin
maturin develop
cd ..
```

`maturin develop` compiles the Rust crate and installs it into the current Python environment as a native extension. You only need to re-run this if you modify the Rust source.

### 3. Install Python dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` into a local virtual environment (`.venv/`). The first time you run an embedding command, `nomic-embed-text-v1.5` (~270 MB) will be downloaded from Hugging Face and cached locally — this only happens once.

---

## API Key Setup (for RAG and Agent commands)

Export your Anthropic API key before running `twin rag` or `twin agent`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

The `ingest` and `query` commands do not require this key.

---

## The Four Commands

### `twin ingest` — Index your notes

Reads a folder of Markdown files, chunks them, embeds them, and stores them locally.

```bash
uv run twin ingest ./notes
```

- Recursively finds all `.md` files under the given path.
- Skips files that haven't changed since the last ingest (SHA-256 hash check).
- Re-ingests files that have been modified.
- Stores everything in `~/.twin/` by default.

Running `ingest` twice on unchanged files is safe — it produces no changes.

**Example output:**

```
twin Scanning ./notes ... 47 files found
Ingesting... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Done. Ingested 47 files (312 chunks). Skipped 0 unchanged.
```

---

### `twin query` — Semantic search (offline)

Searches the knowledge base by semantic similarity and returns the most relevant chunks. No API key needed.

```bash
uv run twin query "What did I write about the Rust ownership model?"
```

Results are ranked by similarity score and displayed in a table showing the source file, heading path, and a text excerpt.

**Example output:**

```
┌───┬───────┬────────────────────────┬──────────────────────────────────┐
│ # │ Score │ Source                 │ Text                             │
├───┼───────┼────────────────────────┼──────────────────────────────────┤
│ 1 │ 0.91  │ rust.md › Ownership   │ "Ownership in Rust is the        │
│   │       │                        │  mechanism by which..."          │
└───┴───────┴────────────────────────┴──────────────────────────────────┘
```

Returns 5 results by default. Change this with the `SECONDBRAIN_TOP_K` environment variable (see [Configuration](#configuration)).

---

### `twin rag` — Retrieval-augmented answer (requires API key)

Retrieves relevant chunks from your notes, then asks Claude to synthesize a grounded answer. The model is instructed to answer only from the retrieved context, cite its sources, and admit when it doesn't know.

```bash
uv run twin rag "What is the Rust ownership model?"
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--top-k N` / `-k N` | `5` | Number of chunks to retrieve before synthesis |

**Example output:**

```
╭─ Answer ──────────────────────────────────────────────────────────────────╮
│ Rust's ownership model gives each value in memory a single owner. When   │
│ that owner goes out of scope, the value is dropped automatically...       │
╰───────────────────────────────────────────────────────────────────────────╯

Sources:
  • rust.md  Ownership
  • systems.md  Memory Safety
```

---

### `twin agent` — Multi-step reasoning agent (requires API key)

Gives Claude access to your knowledge base as a tool it can search repeatedly. The agent decides when to search, how many times, and what to look for — then returns a synthesized final answer.

```bash
uv run twin agent "Summarize everything I've written about async Rust"
```

The agent will search the knowledge base up to 5 times (configurable) before producing its answer.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--verbose` / `-v` | off | Print the full activity log: every tool call, its results, and the termination reason |
| `--max-iter N` | `5` | Maximum number of KB searches before the agent is forced to answer |

**Example output (default):**

```
╭─ Agent Answer ────────────────────────────────────────────────────────────╮
│ Based on your notes, async Rust centers on the Future trait and the      │
│ async/await syntax introduced in Rust 1.39...                            │
╰───────────────────────────────────────────────────────────────────────────╯

Tool calls made: 3
```

**Example output (`--verbose`):**

```
╭─ Agent Answer ─╮
│ ...            │
╰────────────────╯

Tool calls made: 3

Activity Log:
  iter 0  tool_call  search_knowledge_base({'query': 'async rust'})
  iter 0  result     [source: rust.md > Async] The async keyword marks a...
  iter 1  tool_call  search_knowledge_base({'query': 'Future trait'})
  ...
  iter 3  done       (final_answer)
```

---

## Configuration

All settings are read from environment variables. No config file to manage.

| Variable | Default | Description |
|---|---|---|
| `SECONDBRAIN_DATA_DIR` | `~/.twin` | Directory where LanceDB and SQLite are stored |
| `SECONDBRAIN_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model identifier |
| `SECONDBRAIN_CHUNK_TOKENS` | `512` | Max tokens per chunk during ingestion |
| `SECONDBRAIN_OVERLAP` | `64` | Overlap tokens between adjacent chunks |
| `SECONDBRAIN_TOP_K` | `5` | Default number of results for `query` and `rag` |
| `ANTHROPIC_API_KEY` | *(required for rag/agent)* | Your Anthropic API key |

**Example — point to a different data directory:**

```bash
export SECONDBRAIN_DATA_DIR="/data/my-twin"
uv run twin ingest ~/notes
```

---

## Typical Workflow

```bash
# 1. Ingest your notes (run this whenever you add or edit files)
uv run twin ingest ~/notes

# 2. Do a quick semantic search (no API key needed)
uv run twin query "deployment checklists"

# 3. Get a synthesized answer from Claude
export ANTHROPIC_API_KEY="sk-ant-..."
uv run twin rag "What are my notes on CI/CD best practices?"

# 4. Use the agent for multi-step or open-ended tasks
uv run twin agent "What have I written about Python performance, and what are the common themes?"
```

---

## Running Tests

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

The first test run downloads the embedding model if it hasn't been cached yet.
