# How to Use Twin

This guide covers installing Twin and using every CLI command.

---

## Prerequisites

- **Python 3.11+** — check with `python --version`
- **Rust toolchain** — required to build the native chunking extension. Install from [rustup.rs](https://rustup.rs).
- **uv** — fast Python package manager. Install with `pip install uv` or see [the uv docs](https://docs.astral.sh/uv/getting-started/installation/).
- **An LLM provider** — one of:
  - An API key for Anthropic, OpenAI, Google Gemini, or OpenRouter
  - [Ollama](https://ollama.com) for fully local, free-of-cost operation
  - `ingest` and `query` work offline with no key at all

---

## Installation

Clone and run the one-step setup script:

```bash
git clone https://github.com/BarrelDev/twin-repo
cd twin-repo

# Windows
setup.bat

# macOS / Linux
./setup.sh
```

Or manually:

```bash
uv sync

cd twin_core
pip install maturin
maturin develop
cd ..
```

> The first time you run any command that embeds text, `nomic-embed-text-v1.5` (~270 MB) is downloaded from Hugging Face and cached. This only happens once.

---

## API Key Setup

Twin stores API keys encrypted on disk — they are never printed, logged, or echoed.

```bash
twin config set-key
```

Follow the interactive prompt: choose a provider, then paste your key. The key is stored in `~/.twin/keychain.enc` using AES-256-GCM, machine-bound (non-portable).

Alternatively, export the key as an environment variable (the keychain takes priority):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

To use a **local model with no API key**:

```bash
# Start Ollama (https://ollama.com), then:
twin config set-provider ollama
twin config set-model llama3.2
```

The `ingest` and `query` commands never require any API key.

---

## `twin ingest` — Index your knowledge base

Chunks, embeds, and stores content locally. Supports Markdown files, PDFs, and URLs.

### Markdown files

```bash
twin ingest ./notes
```

Recursively finds all `.md` files under the given directory. Parses Obsidian-specific syntax:
- Extracts `[[wikilinks]]` and stores link targets in metadata
- Extracts `#tags` and `#nested/tags` from the body
- Strips `![[embeds]]` from chunk text before embedding
- Preserves full YAML frontmatter in the metadata store

Non-Obsidian Markdown is supported — wikilinks and tags simply come back empty.

**Example output:**

```
twin Scanning ./notes ... 47 files found
Ingesting... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Done. Ingested 47 files (312 chunks). Skipped 0 unchanged.
```

### PDF files

```bash
twin ingest research.pdf
```

Extracts text page-by-page using pymupdf. Source attribution uses `document.pdf › p.4` format.

```
Done. Ingested research.pdf (18 chunks).
```

### URLs

```bash
twin ingest https://example.com/article
```

Fetches and extracts the main article content using trafilatura. Source attribution uses `domain.com › Page Title` format.

```
Done. Ingested URL (6 chunks).
```

### Force format detection

```bash
twin ingest some-file --type pdf
twin ingest some-path --type url
twin ingest some-path --type md
```

### Idempotency

Running `ingest` twice on unchanged content is safe and produces no changes. SHA-256 hashes of file content (or URL content) are stored in SQLite. On re-ingest: hash match → skip; hash changed → delete old chunks, insert new.

---

## `twin query` — Semantic search (no API key needed)

Searches the knowledge base by semantic similarity. Returns ranked chunks with source attribution. No LLM call — runs fully offline.

```bash
twin query "What did I write about the Rust ownership model?"
```

**Example output:**

```
┌───┬───────┬────────────────────────┬──────────────────────────────────┐
│ # │ Score │ Source                 │ Text                             │
├───┼───────┼────────────────────────┼──────────────────────────────────┤
│ 1 │ 0.91  │ rust.md › Ownership   │ "Ownership in Rust is the        │
│   │       │                        │  mechanism by which..."          │
└───┴───────┴────────────────────────┴──────────────────────────────────┘
```

Returns 5 results by default. Change with `TWIN_TOP_K` (see [Configuration](#configuration)).

---

## `twin rag` — Retrieval-augmented answer

Retrieves relevant chunks from your knowledge base, then asks the LLM to synthesize a grounded answer. The model is instructed to cite sources and admit when it doesn't know.

The answer streams token-by-token. Sources and a usage summary are printed after the stream completes.

```bash
twin rag "What is the Rust ownership model?"
twin rag "What is the Rust ownership model?" --provider openai
twin rag "What is the Rust ownership model?" -k 10
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--top-k N` / `-k N` | `5` | Chunks to retrieve before synthesis |
| `--provider P` / `-p P` | active provider | Override provider for this call only |

**Example output:**

```
Rust's ownership model gives each value in memory a single owner. When
that owner goes out of scope, the value is dropped automatically...

Sources:
  • rust.md  Ownership
  • systems.md  Memory Safety

1 call · 1,240 tokens · ~$0.003
```

---

## `twin agent` — Multi-step reasoning agent

Gives the LLM access to your knowledge base as a tool it can search repeatedly. The agent decides when to search, how many times, and what to look for — then returns a synthesized final answer. The final answer streams token-by-token.

If a vault path is configured (via `twin config set-key` → vault option, or `cm.set_vault_path()`), the agent can also write notes to `<vault>/Agents/` using the `write_vault_note` tool.

```bash
twin agent "Summarize everything I've written about async Rust"
twin agent "Draft a summary of my ML notes" --provider gemini
twin agent "..." --verbose --max-iter 10
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--verbose` / `-v` | off | Print the full activity log after the answer |
| `--max-iter N` | `5` | Maximum KB searches before the agent is forced to answer |
| `--provider P` / `-p P` | active provider | Override provider for this call only |

**Example output (default):**

```
  iter 0 → search_knowledge_base  [source: rust.md > Async] The async keyword...
  iter 1 → search_knowledge_base  [source: tokio.md > Runtime] Tokio provides...

Based on your notes, async Rust centers on the Future trait and the
async/await syntax introduced in Rust 1.39...

Tool calls made: 2
2 calls · 890 tokens · ~$0.005
```

**Example output (`--verbose`):**

```
...answer...

Tool calls made: 2
2 calls · 890 tokens · ~$0.005

Activity Log:
  iter 0  tool_call  search_knowledge_base({'query': 'async rust'})
  iter 0  result     [source: rust.md > Async] The async keyword marks a...
  iter 1  tool_call  search_knowledge_base({'query': 'Future trait tokio'})
  iter 1  result     [source: tokio.md > Runtime] Tokio provides an async...
  iter 2  done       (final_answer)
```

---

## `twin config` — Manage providers and API keys

### `twin config set-key`

Interactively store an API key for a provider. The key is encrypted and never echoed.

```bash
twin config set-key
```

### `twin config remove-key <provider>`

Remove a stored key from the keychain.

```bash
twin config remove-key openai
```

### `twin config set-provider <provider>`

Set the active LLM provider. Valid values: `anthropic`, `openai`, `gemini`, `ollama`, `openrouter`.

```bash
twin config set-provider gemini
```

### `twin config set-model <model>`

Set the default model for the active provider.

```bash
twin config set-model gpt-4o-mini
twin config set-model anthropic/claude-sonnet-4-5   # OpenRouter format
```

### `twin config list`

Show the active provider, configured keys (yes/no — never the values), and default models.

```bash
twin config list
```

```
Active provider: anthropic
┌────────────┬─────┬──────────────┬──────────────────────────┐
│ Provider   │ Key │ Source       │ Default model            │
├────────────┼─────┼──────────────┼──────────────────────────┤
│ anthropic  │ yes │ keychain     │ —                        │
│ openai     │ no  │ —            │ gpt-4o-mini              │
│ gemini     │ no  │ —            │ —                        │
│ ollama     │ —   │ —            │ llama3.2                 │
│ openrouter │ no  │ —            │ —                        │
└────────────┴─────┴──────────────┴──────────────────────────┘
```

### `twin config list-models`

List available models for the active provider. Ollama makes a live call to the local API.

```bash
twin config list-models
twin config list-models --provider ollama
```

---

## `twin watch` — Obsidian vault watcher

Watches a vault directory for `.md` file saves and automatically re-ingests changed files after a 500ms debounce. Agent-generated notes in `<vault>/Agents/` are also ingested.

```bash
twin watch ~/my-obsidian-vault
```

```
Watching ~/my-obsidian-vault for .md changes. Log: ~/.twin/watcher.log  (Ctrl-C to stop)
```

Check whether the watcher is running:

```bash
twin watch --status
```

Stop the watcher with `Ctrl-C`. The PID file at `~/.twin/watcher.pid` is cleaned up automatically.

---

## `twin usage` — Token and cost history

Displays a table of LLM usage grouped by date and provider. Data is read from `~/.twin/usage.jsonl`, which is appended after every `twin rag` and `twin agent` call.

```bash
twin usage
```

```
┌────────────┬───────────┬───────┬───────────────┬───────────────────┬──────────┐
│ Date       │ Provider  │ Calls │ Prompt tokens  │ Completion tokens │ Est. cost│
├────────────┼───────────┼───────┼───────────────┼───────────────────┼──────────┤
│ 2026-06-01 │ anthropic │ 14    │ 18,430         │ 3,210             │ $0.0421  │
│ 2026-06-01 │ ollama    │ 3     │ 2,100          │ 680               │ local    │
└────────────┴───────────┴───────┴───────────────┴───────────────────┴──────────┘
```

A one-line summary is also printed at the end of every `rag` and `agent` command:

```
3 calls · 1,240 tokens · ~$0.003       # paid provider
2 calls · 890 tokens · local (no cost) # Ollama
```

---

## Configuration

Settings are read from `TWIN_*` environment variables. `SECONDBRAIN_*` names are supported as a deprecated fallback.

| Variable | Default | Description |
|---|---|---|
| `TWIN_DATA_DIR` | `~/.twin` | LanceDB, SQLite, keychain, usage log |
| `TWIN_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `TWIN_CHUNK_TOKENS` | `512` | Max tokens per chunk |
| `TWIN_OVERLAP` | `64` | Overlap tokens between chunks |
| `TWIN_TOP_K` | `5` | Default results for `query` and `rag` |
| `TWIN_PROVIDER` | `anthropic` | Active LLM provider fallback |
| `TWIN_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `ANTHROPIC_API_KEY` | — | Anthropic key (keychain takes priority) |
| `OPENAI_API_KEY` | — | OpenAI key (keychain takes priority) |
| `GEMINI_API_KEY` | — | Gemini key (keychain takes priority) |
| `OPENROUTER_API_KEY` | — | OpenRouter key (keychain takes priority) |

---

## Typical Workflows

### Quick start — Anthropic

```bash
twin config set-key              # store key once, encrypted
twin ingest ~/notes              # index your notes
twin query "deployment checklist"
twin rag "What are my CI/CD best practices?"
twin agent "Summarize my Python performance notes"
```

### Fully local — Ollama

```bash
# Requires Ollama running with a model pulled
twin config set-provider ollama
twin config set-model llama3.2

twin ingest ~/notes
twin rag "What did I write about caching strategies?"
twin usage                       # shows "local (no cost)"
```

### Obsidian vault with write-back

```bash
twin watch ~/vault               # terminal 1 — watches for changes
twin agent "Draft a summary of my ML reading list"
# → agent writes a note to ~/vault/Agents/Draft-a-summary.../...md
# → watcher picks up the new note and re-ingests it
```

### Multi-provider comparison

```bash
twin rag "What is the CAP theorem?" --provider anthropic
twin rag "What is the CAP theorem?" --provider openai
twin rag "What is the CAP theorem?" --provider ollama
twin usage                       # see cost comparison
```

---

## Running Tests

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

363 tests. The first run downloads the embedding model if not already cached.
