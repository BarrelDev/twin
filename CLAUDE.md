# CLAUDE.md — Twin

This file is read by Claude Code at the start of every session.
It defines architectural decisions, conventions, and constraints.
Do not override these without explicit instruction from the project author.
Do not create extraneous documentation without cause.

---

## Current Phase: 2

Phase 0 (retrieval core) — Complete
Phase 1 (RAG loop + basic agent) — Complete
Phase 2 (multi-provider, keychain, Obsidian sync, expanded ingestion) — In Progress

Do not implement anything from Phase 3 unless explicitly asked.

---

## Project Overview

Twin is a local-first knowledge OS with agent execution. Users ingest Markdown notes,
PDFs, and URLs into a local vector store, then query them with natural language or
run agents that reason across the knowledge base.

**The full CLI surface as of Phase 2:**
```
twin ingest <path|url>          # ingest Markdown, PDF, or URL
twin query "<q>"                # raw semantic search (debug tool)
twin rag "<q>"                  # RAG: retrieve + synthesize + cite
twin agent "<task>"             # multi-step agent with KB search + vault write-back
twin watch <vault-path>         # Obsidian vault watcher (background)
twin config set-key             # interactive encrypted key setup
twin config set-provider <p>    # set active LLM provider
twin config set-model <m>       # set default model for active provider
twin config list                # show current config (never reveal key values)
twin config list-models         # list models for active provider
twin config remove-key <p>      # remove a provider's key
twin usage                      # token and cost summary by provider/day
```

---

## Directory Structure

```
twin/
  __init__.py
  config.py               AppConfig dataclass, env-var overrides, enums
  config_manager.py       Encrypted keychain + config.json read/write
  usage.py                Token/cost logging and twin usage reporting
  cli.py                  Typer CLI — all commands defined here
  ingestion/
    __init__.py
    parser.py             Markdown chunking (calls twin_core Rust extension)
    embedder.py           sentence-transformers wrapper, prefix handling
    pdf.py                pymupdf-based PDF parser
    url.py                trafilatura-based URL ingester
    obsidian.py           Wikilink/tag/frontmatter parser + watchdog watcher
  storage/
    __init__.py
    vector.py             LanceDB schema, ANN search, source filtering
    metadata.py           SQLite document registry, SHA-256 dedup
  query/
    __init__.py
    retriever.py          Search orchestration, ranking, Rich output
  llm/
    __init__.py
    base.py               Abstract LLMProvider interface
    anthropic.py          Anthropic implementation
    openai.py             OpenAI implementation
    gemini.py             Gemini implementation
    ollama.py             Ollama (local) implementation
    openrouter.py         OpenRouter implementation
  rag/
    __init__.py
    pipeline.py           Retrieve → format → generate → return (streaming)
    context.py            Chunk formatting with source attribution
    prompts.py            System prompt definitions
  agent/
    __init__.py
    runtime.py            Multi-step tool-using loop, streaming final answer
    tools.py              search_knowledge_base + write_vault_note tools
    log.py                AgentLog: chronological event log, JSON-serializable
twin_core/
  Cargo.toml
  src/
    lib.rs                PyO3 bindings
    chunker.rs            Heading-aware chunking logic
    tokens.rs             Token counting (word-based)
tests/
  conftest.py             Shared fixtures — mock embedder, tmp LanceDB, tmp vault
  test_parser.py
  test_embedder.py
  test_vector.py
  test_metadata.py
  test_retriever.py
  test_llm.py
  test_context.py
  test_rag_pipeline.py
  test_agent_tools.py
  test_agent_log.py
  test_agent_runtime.py
  test_pdf.py             new in Phase 2
  test_url.py             new in Phase 2
  test_obsidian.py        new in Phase 2
  test_providers.py       new in Phase 2
  test_config_manager.py  new in Phase 2
  test_usage.py           new in Phase 2
pyproject.toml
CLAUDE.md
README.md
setup.sh
setup.bat
```

Do not create files outside this structure without asking first.

---

## Tech Stack — Fixed Decisions

Do not suggest alternatives to these choices.

| Concern | Choice | Do NOT use |
|---|---|---|
| Language | Python 3.11+, Rust (chunking hot path) | — |
| Vector store | LanceDB | ChromaDB, Pinecone, Weaviate, FAISS |
| Embeddings | sentence-transformers (nomic-embed-text-v1.5) | OpenAI embeddings, Cohere |
| Metadata store | SQLite via SQLModel | PostgreSQL, MongoDB, raw sqlite3 |
| CLI | Typer | argparse, click, Fire |
| Terminal output | Rich | print(), logging to stdout |
| Testing | pytest | unittest |
| Dependency mgmt | uv + pyproject.toml | pip + requirements.txt, Poetry |
| Abstraction libs | None | LangChain, LlamaIndex, Haystack |
| PDF extraction | pymupdf | pdfplumber, pypdf, pdfminer |
| Web extraction | trafilatura | BeautifulSoup, newspaper3k, requests+bs4 |
| Encryption | cryptography (PyCA) | pycryptodome, Fernet alone |
| Filesystem watch | watchdog | polling loops, inotify directly |
| HTTP client | httpx | requests (for new code) |

---

## LLM Provider Architecture

### The interface contract (llm/base.py)
Every provider implements exactly these methods. Do not add methods to the
interface without updating all concrete implementations.

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        system: str | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None: ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]: ...
```

### Provider resolution order (do not change)
1. Explicit --provider flag on the CLI command
2. Active provider set via twin config set-provider (stored in config.json)
3. TWIN_PROVIDER environment variable
4. Anthropic (default fallback)

### Model capability detection
Every ModelInfo must include supports_tools: bool.
If the active model does not support tools, twin agent must fail with a
descriptive error — never silently fall back to a tool-less mode.

### Ollama special handling
- No API key. Do not prompt for one or check the keychain.
- Base URL is configurable: TWIN_OLLAMA_URL, default http://localhost:11434
- list_models() makes a live HTTP call to the Ollama API — do not hardcode models.

### OpenRouter
- Single key gives access to multiple underlying providers.
- Model name format: provider/model (e.g. anthropic/claude-sonnet-4-5)
- list_models() fetches the OpenRouter model list from their API.

---

## Encrypted Keychain

### Storage locations
- ~/.twin/keychain.enc   AES-256-GCM encrypted key-value store
- ~/.twin/config.json    Non-sensitive config (plaintext JSON)
- ~/.twin/usage.jsonl    Token/cost log (plaintext JSONL, one record per line)

### Encryption rules
- Keys are encrypted with AES-256-GCM
- The encryption key is derived from username + machine ID via PBKDF2
- The keychain is intentionally non-portable (machine-bound)
- Never log, print, or return a raw API key anywhere in the codebase
- twin config list shows which providers have keys — never the key values

### Key resolution order (do not change)
1. Keychain (highest priority)
2. Environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
3. Raise a descriptive error with onboarding instructions

### First-run onboarding
If no key is found for the active provider, print a clear message:
  - Which provider was requested
  - The twin config set-key command to fix it
  - The fallback option (twin config set-provider ollama for no-key local use)
Do NOT raise a bare exception or Python traceback.

---

## Obsidian Integration

### The hard boundary rule — never violate this
Twin NEVER writes outside <vault>/Agents/.
This is enforced at the path level in write_vault_note, not as a convention.
Sanitize all agent-provided titles before constructing file paths.
Reject any path that would escape the Agents/ directory.

### Write-back format
Agent output notes follow this structure:
  <vault>/Agents/<task-slug>/<ISO-timestamp>-<sanitized-title>.md

Frontmatter on every agent-generated note:
```yaml
---
generated_by: twin-agent
task: <original task string>
created: <ISO 8601 timestamp>
tags: [twin-generated]
---
```

### Watcher behavior
- Debounce: 500ms delay before triggering re-ingest on file change
- Scope: watches *.md files only — ignore all other extensions
- Agents/ folder: watcher DOES ingest agent-generated notes (intentional)
- twin watch --status shows: running/stopped, last event, Agents/ note count

### Obsidian-specific metadata (stored in SQLite, not just vector store)
- link_targets: list of wikilink targets extracted from the note
- tags: list of Obsidian tags (including nested #parent/child tags)
- full frontmatter: all YAML fields, not just id

---

## Streaming

### Which commands stream
- twin rag: streams synthesized answer; sources printed after stream completes
- twin agent: streams final answer; tool call activity printed inline as it happens
- twin query: does NOT stream (returns ranked chunks, no generation)
- twin ingest: does NOT stream (Rich progress bar is sufficient)

### Streaming rules
- Use Rich Live for rendering streamed output — never write to stdout directly
- Sources for twin rag are collected during streaming and deduplicated before printing
- Never print the sources section until the stream is complete
- Tool calls interrupt the stream: flush partial response, execute tool, resume stream
- The activity log captures interruption points

---

## Cost and Token Tracking

Every LLM call must log to ~/.twin/usage.jsonl. The record schema:
```python
@dataclass
class UsageRecord:
    timestamp: str          # ISO 8601
    command: str            # "rag" | "agent"
    provider: str           # "anthropic" | "openai" | etc.
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float | None  # None for Ollama
```

At the end of every rag or agent command, print a one-line summary:
  "3 calls · 1,240 tokens · ~$0.003"
For Ollama: "2 calls · 890 tokens · local (no cost)"

---

## Ingestion Rules

### Idempotency (do not break this)
Running ingest twice on unchanged content produces no changes.
SHA-256 hash of file content (or URL content) is stored in metadata.
On re-ingest: hash match → skip; hash changed → delete old chunks, insert new.

### Format routing (in cli.py ingest command)
- Starts with http:// or https:// → url.py
- Extension .pdf → pdf.py
- Extension .md or .txt → parser.py
- Explicit --type flag overrides detection

### PDF source attribution format
"document.pdf › p.4"  (not just filename)

### URL source attribution format
"domain.com › Page Title"

### Wikilink extraction (obsidian.py)
Extract [[Note Name]] and [[Note Name|Alias]] patterns.
Store note name (not alias) as link target in metadata.
Strip wikilinks from chunk text before embedding — embed clean prose.

---

## Code Conventions

### Type hints are required everywhere
```python
# CORRECT
def complete(self, messages: list[Message], tools: list[ToolDef] | None = None) -> LLMResponse: ...

# WRONG
def complete(self, messages, tools=None): ...
```

### Use dataclasses for structured data, not dicts
```python
# CORRECT
@dataclass
class UsageRecord:
    timestamp: str
    provider: str
    ...

# WRONG
record = {"timestamp": ..., "provider": ...}
```

### No magic strings — use enums or constants
```python
# CORRECT
class Provider(str, Enum):
    ANTHROPIC  = "anthropic"
    OPENAI     = "openai"
    GEMINI     = "gemini"
    OLLAMA     = "ollama"
    OPENROUTER = "openrouter"

# WRONG
provider = "anthropic"
```

### Async for all LLM calls
All provider methods are async. Use asyncio.run() at the CLI boundary.
Do not mix sync and async LLM calls.

### Docstrings on all public functions
Follow the Google style used throughout the existing codebase:
```python
def write_vault_note(self, title: str, content: str, tags: list[str] | None = None) -> Path:
    """
    Write an agent-generated note to the Obsidian vault.

    Writes to <vault>/Agents/<task-slug>/<timestamp>-<title>.md.
    Never writes outside the Agents/ directory.

    Args:
        title: Note title. Used as filename (sanitized) and H1 heading.
        content: Markdown body content.
        tags: Optional list of Obsidian tags.

    Returns:
        Path to the created file, relative to vault root.

    Raises:
        ValueError: If the sanitized path would escape Agents/.
    """
```

### Functions do one thing
If a function does more than one conceptual operation, split it.

---

## Testing Requirements

- Every module has a corresponding test file
- Use tmp_path for all file and DB operations — never write to real paths
- Mock all LLM provider calls in tests — never make real API calls in the test suite
- Mock the embedding model in tests that do not specifically test embedding quality
- The retrieval quality test in test_retriever.py remains the correctness bar:
  correct chunk in top-3 for at least 8/10 known-answer queries
- Aim for >80% line coverage across all modules

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

### Test fixtures (conftest.py)
These fixtures must exist and be used consistently:
- tmp_lance_db: temporary LanceDB instance, cleaned up after test
- tmp_sqlite: temporary SQLite metadata store
- mock_embedder: returns deterministic fake embeddings (no model loaded)
- mock_llm_provider: returns canned responses, records calls made
- tmp_vault: temporary directory structured as an Obsidian vault

---

## What Claude Code Should Always Do

- Write type hints on every function signature
- Write a docstring on every public function
- Write or update the corresponding test when implementing anything
- Use Rich for all terminal output
- Use pathlib.Path for all file paths
- Use async/await for all LLM provider calls
- Check the existing codebase before implementing — ask if a similar function exists
- Run the relevant tests after implementing to verify
- Note any new dependencies explicitly before adding them

## What Claude Code Should Never Do

- Import from LangChain, LlamaIndex, or any abstraction framework
- Make real API calls in tests
- Write to real filesystem paths in tests
- Log, print, or return raw API key values anywhere
- Write agent output outside <vault>/Agents/
- Use dict where a dataclass is clearer
- Skip type hints for brevity
- Suggest switching the vector DB, embedding model, or metadata store
- Add new methods to LLMProvider base without updating all implementations
- Break idempotent ingest behavior

---

## Rust Extension (twin_core)

The chunking hot path lives in twin_core/ as a PyO3-compiled Rust extension.

### The boundary rule
Rust receives plain strings, returns plain data.
It does not touch LanceDB, SQLite, the filesystem, or any Python objects
beyond what PyO3 marshals automatically.

### Fallback import (parser.py)
```python
try:
    from twin_core import Chunk, chunk_text as _rust_chunk_text
    _USE_RUST = True
except ImportError:
    _USE_RUST = False
```
The Python fallback must remain correct and tested. Never remove it.

### Build
```bash
cd twin_core && uv run maturin develop  # dev
cd twin_core && uv run maturin build --release  # production
```

### Tests
The Python test suite (test_parser.py) is the correctness source of truth.
The Rust implementation must pass all existing parser tests.

---

## Config Environment Variables

```
TWIN_DATA_DIR          where LanceDB, SQLite, keychain live — default: ~/.twin
TWIN_EMBED_MODEL       embedding model — default: nomic-ai/nomic-embed-text-v1.5
TWIN_CHUNK_TOKENS      max tokens per chunk — default: 512
TWIN_OVERLAP           overlap tokens — default: 64
TWIN_TOP_K             results per query — default: 5
TWIN_PROVIDER          active LLM provider — default: anthropic
TWIN_OLLAMA_URL        Ollama base URL — default: http://localhost:11434

# API keys (fallback if not in keychain)
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
OPENROUTER_API_KEY
```

Note: SECONDBRAIN_* variable names from earlier phases are deprecated.
Use TWIN_* going forward. Maintain backward-compatible fallback reads during Phase 2.

---

## Phase 3 Preview (do not implement yet)

- Prebuilt native binaries via PyApp + GitHub Actions release workflow
- Graph-aware retrieval using the wikilink link_targets metadata
- Tag-based query filtering (twin query --tag #project/twin)
- Homebrew / winget / scoop packaging
- Agent memory across sessions