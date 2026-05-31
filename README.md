# twin

**Twin** is a local-first knowledge base with semantic search and LLM-augmented retrieval. It ingests a folder of Markdown notes, embeds them with a locally-run transformer model, and lets you query them with natural language — no cloud dependency, no tracking, no per-query cost.

---

## What It Does

```
$ twin ingest ./notes
Scanning ./notes ... 142 files found
Chunking ... 1,203 chunks generated
Embedding ... done (47s, nomic-embed-text-v1.5)
Stored to ~/.twin

$ twin query "What did I write about the Rust ownership model?"
─────────────────────────────────────────────────────────────────
Result 1  [score: 0.91]  notes/languages/rust.md › Ownership\
  "Ownership in Rust is the mechanism by which the compiler\
   tracks which variables are responsible for freeing memory..."
─────────────────────────────────────────────────────────────────

$ twin rag "What did I write about the Rust ownership model?"
╭─ Answer ──────────────────────────────────────────────────╮
│ Rust's ownership model gives each value a single owner... │
╰───────────────────────────────────────────────────────────╯

Sources:
  • rust.md  Ownership
  • systems.md  Memory Safety

$ twin agent "Summarize everything I know about async Rust"
╭─ Agent Answer ────────────────────────────────╮
│ Based on your notes, async Rust centers on... │
╰───────────────────────────────────────────────╯
```


Tool calls made: 3

All four commands are fully functional. For more information on how to use these commands, read [HOW_TO_USE.md](HOW_TO_USE.md)

---

## Architecture

```
Markdown files
      │
      ▼
 parser.py          Heading-first chunking (512-token budget, 64-token overlap),
 [twin-core/        frontmatter extraction. Chunking hot path implemented in Rust
  chunker.rs]       via PyO3 bindings — same pattern as Hugging Face tokenizers.
      │
      ▼
 embedder.py        nomic-embed-text-v1.5 (768-dim, MTEB top-5 for retrieval).
                    Applies task-specific prefixes: search_document: vs search_query:.
      │
      ▼
 vector.py          LanceDB persistent store. ANN search, source_path filtering,
 metadata.py        metadata preserved. SHA-256 hash registry (SQLite) for
                    idempotent ingestion — running ingest twice on unchanged
                    files produces no changes.
      │
      ▼
 retriever.py       Orchestrates search, ranks results, Rich-formatted output.
      │
      ▼
 rag/pipeline.py    [Phase 1] Retrieve → format context with attribution →
                    LLM synthesis → grounded answer + deduplicated sources.
      │
      ▼
 agent/runtime.py   [Phase 1] Multi-step tool-using loop. LLM decides when
                    to search the KB (up to 5 iterations), chains retrievals,
                    and returns a final answer. All tool calls logged.
```

---

## Design Decisions

### No abstraction frameworks

Zero LangChain, LlamaIndex, or similar. Every component is a thin wrapper around its underlying library — LanceDB, sentence-transformers, SQLite, Anthropic API.

**Why:** Abstractions obscure what's happening during retrieval, make debugging harder, and add dependencies with frequent breaking changes. Every retrieval failure in Twin is traceable: query → embedding → ANN search → ranking → formatting. No framework magic in the path.

**Trade-off:** More code, more understanding. The architecture demonstrates knowledge of underlying systems, not reliance on black-box wrappers.

---

### Embedding model selection (evidence-based)

**Choice:** `nomic-ai/nomic-embed-text-v1.5` (768 dimensions)

Benchmarked against the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Ranks top 5 for retrieval tasks among locally-runnable models. 768 dimensions is the sweet spot: higher accuracy than smaller models, lower storage cost than 1536-dim alternatives.

**The non-obvious detail:** The model requires task-specific prefixes — `search_document:` for ingestion, `search_query:` for queries. Most integrations skip this. Twin handles it explicitly because the distinction is measurable in retrieval quality.

**Measured bar:** Correct chunk in top-3 results for 10/10 known queries against a 50-document corpus.

---

### Idempotent ingestion

Running `ingest` twice on unchanged files produces no changes. SHA-256 hashes of file content are stored in the SQLite metadata registry. On ingest: hash match → skip; hash changed → delete old chunks, insert new ones.

This is a data pipeline correctness requirement, not a performance optimization. Without idempotency, repeated ingest silently accumulates duplicates and returns stale results. The behavior is covered by tests in `test_metadata.py`.

---

### Rust for the chunking hot path

Chunking and token counting live in `twin-core/` — a Rust crate exposed via PyO3 bindings. The boundary is clean: Rust receives plain strings, returns text + offsets. It does not touch LanceDB, SQLite, the filesystem, or any Python objects beyond what PyO3 marshals automatically.

**Pattern:** Same approach used by Hugging Face tokenizers. The Python test suite (`test_parser.py`) is the source of truth for correctness — the Rust implementation must pass all existing tests.

---

### Provider-agnostic LLM interface

`llm/base.py` defines an abstract `LLMProvider` interface with three methods: `complete()`, `extract_answer()`, and `list_models()`. The Anthropic implementation (`llm/anthropic.py`) is the only concrete provider in Phase 1.

**Why it matters:** The RAG pipeline and agent runtime only know about the abstract interface. Adding OpenAI, Google, or local Ollama in Phase 2 requires zero changes to retrieval or agent code. The interface was designed with the seam in mind before any implementation existed.

---

### Chunking parameters (documented, not magic)

| Parameter | Value | Reason |
|---|---|---|
| Max chunk tokens | 512 | Precision vs. context trade-off — smaller is sharper |
| Overlap tokens | 64 | Prevents context loss at chunk boundaries |
| Primary split | Markdown headings | Semantic units, not arbitrary length |
| Secondary split | Paragraph breaks | Natural prose boundaries |

---

## Current Status

### Stage 0 — Complete ✓

All retrieval-core components implemented, tested, and wired into the CLI.

| Module | Description | Status |
|---|---|---|
| `ingestion/parser.py` | Heading-aware chunking, overlap, frontmatter extraction | ✓ |
| `ingestion/embedder.py` | nomic-embed-text-v1.5, document + query prefix handling | ✓ |
| `storage/vector.py` | LanceDB schema, ANN search, source filtering, persistence | ✓ |
| `storage/metadata.py` | SQLite document registry, SHA-256 hash dedup | ✓ |
| `query/retriever.py` | Search orchestration, result ranking, Rich output | ✓ |
| `cli.py` | `twin ingest` + `twin query` end-to-end | ✓ |
| `twin-core/` | Rust crate for chunking + token counting (PyO3) | ✓ |

---

### Phase 1 — Complete ✓

| Component | File | Status |
|---|---|---|
| LLM provider abstraction | `llm/base.py`, `llm/anthropic.py` | ✓ |
| Context formatter | `rag/context.py` | ✓ |
| RAG pipeline | `rag/pipeline.py` | ✓ |
| System prompts | `rag/prompts.py` | ✓ |
| KB search tool + dispatch | `agent/tools.py` | ✓ |
| Agent runtime | `agent/runtime.py` | ✓ |
| Activity log | `agent/log.py` | ✓ |
| CLI: `twin rag <query>` | `cli.py` | ✓ |
| CLI: `twin agent <task>` | `cli.py` | ✓ |

#### Phase 1 flow

```
twin rag "query"
  └─ RAGPipeline.query()
       ├─ Retriever.query()          — ANN search, top-k results
       ├─ prepare_rag_context()      — format chunks with source attribution
       ├─ LLMProvider.complete()     — synthesize from context only
       └─ RAGOutput: answer + sources

twin agent "task"
  └─ AgentRuntime.execute()
       └─ Iteration loop (max 5):
            ├─ LLMProvider.complete() + tool definitions
            ├─ Tool call detected?
            │    ├─ Yes: ToolDispatcher.dispatch() → search_knowledge_base()
            │    │        AgentLog.log_tool_call()
            │    │        continue loop
            │    └─ No:  extract final answer
            │             AgentLog.log_final_answer()
            │             return AgentOutput
            └─ AgentOutput: final_answer | tool_calls | activity_log
```

---

### Phase 2+ — Planned

- Multi-provider LLM switching (OpenAI, Google, Groq, Ollama) — interface already designed
- Obsidian vault watcher with automatic re-ingestion
- PDF and URL ingestion
- Streaming responses
- Token accounting and cost tracking

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+, Rust (chunking hot path) |
| Embeddings | sentence-transformers (nomic-embed-text-v1.5) |
| Vector store | LanceDB |
| Metadata store | SQLite via SQLModel |
| LLM | Anthropic API (Claude) |
| CLI | Typer |
| Terminal output | Rich |
| Testing | pytest |
| Dependency management | uv |
| Python-Rust bindings | PyO3 / maturin |

---

## Project Structure

```
twin/
  config.py               AppConfig dataclass, env-var overrides, model enums
  cli.py                  Typer CLI (ingest, query, rag, agent)
  ingestion/
    parser.py             Chunking, heading extraction, frontmatter parsing
    embedder.py           sentence-transformers wrapper, prefix handling
  storage/
    vector.py             LanceDB schema, write, ANN search, source filtering
    metadata.py           SQLite document registry, SHA-256 dedup
  query/
    retriever.py          Search orchestration, ranking, Rich table output
  llm/
    base.py               Abstract LLMProvider interface
    anthropic.py          Anthropic Claude implementation
  rag/
    context.py            Chunk formatting with source attribution
    pipeline.py           Retrieve → format → synthesize → return
    prompts.py            System prompt definitions
  agent/
    tools.py              KB search tool definition + ToolDispatcher
    runtime.py            Multi-step tool-using loop, AgentOutput
    log.py                AgentLog: chronological event log, JSON-serializable
twin-core/
  Cargo.toml
  src/
    lib.rs                PyO3 bindings
    chunker.rs            Heading-aware chunking logic
    tokens.rs             Token counting
tests/
  conftest.py             Shared fixtures (tmp_path, mock embedder, etc.)
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
```

---

## Build and Run

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Rust toolchain (`rustup`) — required to build `twin-core`

### Install

```bash
git clone https://github.com/BarrelDev/twin
cd twin

# Install Python dependencies
uv sync

# Build the Rust extension (twin-core)
cd twin_core
pip install maturin
maturin develop
cd ..
```

> The first test run downloads `nomic-embed-text-v1.5` (~270 MB) from Hugging Face. It is cached locally after that.

### Run the CLI

```bash
# Ingest a folder of Markdown notes
uv run python -m twin ingest ./notes

# Query by semantic similarity (no API key needed)
uv run python -m twin query "What did I write about decorators?"

# RAG: synthesize a grounded answer with source attribution
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python -m twin rag "What is the Rust ownership model?"

# Agent: multi-step reasoning with iterative KB search
uv run python -m twin agent "Summarize what I've written about Python decorators"

# Agent with verbose activity log and custom iteration cap
uv run python -m twin agent "..." --verbose --max-iter 3
```

### Run tests

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

---

## Configuration

All runtime config is read from environment variables with sensible defaults. No config files to manage.

| Variable | Default | Description |
|---|---|---|
| `SECONDBRAIN_DATA_DIR` | `~/.twin` | Where LanceDB and SQLite are stored |
| `SECONDBRAIN_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `SECONDBRAIN_CHUNK_TOKENS` | `512` | Max tokens per chunk |
| `SECONDBRAIN_OVERLAP` | `64` | Overlap tokens between chunks |
| `SECONDBRAIN_TOP_K` | `5` | Results returned per query |
| `ANTHROPIC_API_KEY` | (required for Phase 1) | Anthropic API key |

---

## Engineering Highlights

For a recruiting context, here's what this project demonstrates:

**Systems thinking:** Idempotent data pipelines (SHA-256 hash dedup), compute hot path optimization (Rust via PyO3), and provider abstraction that's designed for multi-vendor extensibility before the second vendor exists.

**Evidence-based decisions:** Embedding model selection backed by MTEB benchmark rankings. Chunking parameters chosen for measurable retrieval quality, documented as constants with justification, not magic numbers.

**Composition over frameworks:** Every layer is independently testable. The retriever, pipeline, and agent runtime can each be instantiated with mock dependencies. No global state, no framework side-channels.

**Production mindset:** Activity logging with ISO 8601 timestamps and JSON serialization for audit trails. Iteration limits on the agent loop. Clear error messages when API keys are missing. Graceful termination on max iterations with partial results preserved.

**Full-stack capability:** Python application layer, Rust extension for compute-bound work, SQL data pipeline, LLM API integration, CLI tooling.

**Type discipline:** 100% type-annotated function signatures across the codebase. `mypy`-compatible. Types serve as documentation and catch bugs before runtime.
