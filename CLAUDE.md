# CLAUDE.md — Twin

This file defines architectural decisions, conventions, and constraints.
Do not override these without explicit instruction from the project author.
Do not create extraneous documentation without cause.

---

## Project Overview

Twin is a local-first knowledge base with agent execution.
This repo contains **Stage 0 only**: the retrieval core (ingest + query).
There is no UI, no agent runtime, no API server, and no cloud dependency.

Stage 0 goal: prove that semantic search over personal notes is good enough to build on.

**The two CLI commands that define Stage 0 done:**
```
python -m twin ingest ./notes
python -m twin query "What did I write about X?"
```

---

## Directory Structure

```
twin/
  __init__.py
  config.py         # AppConfig dataclass, loaded from env or defaults
  cli.py            # Typer CLI entrypoint
  ingestion/
    __init__.py
    parser.py       # file reading, chunking, frontmatter parsing
    embedder.py     # sentence-transformers wrapper
  storage/
    __init__.py
    vector.py       # LanceDB read/write interface
    metadata.py     # SQLite document registry via SQLModel
  query/
    __init__.py
    retriever.py    # search logic, result ranking, formatting
  llm/              # Phase 1: LLM abstraction
    __init__.py
    base.py         # Abstract LLMProvider interface
    anthropic.py    # Anthropic API implementation
  rag/              # Phase 1: RAG pipeline
    __init__.py
    pipeline.py     # retrieve → format → generate → return
    context.py      # chunk formatting and attribution
    prompts.py      # system prompt definitions
  agent/            # Phase 1: Agent runtime
    __init__.py
    runtime.py      # tool-using LLM loop
    tools.py        # tool definitions and dispatch
    log.py          # activity log
tests/
  test_parser.py
  test_embedder.py
  test_vector.py
  test_metadata.py
  test_retriever.py
  test_rag_pipeline.py     # Phase 1
  test_agent_runtime.py    # Phase 1
  test_llm_client.py       # Phase 1
  conftest.py       # shared fixtures
pyproject.toml
CLAUDE.md
README.md
```

Do not create files outside this structure without asking first. Phase 1 directories are already mapped above.

---

## Tech Stack — Fixed Decisions

These are not up for debate in Stage 0. Do not suggest alternatives.

| Concern | Choice | Do NOT use |
|---|---|---|
| Language | Python 3.11+ | — |
| Vector store | LanceDB | ChromaDB, Pinecone, Weaviate, FAISS |
| Embeddings | sentence-transformers (nomic-embed-text-v1.5) | OpenAI embeddings, Cohere |
| Metadata store | SQLite via SQLModel | PostgreSQL, MongoDB, raw sqlite3 |
| CLI | Typer | argparse, click, Fire |
| Terminal output | Rich | print(), logging to stdout |
| Testing | pytest | unittest |
| Dependency mgmt | uv + pyproject.toml | pip + requirements.txt, Poetry |
| Abstraction libs | None | LangChain, LlamaIndex, Haystack |

---

## Code Conventions

### Type hints are required everywhere
```python
# CORRECT
def parse_file(path: Path) -> list[Chunk]:
    ...

# WRONG — never do this
def parse_file(path, chunks):
    ...
```

### Use dataclasses, not dicts, for structured data
```python
# CORRECT
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    token_count: int

# WRONG
chunk = {"text": "...", "source": "..."}
```

### No magic strings — use enums or constants
```python
# CORRECT
class EmbeddingModel(str, Enum):
    NOMIC = "nomic-ai/nomic-embed-text-v1.5"

# WRONG
model = "nomic-ai/nomic-embed-text-v1.5"
```

### Functions should do one thing
If a function is doing more than one conceptual operation, split it.
Prefer many small functions over few large ones.

### Docstrings on all public functions
```python
def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document passages.

    Uses the 'search_document' prefix required by nomic-embed-text.
    Batches internally for efficiency.

    Args:
        texts: List of passage strings to embed.

    Returns:
        List of embedding vectors, same order as input.
    """
```

---

## Chunking Parameters (do not change without discussion)

| Parameter | Value | Reason |
|---|---|---|
| Max chunk tokens | 512 | Balances context vs. retrieval precision |
| Overlap tokens | 64 | Avoids cutting context at chunk boundaries |
| Primary split | Markdown headings | Semantic units, not arbitrary length |
| Secondary split | Paragraph breaks | Natural prose boundaries |
| Metadata on chunk | source_path, heading_path, chunk_index, token_count | Required for source attribution |

---

## Ingestion Must Be Idempotent

Running `ingest` twice on the same unchanged file must produce no changes.

Implementation: store SHA-256 hash of file content in the metadata store.
On ingest, check hash. Skip if unchanged. Update if changed (delete old chunks, insert new).

This is tested in test_metadata.py. Do not break this behavior.

---

## Testing Requirements

- Every module must have a corresponding test file
- Tests must use pytest fixtures defined in conftest.py
- Use tmp_path fixture for any file/DB operations — never write to real paths in tests
- The retrieval quality test in test_retriever.py is the most important:
  - Ingest a known 50-document corpus
  - Run 10 queries where the correct answer is known
  - Assert correct chunk appears in top-3 results for at least 8/10 queries
- Aim for >80% line coverage on all core modules

Run tests:
```
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

---

## What Claude Code Should Always Do

- Write type hints on every function signature
- Write a docstring on every public function
- Write or update the corresponding test when implementing a function
- Use Rich for any terminal output (not print)
- Use pathlib.Path for file paths (not str)
- Check if the function already exists before implementing — ask if unclear
- Keep functions small and single-purpose
- Run the relevant test after implementing to verify
- When implementing Phase 1 features: test against the Anthropic API using a real API key in the test environment

## What Claude Code Should Never Do

- Import from LangChain, LlamaIndex, or any high-abstraction AI framework
- Use OpenAI API or any external API in Stage 0 (Phase 1 uses Anthropic only)
- Add new dependencies without noting them explicitly — list them so the author can approve
- Create files outside the defined directory structure
- Write to real filesystem paths in tests
- Use `dict` where a `dataclass` would be clearer
- Skip type hints "for brevity"
- Suggest switching the vector DB, embedding model, or metadata store
- Hardcode model names or API keys — always read from config or environment variables

---

## Phase 1 Specific Notes

When implementing RAG and agent features, keep these constraints in mind:

### LLM Client
- The LLMProvider interface in `llm/base.py` must remain provider-agnostic
- Anthropic is the only implementation in Phase 1; design the interface so Phase 2 can add OpenAI, Google, etc. without modifying RAG or agent code
- Always read `ANTHROPIC_API_KEY` from environment; raise clear error if missing

### RAG Pipeline
- System prompt must enforce three constraints: answer only from context, cite sources, admit uncertainty when appropriate
- Context formatting in `rag/context.py` must preserve source attribution (filename + heading path) on every chunk
- Chunk token budget: 2560 tokens for top-5 chunks (512 tokens each)
- All outputs must include a deduplicated source list alongside the answer

### Agent Runtime
- Tool definitions must be precise and clear — the model must understand exactly when and how to use the KB search tool
- Agent loop has two termination conditions: final answer without tool call, or max iteration count (default: 5)
- Activity log must track every tool call, its query, results, and final answer — this is for debugging and transparency
- Do not implement write-back to knowledge base in Phase 1 — agents are read-only

---

## Config

All runtime config lives in `twin/config.py` as an `AppConfig` dataclass.
Loaded from environment variables with sensible defaults.
No hardcoded paths or model names outside of config.py and the enums in their respective modules.

**Stage 0 — Retrieval config:**
```
SECONDBRAIN_DATA_DIR     # where LanceDB and SQLite live, default: ~/.twin
SECONDBRAIN_EMBED_MODEL  # embedding model, default: nomic-ai/nomic-embed-text-v1.5
SECONDBRAIN_CHUNK_TOKENS # max tokens per chunk, default: 512
SECONDBRAIN_OVERLAP      # overlap tokens, default: 64
SECONDBRAIN_TOP_K        # results returned per query, default: 5
```

**Phase 1 — LLM config:**
```
ANTHROPIC_API_KEY        # required for RAG and agent commands. Raise clear error if missing.
```

---

## Rust Integration (twin-core)

Rust enters the project at one specific seam: the chunking and tokenization hot path.
This is compute-bound work where Python overhead is measurable at scale.
The precedent is Hugging Face's tokenizers library — same pattern, same justification.

### Directory structure (added at Stage 0 Extension)
```
twin-core/          # Rust crate root
  Cargo.toml
  src/
    lib.rs          # PyO3 bindings
    chunker.rs      # chunking logic
    tokens.rs       # token counting
twin/
  ingestion/
    parser.py       # imports from twin_core — Rust is invisible above this line
```

### The boundary rule
Rust receives plain strings, returns plain data (text, offsets).
It does not touch LanceDB, SQLite, the filesystem, or any Python objects
beyond what PyO3 marshals automatically. Keep the boundary narrow.

### Build toolchain
- maturin compiles the Rust crate into a Python-importable .so extension
- maturin develop for local dev; maturin build --release for distribution
- Add maturin to [build-system] in pyproject.toml when the crate is introduced

### When Claude Code works on twin-core
- The Python test suite in tests/test_parser.py is the source of truth for correctness
- The Rust implementation must pass all existing Python chunker tests
- Do not change chunk semantics — only the implementation language changes
- Do not introduce Rust before the Python reference implementation is complete and tested

---

## Current Stage: Phase 1 (In Development)

Stage 0 is complete and validated. Phase 1 adds a RAG loop and a basic agent with tool access.

### Phase 1 Goals (In Order of Dependency)
1. **Close the RAG loop** — Retrieved chunks are passed to an LLM that synthesizes a grounded, attributed answer.
2. **Basic agent with tools** — A configured LLM instance that can search the knowledge base, incorporate results, and reason across multiple retrieval steps.

### Phase 1 Architecture Additions

**New Directories:**
```
twin/
  llm/                   # LLM provider abstraction
    __init__.py
    base.py              # Abstract interface
    anthropic.py         # Anthropic implementation
  rag/                   # RAG pipeline
    __init__.py
    pipeline.py          # Retrieve → format → generate → return
    context.py           # Chunk formatting and attribution
    prompts.py           # System prompt definitions
  agent/                 # Agent runtime
    __init__.py
    runtime.py           # Tool-using LLM loop
    tools.py             # Tool definitions and dispatch
    log.py               # Activity log
```

**New CLI Commands:**
- `twin rag <query>` — Synthesize an answer from retrieved context with source attribution.
- `twin agent <task>` — Invoke the agent to complete a multi-step task.

**The RAG Pipeline (Four Steps):**
1. Retrieve chunks from Stage 0 retriever
2. Format chunks as context with source attribution
3. Call LLM with system prompt (answer only from context, cite sources, admit uncertainty)
4. Return synthesized answer with source list

**The Agent Loop:**
- Receives a task description
- Decides whether and when to search the knowledge base
- Can retrieve multiple times within a single response
- Chains reasoning across multiple search results
- Terminates on final answer or iteration limit (max 5 tool calls)

**LLM Client Design:**
- Thin wrapper around Anthropic API (Phase 1 only)
- Designed as abstract interface for Phase 2 multi-provider support
- Reads `ANTHROPIC_API_KEY` from environment at initialization

### What Is Deferred from Phase 1

These items are designed into Phase 1 but implementation is deferred to Phase 2+:
- Multi-provider LLM switching (interface designed, Anthropic only in Phase 1)
- Obsidian vault watcher (generic Markdown ingestion sufficient for Phase 1)
- Streaming responses (improves UX but not load-bearing)
- Cost tracking and token accounting
- Agent write-back to knowledge base

### Next Steps (Phase 1 Build Order)

1. Build LLM client and Anthropic implementation
2. Build context formatter for RAG pipeline
3. Wire RAG pipeline (retriever + formatter + LLM)
4. Define KB search tool and dispatch logic
5. Build agent runtime and tool-using loop

Stage 2 will add: multi-provider LLM switching UI, Obsidian integration, PDF/URL ingestion, agent builder UI.
