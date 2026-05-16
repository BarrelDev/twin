# twin

**Twin** is a local-first semantic search engine for personal knowledge bases. It ingests a folder of Markdown notes, embeds them with a locally-run transformer model, and lets you query them with natural language — no API keys, no cloud, no tracking.

> Stage 0 goal: prove that semantic search over personal notes is good enough to build on.

<!-- FUTURE: add a coverage badge here once GitHub Actions CI is set up.
     Steps:
       1. Create .github/workflows/ci.yml — run `uv run pytest --cov=twin --cov-report=xml`
       2. Connect the repo to codecov.io (free for public repos) or use the
          shields.io/endpoint badge with the generated coverage.xml artifact
       3. Replace this comment with:
          [![Coverage](https://codecov.io/gh/BarrelDev/twin-repo/branch/master/graph/badge.svg)](https://codecov.io/gh/BarrelDev/twin-repo)
-->

---

## Why It Exists

Most "AI over your notes" tools are wrappers around OpenAI's API. That means your personal writing leaves your machine, you pay per query, and you stop using it the moment you lose internet. Twin is the opposite: every component runs offline, and the retrieval quality comes from deliberate data preparation choices, not from outsourcing to a hosted model.

---

## Demo

```
$ twin ingest ./notes
Scanning ./notes ... 142 files found
Chunking ... 1,203 chunks generated
Embedding ... done (47s, nomic-embed-text-v1.5)
Stored to ~/.twin

$ twin query "What did I write about the Rust ownership model?"
─────────────────────────────────────────────────────────────
Result 1  [score: 0.91]  notes/languages/rust.md › Ownership
  "Ownership in Rust is the mechanism by which the compiler
   tracks which variables are responsible for freeing memory..."
─────────────────────────────────────────────────────────────
```

*(Full CLI not yet wired — ingest and query commands are in progress)*

<!-- FUTURE: replace the demo block above with an embedded GIF once the CLI is end-to-end.
     Steps:
       1. Install asciinema: `pip install asciinema`
       2. Record: `asciinema rec demo.cast`
          Run: twin ingest ./notes  →  twin query "What did I write about X?"
       3. Convert to GIF: `agg demo.cast demo.gif` (https://github.com/asciinema/agg)
       4. Commit demo.gif to the repo (or host on GitHub releases to keep repo size small)
       5. Replace the fenced code block above with:
          ![demo](demo.gif)
          and remove this comment + the italics note below it.
-->

---

## How It Works

```
Markdown files
      │
      ▼
 parser.py          Heading-first chunking with 512-token budget,
                    64-token overlap, frontmatter extraction
      │
      ▼
 embedder.py        nomic-embed-text-v1.5 (768-dim), MTEB top-5
                    for retrieval tasks. Separate prefixes for
                    documents vs. queries as the model requires.
      │
      ▼
 vector.py          LanceDB persistent store. ANN search,
                    source_path filtering, metadata preserved.
      │
      ▼
 retriever.py       Ranks results, formats output (in progress)
      │
      ▼
 cli.py             Typer CLI — twin ingest / twin query
```

---

## Architecture Decisions

### Chunking strategy
The primary split boundary is **markdown headings**, not arbitrary token windows. A heading boundary is a semantic boundary — it captures one topic, one argument, or one reference entry. Secondary splits are paragraph breaks. This hierarchy means chunks are coherent units of thought, not arbitrary slices of text.

Chunks overlap by 64 tokens to prevent context loss at boundaries. The 512-token budget was chosen to balance retrieval precision (smaller is sharper) against context availability (larger gives the model more to work with). These are documented constants, not magic numbers.

### Embedding model selection
`nomic-ai/nomic-embed-text-v1.5` was selected after reviewing [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) benchmarks. It ranks in the top 5 for retrieval tasks among models that can run locally, at 768 dimensions (a sweet spot between accuracy and storage). It also requires task-specific prefixes (`search_document:` vs `search_query:`), which the embedder handles explicitly — this distinction matters for retrieval quality.

### Idempotent ingestion
Running `ingest` twice on an unchanged file does nothing. SHA-256 hashes of file content are stored in the metadata registry (SQLite). On ingest, hashes are compared; unchanged files are skipped, changed files replace their chunks. This is a data pipeline correctness requirement, not a performance optimization — without idempotency, repeated ingest produces duplicate results.

### No abstraction frameworks
Zero LangChain, LlamaIndex, or similar. Every component is a thin wrapper around its underlying library. This is intentional: abstractions obscure what's happening during retrieval, make debugging harder, and add dependencies that change breaking APIs frequently. The architecture is more lines of code, and more understandable.

### Rust for the hot path
Chunking and token counting are compute-bound operations. At scale, Python overhead on these tasks becomes measurable. The Rust crate (`twin-core`) handles chunking and tokenization via PyO3 bindings — the same pattern used by Hugging Face's tokenizers library. The boundary is clean: Rust receives plain strings and returns text + offsets. It does not touch the database, filesystem, or Python objects beyond what PyO3 marshals automatically. The Python test suite (`test_parser.py`) remains the source of truth for correctness.

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (application), Rust (chunking hot path) |
| Embeddings | sentence-transformers (nomic-embed-text-v1.5) |
| Vector store | LanceDB |
| Metadata store | SQLite via SQLModel |
| CLI | Typer |
| Terminal output | Rich |
| Testing | pytest |
| Dependency management | uv |
| Python-Rust bindings | PyO3 (maturin) |

---

## Project Structure

```
twin/                       # Python application root
  ingestion/
    parser.py               # chunking, heading extraction, frontmatter parsing
    embedder.py             # sentence-transformers wrapper, nomic prefix handling
  storage/
    vector.py               # LanceDB read/write, ANN search, source filtering
    metadata.py             # SQLite document registry, hash-based dedup
  query/
    retriever.py            # search orchestration, result ranking
  cli.py                    # twin ingest / twin query commands
  config.py                 # AppConfig dataclass, env-var overrides
twin-core/                  # Rust crate for chunking hot path
  Cargo.toml
  src/
    lib.rs                  # PyO3 bindings
    chunker.rs              # heading-aware chunking logic
    tokens.rs               # token counting
tests/
  test_parser.py
  test_embedder.py
  test_vector.py
  test_metadata.py
  test_retriever.py
  conftest.py
```

---

## Current Status

### Stage 0 — Complete

All retrieval-core components are implemented and tested. Rust integration for the chunking hot path is included.

| Module | Description | Tests |
|---|---|---|
| `parser.py` | Heading-aware chunking via twin-core (Rust), overlap, frontmatter extraction | `test_parser.py` |
| `embedder.py` | nomic-embed-text-v1.5, batch + query embedding | `test_embedder.py` |
| `vector.py` | LanceDB schema, write/search, source filtering, persistence | `test_vector.py` |
| `config.py` | AppConfig with env-var overrides, model enum | — |
| `metadata.py` | SQLite document registry, SHA-256 hash dedup, change detection | `test_metadata.py` |
| `retriever.py` | Search orchestration, result ranking, Rich table formatting | `test_retriever.py` |
| `cli.py` | `twin ingest` + `twin query` wired end-to-end | — |
| `twin-core/` | Rust crate for chunking and token counting (PyO3 bindings) | — |

**Test suite: 25+ tests, passing.**

The retrieval quality bar — correct chunk in top-3 for all 5 known queries against a 10-document corpus with semantically distinct fillers — passes with `nomic-embed-text-v1.5`.

### Stage 1 — In Development

**RAG loop is the priority.** It is the prerequisite for the agent runtime — an agent that cannot synthesize knowledge is not useful.

| Feature | Status | Notes |
|---|---|---|
| RAG loop (LLM answer synthesis) | In progress | Query retriever → retrieve chunks → synthesize with Claude → return answer |
| Basic agent with tools | Planned | Agent uses knowledge base as a tool to complete multi-step tasks |
| Multi-provider LLM switching | Deferred to Stage 2 | Anthropic API is the only provider in Stage 1; abstraction designed in |
| Obsidian vault watcher | Deferred to Stage 2 | Generic ingestion pipeline from Stage 0 is sufficient for now |

### Stage 2 — Planned

- Multi-provider LLM abstraction (OpenAI, Anthropic, etc.)
- Obsidian vault automatic watcher
- PDF and URL ingestion
- Agent builder UI

### Next Steps (Stage 0 Polish)

- [ ] End-to-end manual test against a real notes folder
- [ ] GitHub Actions CI: `uv run pytest --cov=twin --cov-report=xml` on push
- [ ] Connect codecov for coverage badge
- [ ] Write `DESIGN.md` for external readability (recruiter/collaborator audience)

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/BarrelDev/twin-repo
cd twin-repo
uv sync
```

### Run tests

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

> **Note:** The first test run downloads `nomic-embed-text-v1.5` (~270 MB) from Hugging Face. It is cached locally after that.

### Use the CLI (Stage 0 target)

```bash
uv run python -m twin ingest ./notes
uv run python -m twin query "What did I write about X?"
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

---

## Design Document

<!-- FUTURE: replace the CLAUDE.md link below with a DESIGN.md written for an external reader.
     CLAUDE.md is an internal instruction file for Claude Code — not appropriate to surface publicly.
     DESIGN.md should cover (adapt from CLAUDE.md, rewrite for a recruiter/collaborator audience):
       - Problem framing: why local-first, why not OpenAI wrappers
       - Chunking parameter decisions and the reasoning behind each
       - Embedding model selection: MTEB benchmarks, why nomic-embed-text-v1.5
       - Retrieval quality evaluation methodology
       - Idempotent ingest design and why it matters for data pipelines
       - Stage roadmap (0 → 1 → 2) and what changes at each stage
       - Rust extension plan (twin-core) for the chunking hot path
     Once DESIGN.md exists, update the link below.
-->

The full design rationale — chunking parameter decisions, retrieval quality evaluation methodology, Rust extension plan for the hot path, and stage roadmap — is in [`CLAUDE.md`](./CLAUDE.md).
