# twin

**Twin** is a local-first knowledge OS with semantic search, RAG, and agent execution. It ingests Markdown notes, PDFs, and URLs into a local vector store and lets you query them with natural language or run multi-step agents that reason across your knowledge base — supporting five LLM providers with an encrypted local keychain.

---

## What It Does

```
# Ingest anything — Markdown, PDF, or URL
$ twin ingest ./notes
Done. Ingested 47 files (312 chunks). Skipped 0 unchanged.

$ twin ingest research.pdf
Done. Ingested research.pdf (18 chunks).

$ twin ingest https://example.com/article
Done. Ingested URL (6 chunks).

# Semantic search — no API key needed
$ twin query "What did I write about the Rust ownership model?"
┌───┬───────┬───────────────────────┬──────────────────────────────────┐
│ # │ Score │ Source                │ Text                             │
├───┼───────┼───────────────────────┼──────────────────────────────────┤
│ 1 │ 0.91  │ rust.md › Ownership  │ "Ownership in Rust is the..."    │
└───┴───────┴───────────────────────┴──────────────────────────────────┘

# RAG: streamed answer grounded in your notes
$ twin rag "What is the Rust ownership model?"
Rust's ownership model gives each value a single owner. When the owner
goes out of scope, the value is dropped automatically...

Sources:
  • rust.md  Ownership
  • systems.md  Memory Safety

1 call · 1,240 tokens · ~$0.003

# Multi-step agent with knowledge base search and vault write-back
$ twin agent "Summarize everything I know about async Rust"
  iter 0 → search_knowledge_base  [source: rust.md > Async] The async keyword...
  iter 1 → search_knowledge_base  [source: tokio.md > Runtime] Tokio provides...

Based on your notes, async Rust centers on the Future trait...

Tool calls made: 2
2 calls · 890 tokens · ~$0.005

# Manage API keys — stored encrypted, never printed
$ twin config set-key
Providers: anthropic, openai, gemini, openrouter
Provider: anthropic
API key for anthropic: ****

$ twin config set-provider openai
✓ Active provider set to openai.

$ twin usage
┌────────────┬───────────┬───────┬──────────────┬───────────────────┬──────────┐
│ Date       │ Provider  │ Calls │ Prompt tokens │ Completion tokens │ Est. cost│
├────────────┼───────────┼───────┼──────────────┼───────────────────┼──────────┤
│ 2026-06-01 │ anthropic │ 14    │ 18,430        │ 3,210             │ $0.0421  │
└────────────┴───────────┴───────┴──────────────┴───────────────────┴──────────┘

# Watch an Obsidian vault for changes and re-ingest on save
$ twin watch ~/vault
Watching ~/vault for .md changes. Log: ~/.twin/watcher.log  (Ctrl-C to stop)
```

---

## Architecture

```
Markdown / PDF / URL
        │
        ▼
 ingestion/             Format routing:
   parser.py            • .md/.txt → Obsidian-aware chunker (wikilinks, tags, frontmatter)
   pdf.py               • .pdf     → pymupdf page extractor
   url.py               • URL      → trafilatura web extractor
   obsidian.py          All formats produce the same _Chunk shape.
        │
        ▼
 embedder.py            nomic-embed-text-v1.5 (768-dim). Applies task prefixes:
                        search_document: for ingestion, search_query: for queries.
        │
        ▼
 vector.py              LanceDB persistent store. ANN search. Stores link_targets
 metadata.py            and tags for Obsidian notes. SHA-256 hash registry (SQLite)
                        for idempotent ingestion.
        │
        ▼
 retriever.py           Search orchestration, ranking, Rich-formatted output.
        │
        ▼
 rag/pipeline.py        Retrieve → format context with source attribution →
                        stream LLM synthesis → grounded answer + sources.
        │
        ▼
 agent/runtime.py       Multi-step tool-using loop. LLM decides when to search
 agent/tools.py         the KB or write a note to the vault. Streams final answer.
                        All tool calls logged. Usage tracked per session.
        │
        ▼
 llm/                   Five provider implementations behind one async interface:
   anthropic.py         Claude (default)
   openai.py            GPT-4o and variants
   gemini.py            Gemini 2.0 Flash and variants
   ollama.py            Local models — no API key, no cost
   openrouter.py        Unified access to 100+ models

 config_manager.py      AES-256-GCM encrypted keychain (~/.twin/keychain.enc).
                        Key derived from username + machine ID via PBKDF2 — non-portable.

 usage.py               JSONL token and cost log (~/.twin/usage.jsonl).
                        Session summaries printed at end of each rag/agent call.
```

---

## Design Decisions

### No abstraction frameworks

Zero LangChain, LlamaIndex, or similar. Every component is a thin wrapper around its underlying library — LanceDB, sentence-transformers, SQLite, provider SDKs.

**Why:** Abstractions obscure what's happening during retrieval, make debugging harder, and add dependencies with frequent breaking changes. Every retrieval failure in Twin is traceable: query → embedding → ANN search → ranking → formatting. No framework magic in the path.

---

### Embedding model selection (evidence-based)

**Choice:** `nomic-ai/nomic-embed-text-v1.5` (768 dimensions)

Benchmarked against the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Ranks top 5 for retrieval tasks among locally-runnable models. The model requires task-specific prefixes — `search_document:` for ingestion, `search_query:` for queries — which Twin applies explicitly because the distinction is measurable in retrieval quality.

---

### Provider-agnostic LLM interface

`llm/base.py` defines an abstract `LLMProvider` with four methods: `complete()`, `stream()`, `estimate_cost()`, and `list_models()`. All are async. Five concrete implementations ship with Phase 2: Anthropic, OpenAI, Gemini, Ollama, and OpenRouter.

Provider resolution order: `--provider` flag → `config.json` → `TWIN_PROVIDER` env var → Anthropic.

The runtime always appends messages in Anthropic content-block format; non-Anthropic providers convert internally in their `complete()` method. The agent runtime and RAG pipeline have zero provider-specific code.

---

### Encrypted machine-bound keychain

API keys are stored encrypted in `~/.twin/keychain.enc` using AES-256-GCM. The encryption key is derived from `username:machine_id` via PBKDF2-SHA256 (480,000 iterations) — intentionally non-portable. Keys are never printed, logged, or returned anywhere in the codebase.

Resolution order: keychain → environment variable → descriptive error with onboarding instructions.

---

### Idempotent ingestion

Running `ingest` twice on unchanged content produces no changes. SHA-256 hashes of file content (or URL content) are stored in the SQLite registry. On re-ingest: hash match → skip; hash changed → delete old chunks, insert new ones.

---

### Obsidian-native parsing

All `.md` files go through the Obsidian-aware parser. Non-vault Markdown simply yields empty `link_targets` and `tags`. The parser:
- Extracts `[[Note Name]]` and `[[Note Name|Alias]]` → `link_targets` (note names, deduplicated)
- Extracts `#tags` and `#nested/child` from the body (not YAML frontmatter)
- Strips `![[embed.png]]` from chunk text
- Converts wikilinks to plain text for embedding
- Preserves full YAML frontmatter as structured metadata in SQLite

---

### Hard vault boundary

`write_vault_note` enforces that agent output never escapes `<vault>/Agents/`. The path is sanitized (slashes and control characters replaced), then a `relative_to` boundary check is applied as defense-in-depth. The constraint is at the path level, not a convention.

---

### Rust for the chunking hot path

Chunking and token counting live in `twin_core/` — a Rust crate exposed via PyO3 bindings. The boundary is clean: Rust receives plain strings, returns text + offsets. It does not touch LanceDB, SQLite, the filesystem, or any Python objects. Same pattern as Hugging Face tokenizers.

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

### Phase 0 — Complete ✓

| Module | Description |
|---|---|
| `ingestion/parser.py` | Heading-aware chunking, overlap, frontmatter extraction |
| `ingestion/embedder.py` | nomic-embed-text-v1.5, document + query prefix handling |
| `storage/vector.py` | LanceDB schema, ANN search, source filtering, persistence |
| `storage/metadata.py` | SQLite document registry, SHA-256 hash dedup |
| `query/retriever.py` | Search orchestration, result ranking, Rich output |
| `cli.py` | `twin ingest` + `twin query` end-to-end |
| `twin_core/` | Rust crate for chunking + token counting (PyO3) |

### Phase 1 — Complete ✓

| Component | File |
|---|---|
| LLM provider abstraction | `llm/base.py`, `llm/anthropic.py` |
| Context formatter | `rag/context.py` |
| RAG pipeline | `rag/pipeline.py` |
| KB search tool + dispatch | `agent/tools.py` |
| Agent runtime | `agent/runtime.py` |
| Activity log | `agent/log.py` |
| CLI: `twin rag`, `twin agent` | `cli.py` |

### Phase 2 — Complete ✓

| Component | File |
|---|---|
| Encrypted keychain + config manager | `config_manager.py` |
| Config CLI (`set-key`, `set-provider`, `set-model`, `list`, `list-models`, `remove-key`) | `cli.py` |
| OpenAI, Gemini, Ollama, OpenRouter providers | `llm/openai.py`, `llm/gemini.py`, `llm/ollama.py`, `llm/openrouter.py` |
| Streaming (`twin rag`, `twin agent` final answer) | `rag/pipeline.py`, `agent/runtime.py` |
| PDF ingestion | `ingestion/pdf.py` |
| URL ingestion | `ingestion/url.py` |
| Obsidian parser (wikilinks, tags, frontmatter) | `ingestion/obsidian.py` |
| Vault writer (`write_vault_note` agent tool) | `agent/tools.py` |
| Vault watcher (`twin watch`) | `ingestion/obsidian.py`, `cli.py` |
| Token/cost tracking + `twin usage` | `usage.py`, `cli.py` |

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+, Rust (chunking hot path) |
| Embeddings | sentence-transformers (nomic-embed-text-v1.5) |
| Vector store | LanceDB |
| Metadata store | SQLite via SQLModel |
| LLM providers | Anthropic, OpenAI, Google Gemini, Ollama, OpenRouter |
| CLI | Typer |
| Terminal output | Rich |
| Encryption | PyCA cryptography (AES-256-GCM + PBKDF2) |
| PDF extraction | pymupdf |
| Web extraction | trafilatura |
| Filesystem watch | watchdog |
| HTTP client | httpx |
| Testing | pytest |
| Dependency management | uv |
| Python-Rust bindings | PyO3 / maturin |

---

## Project Structure

```
twin/
  config.py               Provider enum, ModelInfo, AppConfig (TWIN_* env vars)
  config_manager.py       AES-256-GCM keychain + config.json read/write
  usage.py                UsageRecord, UsageLogger, format_session_summary
  cli.py                  Typer CLI — all commands
  ingestion/
    parser.py             Markdown chunking (Rust extension)
    embedder.py           sentence-transformers wrapper, prefix handling
    pdf.py                pymupdf-based PDF parser
    url.py                trafilatura-based URL ingester
    obsidian.py           Wikilink/tag/frontmatter parser + VaultWatcher
  storage/
    vector.py             LanceDB schema, ANN search, link_targets/tags fields
    metadata.py           SQLite document registry, frontmatter_json field
  query/
    retriever.py          Search orchestration, ranking, Rich output
  llm/
    base.py               LLMProvider ABC, ToolDefinition, ToolCall, LLMResponse
    anthropic.py          Async Claude
    openai.py             Async OpenAI
    gemini.py             Google Gemini via google-genai SDK
    ollama.py             Local Ollama via httpx
    openrouter.py         OpenRouter (unified multi-provider access)
  rag/
    pipeline.py           query() + query_stream(), session usage tracking
    context.py            Chunk formatting with source attribution
    prompts.py            System prompt definitions
  agent/
    runtime.py            execute() + execute_stream(), session usage tracking
    tools.py              search_knowledge_base + VaultWriter + ToolDispatcher
    log.py                AgentLog: chronological event log, JSON-serializable
twin_core/
  Cargo.toml
  src/
    lib.rs                PyO3 bindings
    chunker.rs            Heading-aware chunking logic
    tokens.rs             Token counting (word-based)
```

---

## Build and Run

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Rust toolchain (`rustup`) — required to build `twin_core`
- An API key for at least one LLM provider, **or** [Ollama](https://ollama.com) for fully local operation

### Install

```bash
git clone https://github.com/BarrelDev/twin
cd twin

uv sync

cd twin_core
pip install maturin
maturin develop
cd ..
```

> Alternatively, run `setup.bat` (Windows) or `setup.sh` (Unix-based) to do all of the above in one step.

> The first run downloads `nomic-embed-text-v1.5` (~270 MB) from Hugging Face. It is cached after that.

### Set up an API key

```bash
twin config set-key
# Follow the prompt — key is stored encrypted, never echoed

# Or, use an environment variable as a fallback
export ANTHROPIC_API_KEY="sk-ant-..."
```

To use a local model with no API key:

```bash
twin config set-provider ollama
twin config set-model llama3.2
```

### Run

```bash
twin ingest ./notes                                     # index Markdown
twin ingest research.pdf                                # index PDF
twin ingest https://example.com/article                 # index URL

twin query "What did I write about Rust lifetimes?"    # semantic search
twin rag "What is the Rust ownership model?"           # RAG answer
twin agent "Summarize my notes on async Rust"          # agent

twin config list                                       # show active provider and keys
twin config set-provider gemini                        # switch provider
twin usage                                             # show token/cost history

twin watch ~/my-vault                                  # watch Obsidian vault
```

### Run tests

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

---

## Configuration

Settings are read from `TWIN_*` environment variables. `SECONDBRAIN_*` names are supported as a deprecated fallback.

| Variable | Default | Description |
|---|---|---|
| `TWIN_DATA_DIR` | `~/.twin` | Where LanceDB, SQLite, keychain, and usage log are stored |
| `TWIN_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `TWIN_CHUNK_TOKENS` | `512` | Max tokens per chunk |
| `TWIN_OVERLAP` | `64` | Overlap tokens between chunks |
| `TWIN_TOP_K` | `5` | Results returned per query |
| `TWIN_PROVIDER` | `anthropic` | Active LLM provider (fallback if not set in config.json) |
| `TWIN_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `ANTHROPIC_API_KEY` | — | Anthropic key fallback (keychain takes priority) |
| `OPENAI_API_KEY` | — | OpenAI key fallback |
| `GEMINI_API_KEY` | — | Gemini key fallback |
| `OPENROUTER_API_KEY` | — | OpenRouter key fallback |

---

## Engineering Highlights

**Systems thinking:** Idempotent data pipelines (SHA-256 hash dedup), compute hot path optimization (Rust via PyO3), provider abstraction designed for multi-vendor extensibility, and a machine-bound encrypted keychain. Phase 2 added four LLM providers, three ingestion formats, vault watching, and streaming without touching the retrieval core.

**Evidence-based decisions:** Embedding model selection backed by MTEB benchmark rankings. Chunking parameters documented as constants with justification. Provider pricing tables maintained per-model for accurate cost attribution.

**Composition over frameworks:** Every layer is independently testable. The retriever, pipeline, and agent runtime can each be instantiated with mock dependencies. No global state, no framework side-channels.

**Security by design:** API keys are never logged, printed, or returned anywhere in the codebase. The vault boundary enforced at path level (not convention). Path traversal sanitization with defense-in-depth boundary check. All key material machine-bound and non-portable.

**Full-stack capability:** Python application layer, Rust extension for compute-bound work, SQL data pipeline, five LLM API integrations, streaming I/O, filesystem watching, encrypted local storage, CLI tooling.

**Type discipline:** 100% type-annotated function signatures across the codebase. Types serve as documentation and catch bugs before runtime.
