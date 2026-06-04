# CLAUDE.md — Twin

This file is read by Claude Code at the start of every session.
It defines architectural decisions, conventions, and constraints.
Do not override these without explicit instruction from the project author.

---

## Current Phase: 3

Phase 0 (retrieval core) — Complete
Phase 1 (RAG loop + basic agent) — Complete
Phase 2 (multi-provider, keychain, Obsidian sync, expanded ingestion) — Complete
Phase 3 (egui desktop UI, graph-aware retrieval, agent memory, distribution) — In Progress

Do not implement anything from Phase 4 unless explicitly asked.

---

## Project Overview

Twin is a local-first knowledge OS with agent execution and a native desktop interface.
Users ingest Markdown notes, PDFs, and URLs into a local vector store, then interact
via a Jarvis-style egui desktop app or the twin CLI.

The interface is ambient intelligence: chat at the center, contextual panels for KB
browsing and agent execution, dashboard widgets showing live system state.

**Full CLI surface:**
```
twin ingest <path|url>          ingest Markdown, PDF, or URL
twin query "<q>"                raw semantic search (debug)
twin rag "<q>"                  retrieve + synthesize + cite (streaming)
twin agent "<task>"             multi-step agent with KB search + vault write-back
twin watch <vault-path>         Obsidian vault watcher (background process)
twin config set-key             interactive encrypted key setup
twin config set-provider <p>    set active LLM provider
twin config set-model <m>       set default model for active provider
twin config list                show config (never reveal key values)
twin config list-models         list models for active provider
twin config remove-key <p>      remove a provider's key
twin usage                      token and cost summary
twin memory list                show long-term memory records
twin memory show <id>           show full content of a memory record
twin memory delete <id>         delete a memory record
twin memory distill             manually trigger session distillation
twin memory clear               delete all memory (with confirmation)

twin-ui                         launch the desktop interface
```

**All structured CLI output uses --json flag for machine-readable responses.**
twin-ui always passes --json and parses responses. Never break this contract.

---

## Repository Structure

```
twin-repo/
  Cargo.toml              workspace root — members: twin-core, twin-ui
  twin-core/              Rust crate: chunker + tokenizer + PyO3 bindings
    Cargo.toml
    src/
      lib.rs              PyO3 bindings
      chunker.rs          heading-aware chunking logic
      tokens.rs           word-based token counting
  twin-ui/                Rust binary: egui desktop application
    Cargo.toml
    src/
      main.rs             eframe entry point, app state init
      app.rs              TwinApp struct, egui update() loop
      panels/
        chat.rs           chat panel: message history, streaming, citations
        kb_browser.rs     document list + graph view
        agent.rs          agent task input + reasoning trace display
        settings.rs       provider/model/key config panel
      widgets/
        dashboard.rs      KB stats, usage, provider, watcher widgets
        source_chip.rs    inline citation chip component
        streaming_text.rs token-by-token text renderer
      subprocess.rs       twin CLI spawner, JSON parser, channel bridge
      graph.rs            petgraph wrapper, force layout, painter renderer
      memory.rs           session memory log, distillation trigger
  twin/                   Python package
    __init__.py
    config.py
    config_manager.py
    usage.py
    cli.py
    ingestion/
      parser.py           calls twin_core Rust extension
      embedder.py
      pdf.py
      url.py
      obsidian.py
    storage/
      vector.py
      metadata.py
    query/
      retriever.py        updated: graph-augmented retrieval in Phase 3
    llm/
      base.py
      anthropic.py
      openai.py
      gemini.py
      ollama.py
      openrouter.py
    rag/
      pipeline.py
      context.py
      prompts.py
    agent/
      runtime.py          updated: memory-augmented context in Phase 3
      tools.py
      log.rs
    memory/               new in Phase 3
      __init__.py
      session.py          in-process session event log
      store.py            SQLite long-term memory records + LanceDB embeddings
      distill.py          LLM-based session distillation
  tests/
    conftest.py
    ... (existing test files)
    test_graph.py         new in Phase 3
    test_memory_session.py    new in Phase 3
    test_memory_store.py      new in Phase 3
    test_memory_distill.py    new in Phase 3
    test_subprocess_bridge.py new in Phase 3
  pyproject.toml
  CLAUDE.md
  README.md
  setup.sh
  setup.bat
```

Do not create files outside this structure without asking first.

---

## Cargo Workspace Rules

The repo root Cargo.toml is a workspace manifest. It must always read:

```toml
[workspace]
members = ["twin-core", "twin-ui"]
resolver = "2"
```

Both crates share a single Cargo.lock. Do not create separate lockfiles.
Build both from the workspace root: `cargo build --release`
Do not run cargo commands from inside twin-core/ or twin-ui/ unless
specifically testing one crate in isolation.

---

## twin-ui Architecture

### The core rule: twin-ui never imports Python
twin-ui communicates with the Twin Python backend exclusively via subprocess.
It does not link against the Python runtime, does not import twin_core as a library,
and does not call any Python functions directly.
Every action in the UI maps to a twin CLI command.

### The --json contract (never break this)
Every CLI command that produces structured output must support a --json flag.
twin-ui always passes --json. Responses must be valid JSON on stdout.
Errors go to stderr as plain text. Exit code 0 = success, non-zero = failure.

JSON response schemas (do not change without updating twin-ui parsers):

```
twin query --json "<q>"
→ [{"text": str, "source": str, "score": float, "heading_path": [str]}]

twin rag --json "<q>"
→ {"answer": str, "sources": [{"file": str, "heading": str}], "usage": {"tokens": int, "cost": float|null}}

twin agent --json "<task>"
→ stream of newline-delimited JSON:
  {"type": "tool_call", "tool": str, "input": str}
  {"type": "tool_result", "tool": str, "output": str}
  {"type": "answer", "text": str}
  {"type": "log", "message": str}

twin ingest --json <path>
→ {"doc_id": str, "chunks_added": int, "skipped": bool}

twin usage --json
→ [{"date": str, "provider": str, "calls": int, "tokens": int, "cost": float|null}]

twin config list --json
→ {"provider": str, "model": str, "keys_present": [str]}

twin memory list --json
→ [{"id": str, "created_at": str, "type": str, "content": str}]
```

### Async subprocess pattern (always use this, never block the UI thread)
```rust
// CORRECT — background thread + channel
let (tx, rx) = std::sync::mpsc::channel::<UiEvent>();
std::thread::spawn(move || {
    let output = spawn_twin_command(&["rag", "--json", &query]);
    tx.send(UiEvent::RagResponse(output)).unwrap();
});
// Store rx in app state, poll on each frame

// WRONG — blocks the UI thread, causes freeze
let output = spawn_twin_command(&["rag", "--json", &query]);
```

### egui immediate-mode rules
- All state lives in TwinApp struct fields — never in local variables across frames
- Do not call expensive operations inside the update() loop directly
- Use background threads for all subprocess calls, file I/O, and graph computation
- Trigger repaints with ctx.request_repaint() after receiving channel messages
- Do not use .unwrap() on channel recv — use try_recv() and handle Empty gracefully

---

## UI Layout — Do Not Deviate From This

```
┌─────────────────────────────────────────────────────┐
│  TOP BAR: KB name | provider+model | tokens | ⚙     │
├──────────────┬──────────────────────┬───────────────┤
│              │                      │               │
│  LEFT PANEL  │     CHAT (center)    │  RIGHT PANEL  │
│  KB Browser  │                      │  Agent Panel  │
│  (toggle)    │  message bubbles     │  (toggle)     │
│              │  streaming text      │               │
│  doc list    │  source chips        │  task input   │
│  graph view  │                      │  tool calls   │
│              │  [input bar]         │  trace log    │
├──────────────┴──────────────────────┴───────────────┤
│  BOTTOM: last ingest | watcher status | docs | cost  │
└─────────────────────────────────────────────────────┘
```

Panels are collapsible. Chat is always visible. Dashboard widgets are always visible.
Do not add new top-level layout zones without explicit instruction.

### Visual design constants (define in app.rs, reference everywhere)
```rust
pub const BG_PRIMARY:    egui::Color32 = egui::Color32::from_rgb(10, 10, 15);
pub const BG_SECONDARY:  egui::Color32 = egui::Color32::from_rgb(18, 18, 28);
pub const ACCENT:        egui::Color32 = egui::Color32::from_rgb(61, 90, 254);
pub const TEXT_PRIMARY:  egui::Color32 = egui::Color32::from_rgb(220, 220, 235);
pub const TEXT_DIM:      egui::Color32 = egui::Color32::from_rgb(120, 120, 145);
pub const SUCCESS:       egui::Color32 = egui::Color32::from_rgb(40, 200, 100);
pub const WARNING:       egui::Color32 = egui::Color32::from_rgb(255, 180, 0);
```

Do not hardcode color values outside this block. Reference these constants everywhere.

---

## Graph-Aware Retrieval

### The scoring formula (do not change without discussion)
```python
combined_score = (GRAPH_VECTOR_WEIGHT * vector_similarity) + \
                 (GRAPH_PROXIMITY_WEIGHT * graph_proximity)
```

Defaults: GRAPH_VECTOR_WEIGHT = 0.7, GRAPH_PROXIMITY_WEIGHT = 0.3
Configurable via TWIN_GRAPH_VECTOR_WEIGHT and TWIN_GRAPH_PROXIMITY_WEIGHT env vars.

### Graph proximity values
- Direct neighbor (1 hop): 1.0
- Two-hop neighbor: 0.5
- Beyond 2 hops: 0.0 (not included)

### Graph rebuild rules
- Full rebuild: on first startup or metadata store reset only
- Incremental rebuild: on vault watcher ingest event (update only changed doc's edges)
- Never rebuild synchronously on the UI thread

### petgraph usage
- Graph type: DiGraph<DocNode, EdgeWeight>
- DocNode: { doc_id: String, title: String, source_path: String }
- EdgeWeight: f32 (link frequency, normalized 0.0-1.0)
- Store as Arc<RwLock<DiGraph<...>>> for safe cross-thread access

---

## Agent Memory

### Two-tier architecture (do not conflate the tiers)

**Session memory (twin/memory/session.py)**
- In-process Vec of SessionEvent dataclasses
- Cleared on application exit — never persisted directly
- Event types: query_asked, rag_response, agent_task, tool_result
- Maximum 50 events retained (oldest dropped when exceeded)
- Passed to agent as additional context: last 5 events by default

**Long-term memory (twin/memory/store.py)**
- SQLite table: memory_records
- LanceDB collection: memory_embeddings (for semantic search)
- Written only by the distillation process — never written directly
- Read at agent startup: top-3 records by semantic similarity to task

### Memory record schema
```python
@dataclass
class MemoryRecord:
    memory_id: str          # UUID
    created_at: str         # ISO 8601
    session_id: str         # links to originating session
    memory_type: str        # "fact" | "preference" | "task_outcome" | "context"
    content: str            # distilled text, max 512 tokens
    source_docs: list[str]  # doc_ids referenced in the session
```

### Distillation rules
- Triggered automatically on application close
- Triggered manually by twin memory distill
- Uses the active LLM provider — respect the same provider resolution order
- Distillation prompt lives in agent/prompts.py — do not hardcode it elsewhere
- If distillation fails (API error, no provider), log the error and skip silently
  — never block application close on distillation

### Context assembly order (do not change)
1. System prompt (fixed)
2. Top-3 long-term memory records (semantic search against task)
3. Last 5 session events
4. Top-5 KB retrieval chunks
5. User task/query

---

## Tech Stack — Fixed Decisions

### Python side (unchanged from Phase 2)
| Concern | Choice | Do NOT use |
|---|---|---|
| Vector store | LanceDB | ChromaDB, Pinecone, Weaviate |
| Embeddings | sentence-transformers (nomic-embed-text-v1.5) | OpenAI embeddings |
| Metadata store | SQLite via SQLModel | PostgreSQL, MongoDB |
| CLI | Typer | argparse, click, Fire |
| Terminal output | Rich | print() |
| Testing | pytest | unittest |
| Dependency mgmt | uv + pyproject.toml | pip + requirements.txt |
| Abstraction libs | None | LangChain, LlamaIndex |
| PDF extraction | pymupdf | pdfplumber, pypdf |
| Web extraction | trafilatura | BeautifulSoup |
| Encryption | cryptography (PyCA) | pycryptodome |
| Filesystem watch | watchdog | polling loops |

### Rust side
| Concern | Choice | Do NOT use |
|---|---|---|
| UI framework | egui + eframe | Tauri, Iced, GTK, Qt |
| Graph | petgraph | custom adjacency list |
| JSON | serde + serde_json | manual parsing |
| Async | tokio (twin-ui only) | async-std |
| HTTP (if needed) | reqwest | hyper directly |

---

## Code Conventions

### Rust (twin-ui and twin-core)
- Use `?` for error propagation — never .unwrap() in production paths
- .unwrap() is acceptable only for: static regex compilation, test code, truly
  unreachable paths (with a comment explaining why)
- All public functions have doc comments (///)
- Use thiserror for custom error types, not Box<dyn Error>
- Structs that cross thread boundaries must be Send + Sync — verify at compile time
- No unsafe code without explicit author approval and a safety comment

### Python (twin/)
- Type hints required on every function signature
- Dataclasses for structured data, not dicts
- No magic strings — use enums or constants
- Async for all LLM provider calls
- Docstrings on all public functions (Google style)
- No LangChain, LlamaIndex, or abstraction frameworks

### The --json contract (Python cli.py)
Every command that twin-ui calls must handle --json:
```python
@app.command()
def rag(query: str, json_output: bool = typer.Option(False, "--json")):
    result = run_rag_pipeline(query)
    if json_output:
        print(json.dumps(result.to_dict()))
    else:
        # Rich formatted output for human use
        console.print(result.answer)
```
Do not break existing --json response schemas. twin-ui parsers depend on them.

---

## Testing Requirements

### Python tests
- Every module has a corresponding test file
- Use tmp_path for all file/DB operations
- Mock all LLM provider calls — never make real API calls in tests
- Mock the embedding model in tests not specifically testing embedding quality
- Retrieval quality bar: correct chunk in top-3 for 8/10 known-answer queries
- New in Phase 3: graph retrieval must outperform or match pure vector on the
  same 10-query benchmark

```bash
uv run pytest tests/ -v --cov=twin --cov-report=term-missing
```

### Rust tests (twin-ui)
- Unit tests in #[cfg(test)] blocks within each source file
- subprocess.rs must have tests that mock the twin binary response
- graph.rs must have tests for: graph construction, proximity scoring,
  combined score calculation, force layout convergence
- Do not make real subprocess calls in Rust tests — use a mock twin binary
  or a fixture response string

```bash
cargo test --workspace
```

### Fixtures (conftest.py — must exist and be used)
- tmp_lance_db, tmp_sqlite, mock_embedder, mock_llm_provider, tmp_vault
- New in Phase 3: tmp_graph (pre-built petgraph DiGraph with 10 nodes/edges)
- New in Phase 3: mock_session_memory (SessionMemory with 5 test events)

---

## What Claude Code Should Always Do

### General
- Write type hints / doc comments on every public function (Python and Rust)
- Write or update the corresponding test when implementing anything
- Check if the function already exists before implementing — ask if unclear
- Note any new dependencies explicitly before adding them
- Run relevant tests after implementing to verify

### Rust-specific
- Use ? for error propagation
- Run cargo clippy before considering a Rust implementation done
- Keep egui rendering code inside the update() loop only
- Spawn background threads for anything that takes >1ms

### Python-specific
- Use Rich for all terminal output
- Use pathlib.Path for all file paths
- Use async/await for all LLM provider calls
- Always pass --json when calling twin from within twin-ui subprocess bridge

## What Claude Code Should Never Do

### General
- Break the --json CLI contract
- Make real API calls in tests
- Write to real filesystem paths in tests
- Add new top-level layout zones to the UI without instruction
- Implement Phase 4 features

### Rust-specific
- Call Python from twin-ui (no PyO3 in twin-ui — that is twin-core only)
- Block the egui update() thread with I/O or subprocess calls
- Use .unwrap() in non-test, non-unreachable production code
- Hardcode color values — reference the constants in app.rs
- Use unsafe without explicit approval and a safety comment

### Python-specific
- Import from LangChain, LlamaIndex, or abstraction frameworks
- Log, print, or return raw API key values
- Write agent output outside <vault>/Agents/
- Break idempotent ingest behavior
- Skip type hints for brevity

---

## Config Environment Variables

```
TWIN_DATA_DIR               ~/.twin (LanceDB, SQLite, keychain)
TWIN_EMBED_MODEL            nomic-ai/nomic-embed-text-v1.5
TWIN_CHUNK_TOKENS           512
TWIN_OVERLAP                64
TWIN_TOP_K                  5
TWIN_PROVIDER               anthropic
TWIN_OLLAMA_URL             http://localhost:11434
TWIN_GRAPH_VECTOR_WEIGHT    0.7
TWIN_GRAPH_PROXIMITY_WEIGHT 0.3

# API keys (fallback if not in keychain)
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
OPENROUTER_API_KEY
```

---

## Rust Extension (twin_core)

### The boundary rule
twin_core is the PyO3 bridge. It receives plain strings, returns plain data.
It does NOT appear in twin-ui's dependencies. twin-ui never imports twin_core.
twin_core is only used by the Python package (twin/).

### Fallback import in parser.py
```python
try:
    from twin_core import Chunk, chunk_text as _rust_chunk_text
    _USE_RUST = True
except ImportError:
    _USE_RUST = False
```
Never remove the Python fallback.

### Build
```bash
# From workspace root:
cd twin-core && uv run maturin develop   # dev
cd twin-core && uv run maturin build --release  # production
```

---

## Phase 4 Preview (do not implement)

- Persistent chat history across sessions
- Plugin API for third-party agent tools
- Multi-location sync (pluggable backends: S3, self-hosted, hosted tier)
- Mobile companion app
- Voice input via Whisper
- Tag-based query filtering