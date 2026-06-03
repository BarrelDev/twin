import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from twin.config import AppConfig, Provider
from twin.config_manager import ConfigManager
from twin.usage import UsageLogger, format_session_summary

app = typer.Typer(name="twin", help="Local-first semantic search for personal notes")
config_app = typer.Typer(help="Manage Twin configuration and API keys")
app.add_typer(config_app, name="config")

console = Console()


def _build_provider(cm: ConfigManager, provider_override: str | None = None):
    """
    Instantiate the active LLM provider from config.

    Resolution order: --provider flag → config.json → TWIN_PROVIDER env → Anthropic.

    Args:
        cm: ConfigManager instance for key and provider resolution.
        provider_override: Optional provider name from a CLI --provider flag.

    Returns:
        Instantiated LLMProvider ready to use.

    Raises:
        typer.Exit: On missing key or unsupported provider (prints user-facing error).
    """
    if provider_override:
        try:
            provider = Provider(provider_override.lower())
        except ValueError:
            valid = ", ".join(p.value for p in Provider)
            console.print(f"[red]Unknown provider: {provider_override}. Valid: {valid}[/red]")
            raise typer.Exit(1)
    else:
        provider = cm.get_active_provider()

    try:
        api_key = cm.resolve_api_key(provider)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    model = cm.get_model(provider)

    try:
        match provider:
            case Provider.ANTHROPIC:
                from twin.llm.anthropic import Claude
                return Claude(api_key=api_key, model=model)
            case Provider.OPENAI:
                from twin.llm.openai import OpenAIProvider
                return OpenAIProvider(api_key=api_key, model=model)
            case Provider.GEMINI:
                from twin.llm.gemini import GeminiProvider
                return GeminiProvider(api_key=api_key, model=model)
            case Provider.OLLAMA:
                from twin.llm.ollama import OllamaProvider
                return OllamaProvider(model=model)
            case Provider.OPENROUTER:
                if not model:
                    console.print(
                        "[red]OpenRouter requires a model. "
                        "Run: twin config set-model <provider/model>[/red]"
                    )
                    raise typer.Exit(1)
                from twin.llm.openrouter import OpenRouterProvider
                return OpenRouterProvider(api_key=api_key, model=model)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


# ── Core commands ────────────────────────────────────────────────────────────

def _ingest_url_content(url: str, config: AppConfig) -> None:
    """
    Fetch, chunk, embed, and store a URL in the knowledge base.

    Args:
        url: HTTP or HTTPS URL to ingest.
        config: Runtime AppConfig.
    """
    from twin.ingestion.url import ingest_url
    from twin.ingestion.embedder import build_embedder
    from twin.storage.vector import VectorStore
    from twin.storage.metadata import DocRecord, MetadataStore

    store = VectorStore(config.data_dir / "lancedb")
    meta = MetadataStore(config.data_dir / "meta.db")
    embedder = build_embedder(config)

    try:
        chunks, content_hash = ingest_url(url, config)
    except (ImportError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if meta.get_hash(url) == content_hash:
        console.print("[yellow]URL content unchanged, skipping.[/yellow]")
        return

    if not chunks:
        console.print("[yellow]No extractable content found at URL.[/yellow]")
        return

    embeddings = embedder.embed_batch([c.text for c in chunks])
    store.write_chunks(chunks, embeddings)
    meta.upsert_doc(DocRecord(
        doc_id=chunks[0].doc_id,
        source_path=url,
        file_hash=content_hash,
        ingest_timestamp=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
        embedding_model=config.embed_model.value,
    ))
    console.print(
        f"[green]Done.[/green] Ingested URL ([bold]{len(chunks)}[/bold] chunks)."
    )


def _ingest_pdf_file(path: Path, config: AppConfig) -> None:
    """
    Parse, embed, and store a PDF file in the knowledge base.

    Args:
        path: Path to the PDF file.
        config: Runtime AppConfig.
    """
    from twin.ingestion.pdf import parse_pdf
    from twin.ingestion.embedder import build_embedder
    from twin.storage.vector import VectorStore
    from twin.storage.metadata import DocRecord, MetadataStore

    if not path.exists():
        console.print(f"[red]Error:[/red] {path} does not exist")
        raise typer.Exit(1)

    store = VectorStore(config.data_dir / "lancedb")
    meta = MetadataStore(config.data_dir / "meta.db")
    embedder = build_embedder(config)

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get_hash(str(path)) == file_hash:
        console.print("[yellow]PDF unchanged, skipping.[/yellow]")
        return

    try:
        chunks = parse_pdf(path, config)
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if not chunks:
        console.print("[yellow]No extractable content in PDF.[/yellow]")
        return

    embeddings = embedder.embed_batch([c.text for c in chunks])
    store.write_chunks(chunks, embeddings)
    meta.upsert_doc(DocRecord(
        doc_id=chunks[0].doc_id,
        source_path=str(path),
        file_hash=file_hash,
        ingest_timestamp=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
        embedding_model=config.embed_model.value,
    ))
    console.print(
        f"[green]Done.[/green] Ingested [bold]{path.name}[/bold] "
        f"([bold]{len(chunks)}[/bold] chunks)."
    )


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to notes folder, file (.md/.pdf), or URL"),
    type_: str | None = typer.Option(None, "--type", help="Force format: url, pdf, or md"),
) -> None:
    """Ingest Markdown files, a PDF, or a URL into the knowledge base."""
    from rich.progress import track

    config = AppConfig.from_env()

    is_url = path.startswith(("http://", "https://")) or type_ == "url"
    is_pdf = not is_url and (path.endswith(".pdf") or type_ == "pdf")

    if is_url:
        _ingest_url_content(path, config)
        return

    if is_pdf:
        _ingest_pdf_file(Path(path), config)
        return

    # Existing Markdown directory path
    notes_dir = Path(path)

    if not notes_dir.exists():
        console.print(f"[red]Error:[/red] {path} does not exist")
        raise typer.Exit(1)

    md_files = list(notes_dir.rglob("*.md"))
    if not md_files:
        console.print(f"[yellow]No .md files found in {path}[/yellow]")
        return

    console.print(
        f"[bold blue]twin[/bold blue] Scanning {path} ... "
        f"[bold]{len(md_files)}[/bold] files found"
    )

    from twin.ingestion.embedder import build_embedder
    from twin.storage.vector import VectorStore
    from twin.storage.metadata import DocRecord, MetadataStore
    from twin.ingestion.obsidian import parse_obsidian_file, _frontmatter_to_json

    store = VectorStore(config.data_dir / "lancedb")
    meta = MetadataStore(config.data_dir / "meta.db")
    embedder = build_embedder(config)

    skipped = ingested = total_chunks = 0

    for md_file in track(md_files, description="Ingesting..."):
        file_hash = hashlib.sha256(md_file.read_bytes()).hexdigest()
        if meta.get_hash(str(md_file)) == file_hash:
            skipped += 1
            continue

        chunks, obsidian_meta = parse_obsidian_file(md_file, config)
        if not chunks:
            continue

        embeddings = embedder.embed_batch([c.text for c in chunks])
        store.write_chunks(chunks, embeddings)

        meta.upsert_doc(DocRecord(
            doc_id=chunks[0].doc_id,
            source_path=str(md_file),
            file_hash=file_hash,
            ingest_timestamp=datetime.now(timezone.utc).isoformat(),
            chunk_count=len(chunks),
            embedding_model=config.embed_model.value,
            frontmatter_json=_frontmatter_to_json(obsidian_meta["frontmatter"]),
        ))

        ingested += 1
        total_chunks += len(chunks)

    console.print(
        f"[green]Done.[/green] "
        f"Ingested [bold]{ingested}[/bold] files "
        f"([bold]{total_chunks}[/bold] chunks). "
        f"Skipped [bold]{skipped}[/bold] unchanged."
    )


@app.command()
def query(q: str = typer.Argument(..., help="Natural-language query")) -> None:
    """Query the knowledge base with natural language."""
    from twin.ingestion.embedder import build_embedder
    from twin.storage.vector import VectorStore
    from twin.query.retriever import Retriever

    config = AppConfig.from_env()
    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)
    results = retriever.query(q, k=config.top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(retriever.format_results(results))


@app.command()
def rag(
    q: str = typer.Argument(..., help="Natural-language query for RAG synthesis"),
    k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="LLM provider override"),
) -> None:
    """Synthesize an answer from retrieved context using an LLM."""
    import asyncio
    from twin.ingestion.embedder import build_embedder
    from twin.storage.vector import VectorStore
    from twin.query.retriever import Retriever
    from twin.rag.pipeline import RAGPipeline

    config = AppConfig.from_env()
    cm = ConfigManager()
    llm = _build_provider(cm, provider)

    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)
    pipeline = RAGPipeline(retriever, llm)

    async def _stream() -> tuple[str, list[dict]]:
        stream_gen, sources = await pipeline.query_stream(q, k=k)
        full_text = ""
        with Live("", console=console, refresh_per_second=20) as live:
            async for token in stream_gen:
                full_text += token
                live.update(full_text)
        return full_text, sources

    full_text, sources = asyncio.run(_stream())

    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for src in sources:
            path = Path(src["path"]).name
            heading = " > ".join(src["heading_path"]) if src["heading_path"] else ""
            line = Text(f"  • {path}")
            if heading:
                line.append(f"  {heading}", style="dim")
            console.print(line)

    summary = format_session_summary(pipeline.session_records)
    if summary:
        console.print(f"\n[dim]{summary}[/dim]")


@app.command()
def agent(
    task: str = typer.Argument(..., help="Task description for the agent"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full activity log"),
    max_iterations: int = typer.Option(5, "--max-iter", help="Max tool calls before forced termination"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="LLM provider override"),
) -> None:
    """Invoke the agent to complete a multi-step task over the knowledge base."""
    import asyncio
    from twin.ingestion.embedder import build_embedder
    from twin.storage.vector import VectorStore
    from twin.query.retriever import Retriever
    from twin.agent.runtime import AgentRuntime
    from twin.agent.tools import ToolDispatcher

    config = AppConfig.from_env()
    cm = ConfigManager()
    llm = _build_provider(cm, provider)

    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)

    vault_writer = None
    vault_path = cm.get_vault_path()
    if vault_path:
        from twin.agent.tools import VaultWriter
        vault_writer = VaultWriter(vault_path)

    dispatcher = ToolDispatcher(retriever, vault_writer=vault_writer)
    runtime = AgentRuntime(llm, dispatcher, max_iterations=max_iterations)

    async def _run() -> tuple[str, int, list[dict]]:
        token_parts: list[str] = []
        tool_calls_made = 0
        activity_log: list[dict] = []

        async for event in runtime.execute_stream(task):
            if event["type"] == "tool_call":
                snippet = str(event.get("result", ""))[:80]
                console.print(
                    f"  [cyan]iter {event['iteration']}[/cyan] "
                    f"[yellow]→[/yellow] {event['name']}  "
                    f"[dim]{snippet}…[/dim]"
                )
            elif event["type"] == "token":
                token_parts.append(event["text"])
            elif event["type"] == "done":
                tool_calls_made = event["tool_calls"]
                activity_log = event["activity_log"]

        full_text = ""
        with Live("", console=console, refresh_per_second=20) as live:
            for part in token_parts:
                full_text += part
                live.update(full_text)

        return full_text, tool_calls_made, activity_log

    full_text, tool_calls_made, activity_log = asyncio.run(_run())

    console.print(f"\n[dim]Tool calls made: {tool_calls_made}[/dim]")

    summary = format_session_summary(runtime.session_records)
    if summary:
        console.print(f"[dim]{summary}[/dim]")

    if verbose and activity_log:
        console.print("\n[bold]Activity Log:[/bold]")
        for entry in activity_log:
            event_type = entry.get("event_type", "")
            iteration = entry.get("iteration", 0)
            details = entry.get("details", {})
            if event_type == "tool_call":
                console.print(
                    f"  [cyan]iter {iteration}[/cyan] [yellow]tool_call[/yellow] "
                    f"{details.get('tool_name', '')}({details.get('tool_input', '')})"
                )
            elif event_type == "tool_result":
                snippet = str(details.get("result", ""))[:120]
                console.print(f"  [cyan]iter {iteration}[/cyan] [dim]result[/dim] {snippet}...")
            elif event_type == "final_answer":
                reason = details.get("termination_reason", "final_answer")
                console.print(f"  [cyan]iter {iteration}[/cyan] [green]done[/green] ({reason})")


@app.command()
def usage() -> None:
    """Show token and cost summary by provider and day."""
    from collections import defaultdict

    config = AppConfig.from_env()
    log = UsageLogger(config.data_dir)
    records = log.read_all()

    if not records:
        console.print("[yellow]No usage data found.[/yellow]")
        return

    # Group by (date, provider)
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0, "local": False}
    )
    for r in records:
        key = (r.timestamp[:10], r.provider)
        g = groups[key]
        g["calls"] += 1
        g["prompt"] += r.prompt_tokens
        g["completion"] += r.completion_tokens
        if r.estimated_cost_usd is not None:
            g["cost"] += r.estimated_cost_usd
        else:
            g["local"] = True

    table = Table(show_header=True, header_style="bold")
    table.add_column("Date")
    table.add_column("Provider")
    table.add_column("Calls", justify="right")
    table.add_column("Prompt tokens", justify="right")
    table.add_column("Completion tokens", justify="right")
    table.add_column("Est. cost", justify="right")

    for (date, provider), g in sorted(groups.items()):
        cost_str = "local" if g["local"] else f"${g['cost']:.4f}"
        table.add_row(
            date,
            provider,
            str(g["calls"]),
            f"{g['prompt']:,}",
            f"{g['completion']:,}",
            cost_str,
        )

    console.print(table)


@app.command()
def watch(
    vault_path: str | None = typer.Argument(None, help="Path to Obsidian vault root"),
    status: bool = typer.Option(False, "--status", help="Show watcher status and exit"),
) -> None:
    """Watch an Obsidian vault for changes and re-ingest modified .md files."""
    import os
    import time

    config = AppConfig.from_env()
    pid_file = config.data_dir / "watcher.pid"
    log_path = config.data_dir / "watcher.log"

    if status:
        if not pid_file.exists():
            console.print("[yellow]Watcher is not running.[/yellow]")
            return
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            console.print(f"[green]Watcher running.[/green] PID: {pid}")
        except (ValueError, OSError):
            console.print("[yellow]Watcher process not found (stale PID file).[/yellow]")
        return

    if not vault_path:
        console.print(
            "[red]Error:[/red] vault path required. Usage: twin watch <vault-path>"
        )
        raise typer.Exit(1)

    vault = Path(vault_path)
    if not vault.exists():
        console.print(f"[red]Error:[/red] {vault_path} does not exist")
        raise typer.Exit(1)

    try:
        from watchdog.observers import Observer
        from twin.ingestion.obsidian import VaultWatcher
    except ImportError as e:
        console.print("[red]Error:[/red] watchdog not installed. Run: uv add watchdog")
        raise typer.Exit(1)

    config.data_dir.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    handler = VaultWatcher(vault, config, log_path=log_path)
    observer = Observer()
    observer.schedule(handler, str(vault), recursive=True)
    observer.start()

    console.print(
        f"[green]Watching[/green] {vault_path} for .md changes. "
        f"Log: {log_path}  (Ctrl-C to stop)"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        pid_file.unlink(missing_ok=True)
        console.print("\n[yellow]Watcher stopped.[/yellow]")


# ── Config subcommands ───────────────────────────────────────────────────────

@config_app.command("set-key")
def config_set_key() -> None:
    """Interactively set an API key for a provider (key is never echoed)."""
    import getpass

    providers = [p.value for p in Provider if p != Provider.OLLAMA]
    console.print("[bold]Providers:[/bold] " + ", ".join(providers))

    provider_str = typer.prompt("Provider")
    try:
        provider = Provider(provider_str.lower())
    except ValueError:
        console.print(f"[red]Unknown provider: {provider_str}[/red]")
        raise typer.Exit(1)

    if provider == Provider.OLLAMA:
        console.print("[yellow]Ollama runs locally and requires no API key.[/yellow]")
        raise typer.Exit(0)

    api_key = getpass.getpass(f"API key for {provider.value}: ")
    if not api_key.strip():
        console.print("[red]Empty key not accepted.[/red]")
        raise typer.Exit(1)

    cm = ConfigManager()
    cm.set_key(provider, api_key.strip())
    console.print(f"[green]✓[/green] Key stored for [bold]{provider.value}[/bold].")


@config_app.command("remove-key")
def config_remove_key(
    provider: str = typer.Argument(..., help="Provider name (e.g. openai)"),
) -> None:
    """Remove a provider's API key from the keychain."""
    try:
        p = Provider(provider.lower())
    except ValueError:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        raise typer.Exit(1)

    cm = ConfigManager()
    cm.remove_key(p)
    console.print(f"[green]✓[/green] Key removed for [bold]{p.value}[/bold].")


@config_app.command("set-provider")
def config_set_provider(
    provider: str = typer.Argument(..., help="Provider to activate (e.g. openai)"),
) -> None:
    """Set the active LLM provider."""
    try:
        p = Provider(provider.lower())
    except ValueError:
        valid = ", ".join(v.value for v in Provider)
        console.print(f"[red]Unknown provider: {provider}. Valid: {valid}[/red]")
        raise typer.Exit(1)

    cm = ConfigManager()
    cm.set_active_provider(p)
    console.print(f"[green]✓[/green] Active provider set to [bold]{p.value}[/bold].")


@config_app.command("set-model")
def config_set_model(
    model: str = typer.Argument(..., help="Model identifier to use as default"),
) -> None:
    """Set the default model for the active provider."""
    cm = ConfigManager()
    provider = cm.get_active_provider()
    cm.set_model(provider, model)
    console.print(f"[green]✓[/green] Default model for [bold]{provider.value}[/bold] set to [bold]{model}[/bold].")


@config_app.command("list")
def config_list() -> None:
    """Show current configuration. Never reveals API key values."""
    cm = ConfigManager()
    info = cm.list_config()

    console.print(f"\n[bold]Active provider:[/bold] {info['active_provider']}")
    if info.get("vault_path"):
        console.print(f"[bold]Vault path:[/bold]      {info['vault_path']}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Key")
    table.add_column("Source")
    table.add_column("Default model")

    for name, details in info["providers"].items():
        key_cell = "[green]yes[/green]" if details["key_configured"] else "[dim]no[/dim]"
        source = details.get("key_source") or "—"
        model = details.get("model") or "—"
        table.add_row(name, key_cell, source, model)

    console.print(table)


@config_app.command("list-models")
def config_list_models(
    provider: str | None = typer.Option(None, "--provider", "-p", help="Provider to query (default: active)"),
) -> None:
    """List available models for the active provider."""
    cm = ConfigManager()
    llm = _build_provider(cm, provider)
    models = llm.list_models()

    if not models:
        console.print("[yellow]No models returned by provider.[/yellow]")
        return

    console.print(f"\n[bold]Available models:[/bold]")
    for m in models:
        console.print(f"  [cyan]•[/cyan] {m}")


if __name__ == "__main__":
    app()
