import hashlib
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text

from twin.config import AppConfig, Provider
from twin.config_manager import ConfigManager
from twin.ingestion.embedder import build_embedder
from twin.ingestion.parser import parse_file
from twin.llm.base import LLMProvider
from twin.query.retriever import Retriever
from twin.storage.metadata import DocRecord, MetadataStore
from twin.storage.vector import VectorStore

app = typer.Typer(name="twin", help="Local-first semantic search for personal notes")
config_app = typer.Typer(help="Manage Twin configuration and API keys")
app.add_typer(config_app, name="config")

console = Console()


def _build_provider(cm: ConfigManager, provider_override: str | None = None) -> LLMProvider:
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
    from twin.llm.anthropic import Claude
    from twin.llm.openai import OpenAIProvider
    from twin.llm.gemini import GeminiProvider
    from twin.llm.ollama import OllamaProvider
    from twin.llm.openrouter import OpenRouterProvider

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
                return Claude(api_key=api_key, model=model)
            case Provider.OPENAI:
                return OpenAIProvider(api_key=api_key, model=model)
            case Provider.GEMINI:
                return GeminiProvider(api_key=api_key, model=model)
            case Provider.OLLAMA:
                return OllamaProvider(model=model)
            case Provider.OPENROUTER:
                if not model:
                    console.print(
                        "[red]OpenRouter requires a model. "
                        "Run: twin config set-model <provider/model>[/red]"
                    )
                    raise typer.Exit(1)
                return OpenRouterProvider(api_key=api_key, model=model)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


# ── Core commands ────────────────────────────────────────────────────────────

@app.command()
def ingest(path: str = typer.Argument(..., help="Path to notes folder")) -> None:
    """Ingest a folder of Markdown notes into the knowledge base."""
    config = AppConfig.from_env()
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

    store = VectorStore(config.data_dir / "lancedb")
    meta = MetadataStore(config.data_dir / "meta.db")
    embedder = build_embedder(config)

    skipped = ingested = total_chunks = 0

    for md_file in track(md_files, description="Ingesting..."):
        file_hash = hashlib.sha256(md_file.read_bytes()).hexdigest()
        if meta.get_hash(str(md_file)) == file_hash:
            skipped += 1
            continue

        chunks = parse_file(md_file)
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
    from twin.rag.pipeline import RAGPipeline

    config = AppConfig.from_env()
    cm = ConfigManager()
    llm = _build_provider(cm, provider)

    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)
    pipeline = RAGPipeline(retriever, llm)

    import asyncio

    with console.status("[bold blue]Searching and synthesizing...[/bold blue]"):
        output = asyncio.run(pipeline.query(q, k=k))

    console.print(Panel(output.answer, title="[bold green]Answer[/bold green]", border_style="green"))

    if output.sources:
        console.print("\n[bold]Sources:[/bold]")
        for src in output.sources:
            path = Path(src["path"]).name
            heading = " > ".join(src["heading_path"]) if src["heading_path"] else ""
            line = Text(f"  • {path}")
            if heading:
                line.append(f"  {heading}", style="dim")
            console.print(line)


@app.command()
def agent(
    task: str = typer.Argument(..., help="Task description for the agent"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full activity log"),
    max_iterations: int = typer.Option(5, "--max-iter", help="Max tool calls before forced termination"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="LLM provider override"),
) -> None:
    """Invoke the agent to complete a multi-step task over the knowledge base."""
    from twin.agent.runtime import AgentRuntime
    from twin.agent.tools import ToolDispatcher

    config = AppConfig.from_env()
    cm = ConfigManager()
    llm = _build_provider(cm, provider)

    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)
    dispatcher = ToolDispatcher(retriever)
    runtime = AgentRuntime(llm, dispatcher, max_iterations=max_iterations)

    import asyncio

    with console.status("[bold blue]Agent thinking...[/bold blue]"):
        output = asyncio.run(runtime.execute(task))

    console.print(Panel(output.final_answer, title="[bold green]Agent Answer[/bold green]", border_style="green"))
    console.print(f"\n[dim]Tool calls made: {output.tool_calls}[/dim]")

    if verbose and output.activity_log:
        console.print("\n[bold]Activity Log:[/bold]")
        for entry in output.activity_log:
            event = entry.get("event_type", "")
            iteration = entry.get("iteration", 0)
            details = entry.get("details", {})
            if event == "tool_call":
                console.print(
                    f"  [cyan]iter {iteration}[/cyan] [yellow]tool_call[/yellow] "
                    f"{details.get('tool_name', '')}({details.get('tool_input', '')})"
                )
            elif event == "tool_result":
                snippet = str(details.get("result", ""))[:120]
                console.print(f"  [cyan]iter {iteration}[/cyan] [dim]result[/dim] {snippet}...")
            elif event == "final_answer":
                reason = details.get("reason", "final_answer")
                console.print(f"  [cyan]iter {iteration}[/cyan] [green]done[/green] ({reason})")


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
