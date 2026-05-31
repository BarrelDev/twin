import hashlib
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.text import Text

from twin.config import AppConfig
from twin.ingestion.embedder import build_embedder
from twin.ingestion.parser import parse_file
from twin.query.retriever import Retriever
from twin.storage.metadata import DocRecord, MetadataStore
from twin.storage.vector import VectorStore

app = typer.Typer(name="twin", help="Local-first semantic search for personal notes")
console = Console()


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
) -> None:
    """Synthesize an answer from retrieved context using an LLM."""
    from twin.llm.anthropic import Claude
    from twin.rag.pipeline import RAGPipeline

    config = AppConfig.from_env()

    try:
        llm = Claude()
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)
    pipeline = RAGPipeline(retriever, llm)

    with console.status("[bold blue]Searching and synthesizing...[/bold blue]"):
        output = pipeline.query(q, k=k)

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
) -> None:
    """Invoke the agent to complete a multi-step task over the knowledge base."""
    from twin.llm.anthropic import Claude
    from twin.agent.runtime import AgentRuntime
    from twin.agent.tools import ToolDispatcher

    config = AppConfig.from_env()

    try:
        llm = Claude()
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    store = VectorStore(config.data_dir / "lancedb")
    embedder = build_embedder(config)
    retriever = Retriever(store, embedder)
    dispatcher = ToolDispatcher(retriever)
    runtime = AgentRuntime(llm, dispatcher, max_iterations=max_iterations)

    with console.status("[bold blue]Agent thinking...[/bold blue]"):
        output = runtime.execute(task)

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


if __name__ == "__main__":
    app()
