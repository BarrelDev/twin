import hashlib
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

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


if __name__ == "__main__":
    app()
