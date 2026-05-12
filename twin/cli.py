import typer
from rich.console import Console

app = typer.Typer(name="twin", help="Local-first knowledge OS")
console = Console()

@app.command()
def ingest(path: str = typer.Argument(..., help="Path to notes folder")):
    """Ingest a folder of notes into the knowledge base."""
    console.print(f"[bold blue]twin[/bold blue] ingest: {path} (not yet implemented)")

@app.command()
def query(q: str = typer.Argument(..., help="Query string")):
    """Query the knowledge base."""
    console.print(f"[bold blue]twin[/bold blue] query: {q} (not yet implemented)")

if __name__ == "__main__":
    app()