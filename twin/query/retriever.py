import json
from dataclasses import dataclass
from pathlib import Path

from rich.table import Table

from twin.ingestion.embedder import Embedder
from twin.storage.vector import VectorStore


@dataclass
class QueryResult:
    """A single ranked result returned by the retriever."""

    chunk_id: str
    text: str
    source_path: str
    heading_path: list[str]
    score: float


class Retriever:
    """Orchestrates embedding, search, and result formatting for CLI queries."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        """
        Args:
            vector_store: Populated LanceDB vector store to search against.
            embedder: Embedder instance used to encode the query string.
        """
        self._vector_store = vector_store
        self._embedder = embedder

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        """
        Embed a natural-language query and return the top-k matching chunks.

        Args:
            text: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of QueryResult ordered by relevance (best first).
        """
        vec = self._embedder.embed_query(text)
        res = self._vector_store.search(vec, k=k)

        def toQueryResult(s) -> QueryResult:
            return QueryResult(chunk_id=s.chunk_id, 
                               text=s.text, 
                               source_path=s.source_path, 
                               heading_path=s.heading_path, 
                               score=s.score
                            )

        return list(map(toQueryResult, res))

    def format_results(self, results: list[QueryResult]) -> Table:
        """
        Render a list of QueryResult as a Rich Table for CLI display.

        Columns: rank, score, source (filename + heading path), text excerpt.

        Args:
            results: Ranked list of QueryResult from query().

        Returns:
            Rich Table object ready to pass to console.print().
        """
        table = Table(title="Search Results")

        table.add_column("rank", justify="left")
        table.add_column("score", justify="right")
        table.add_column("source", justify="left")
        table.add_column("text", justify="left")

        for i, q in enumerate(results):
            source = Path(q.source_path).name + " › " + " › ".join(q.heading_path)
            table.add_row(str(i + 1), "{:.2f}".format(q.score), source, q.text.replace("\n", " ")[:120])

        return table
    
    def results_as_json_str(self, results: list[QueryResult]) -> str:
        """
        Render a list of QueryResult as a JSON string.

        Args:
            results: Ranked list of QueryResult from query().

        Returns:
            JSON string: [{"text": str, "source": str, "score": float, "heading_path": [str]}]
        """
        data = []
        for q in results:
            source = Path(q.source_path).name + " › " + " › ".join(q.heading_path)
            data.append({
                "text": q.text,
                "source": source,
                "score": q.score,
                "heading_path": q.heading_path
            })
        return json.dumps(data)
