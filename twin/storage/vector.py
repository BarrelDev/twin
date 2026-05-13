import json
from dataclasses import dataclass
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

TABLE_NAME = "chunks"
EMBEDDING_DIM = 768


class ChunkRecord(LanceModel):
    chunk_id: str
    doc_id: str
    text: str
    embedding: Vector(dim=EMBEDDING_DIM)
    source_path: str
    heading_path: str  # JSON-encoded list[str]
    chunk_index: int


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    score: float


class VectorStore:
    """LanceDB-backed vector store for chunk embeddings."""

    def __init__(self, db_path: Path) -> None:
        self._db = lancedb.connect(str(db_path))
        self._table = self._db.create_table(TABLE_NAME)

    def _open_or_create_table(self) -> lancedb.table.Table:
        """Open the chunks table if it exists, otherwise create it."""
        for s in self._db.table_names():
            if s == TABLE_NAME:
                return self._db.open_table(TABLE_NAME)
        
        return self._db.create_table(TABLE_NAME)

    def write_chunks(
        self,
        chunks: list,  # list[Chunk] — avoid circular import
        embeddings: list[list[float]],
    ) -> None:
        """
        Persist a batch of chunks and their embeddings.

        Args:
            chunks: Chunk objects to store.
            embeddings: Embedding vectors, one per chunk, same order.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        source_path: str | None = None,
    ) -> list[SearchResult]:
        """
        Run ANN search and return ranked results.

        Args:
            query_embedding: Query vector to search against.
            k: Number of results to return.
            source_path: If provided, restrict results to this source file.

        Returns:
            List of SearchResult ordered by relevance (best first).
        """
        ...
