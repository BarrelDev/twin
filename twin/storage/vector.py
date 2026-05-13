import json
from dataclasses import dataclass
from pathlib import Path

from lancedb import connect
from lancedb import table
from lancedb.pydantic import LanceModel, Vector

from twin.ingestion.parser import Chunk

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
        self._db = connect(str(db_path))
        self._table = self._open_or_create_table()

    def _open_or_create_table(self) -> table.Table:
        """Open the chunks table if it exists, otherwise create it."""
        names = self._db.list_tables().tables
        if TABLE_NAME in str(names):
            return self._db.open_table(TABLE_NAME)
        
        return self._db.create_table(TABLE_NAME, schema=ChunkRecord)

    def write_chunks(
        self,
        chunks: list[Chunk],  # list[Chunk] — avoid circular import
        embeddings: list[list[float]],
    ) -> None:
        """
        Persist a batch of chunks and their embeddings.

        Args:
            chunks: Chunk objects to store.
            embeddings: Embedding vectors, one per chunk, same order.
        """
        table = self._open_or_create_table()
        records = [
            ChunkRecord(
                chunk_id=c.chunk_id, 
                doc_id=c.doc_id, 
                text=c.text, 
                embedding=e, 
                source_path=c.source_path, 
                heading_path=json.dumps(c.heading_path), 
                chunk_index=c.chunk_index
            )
            for (c, e) in zip(chunks, embeddings)
        ]
        table.add(records)


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
        table = self._open_or_create_table()
        query = table.search(query_embedding).limit(k)

        if source_path: 
            query = query.where(f"source_path = '{source_path}'")
        results = query.to_list()

        # Convert each 
        def toSearchResult (row: dict) -> SearchResult:
            return SearchResult(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                source_path=row["source_path"],
                heading_path=json.loads(row["heading_path"]),
                chunk_index=row["chunk_index"],
                score=row["_distance"],
            )

        return list(map(toSearchResult, results))

