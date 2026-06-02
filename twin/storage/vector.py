import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lancedb import connect, table
from lancedb.pydantic import LanceModel, Vector

TABLE_NAME = "chunks"
EMBEDDING_DIM = 768


class ChunkRecord(LanceModel):
    chunk_id: str
    doc_id: str
    text: str
    embedding: Vector(dim=EMBEDDING_DIM)
    source_path: str
    heading_path: str       # JSON-encoded list[str]
    chunk_index: int
    link_targets: str = ""  # JSON-encoded list[str] — Obsidian wikilink note names
    tags: str = ""          # JSON-encoded list[str] — Obsidian tags


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    score: float
    link_targets: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


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
        chunks: list[Any],  # duck-typed: Chunk, _Chunk from pdf/url/obsidian, or MockChunk
        embeddings: list[list[float]],
    ) -> None:
        """
        Persist a batch of chunks and their embeddings.

        Accepts any duck-typed chunk object with chunk_id, doc_id, text,
        source_path, heading_path, and chunk_index attributes. Optional
        link_targets and tags attributes are stored when present.

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
                chunk_index=c.chunk_index,
                link_targets=json.dumps(getattr(c, "link_targets", [])),
                tags=json.dumps(getattr(c, "tags", [])),
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

        def toSearchResult(row: dict) -> SearchResult:
            return SearchResult(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                source_path=row["source_path"],
                heading_path=json.loads(row["heading_path"]),
                chunk_index=row["chunk_index"],
                score=row["_distance"],
                link_targets=json.loads(row.get("link_targets") or "[]"),
                tags=json.loads(row.get("tags") or "[]"),
            )

        return list(map(toSearchResult, results))

