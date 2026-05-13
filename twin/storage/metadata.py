from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select


class DocRecord(SQLModel, table=True):
    doc_id:           str = Field(primary_key=True)
    source_path:      str
    file_hash:        str
    ingest_timestamp: str
    chunk_count:      int
    embedding_model:  str


class MetadataStore:
    """SQLite-backed registry of ingested documents."""

    def __init__(self, db_path: Path) -> None:
        """
        Open (or create) the SQLite database and ensure the schema exists.

        Args:
            db_path: Path to the .db file. Created if it does not exist.
        """
        engine = create_engine(f"sqlite:///{str(db_path)}") 
        SQLModel.metadata.create_all(engine)

        self._engine = engine

    def get_hash(self, source_path: str) -> str | None:
        """
        Return the stored SHA-256 hash for a source file, or None if not yet ingested.

        Args:
            source_path: Absolute path string of the source file.

        Returns:
            Hex digest string if the file has been ingested, None otherwise.
        """
        with Session(self._engine) as session:
            statement = select(DocRecord).where(DocRecord.source_path == source_path)
            record = session.exec(statement).first()
            return record.file_hash if record else None


    def upsert_doc(self, record: DocRecord) -> None:
        """
        Insert or update a document record, matched on doc_id.

        Args:
            record: DocRecord to persist. Overwrites any existing record with the same doc_id.
        """
        with Session(self._engine) as session:
            session.merge(record)
            session.commit()

    def delete_doc(self, doc_id: str) -> None:
        """
        Remove a document record by doc_id.

        Called before re-ingesting a changed file so stale metadata is cleared.

        Args:
            doc_id: Primary key of the record to delete.
        """
        with Session(self._engine) as session:
            record = session.get(DocRecord, doc_id)
            if record:
                session.delete(record)
                session.commit()


    def list_docs(self) -> list[DocRecord]:
        """
        Return all tracked document records.

        Returns:
            List of DocRecord instances, one per ingested document.
        """
        with Session(self._engine) as session:
            return list(session.exec(select(DocRecord)).all())
