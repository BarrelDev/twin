from dataclasses import dataclass
from pathlib import Path

from twin.config import AppConfig

try:
    import fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

try:
    from twin_core import (
        count_tokens as _rust_count_tokens,
        split_section_into_chunks as _rust_split,
    )
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    token_count: int


def _count_tokens(text: str) -> int:
    if _USE_RUST:
        return _rust_count_tokens(text)
    return len(text.split())


def parse_pdf(path: Path, config: AppConfig) -> list[_Chunk]:
    """
    Extract text from a PDF and split into chunks.

    Each page is treated as a separate section. Source attribution format is
    "filename.pdf › p.N" via heading_path=["p.N"].

    Args:
        path: Path to the PDF file.
        config: AppConfig with chunk_tokens and overlap_tokens settings.

    Returns:
        List of chunks, one or more per non-empty PDF page.

    Raises:
        ImportError: If pymupdf or twin_core are not installed.
        FileNotFoundError: If the PDF does not exist.
    """
    if not _HAS_FITZ:
        raise ImportError(
            "pymupdf is required for PDF ingestion. Install with: uv add pymupdf"
        )
    if not _USE_RUST:
        raise ImportError(
            "twin_core Rust extension is required. Run: cd twin_core && uv run maturin develop"
        )
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    doc = fitz.open(str(path))
    doc_id = path.stem
    chunks: list[_Chunk] = []
    chunk_index = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        for chunk_text in _rust_split(text, config.chunk_tokens, config.overlap_tokens):
            if not chunk_text.strip():
                continue
            chunks.append(
                _Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    doc_id=doc_id,
                    text=chunk_text,
                    source_path=str(path),
                    heading_path=[f"p.{page_num}"],
                    chunk_index=chunk_index,
                    token_count=_count_tokens(chunk_text),
                )
            )
            chunk_index += 1

    return chunks
