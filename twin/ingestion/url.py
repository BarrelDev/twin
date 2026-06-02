import hashlib
from dataclasses import dataclass
from urllib.parse import urlparse

from twin.config import AppConfig

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

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


def ingest_url(url: str, config: AppConfig) -> tuple[list[_Chunk], str]:
    """
    Fetch and extract main content from a URL.

    Source attribution format is "domain.com › Page Title" via
    source_path=url and heading_path=[title].

    Args:
        url: HTTP or HTTPS URL to fetch and ingest.
        config: AppConfig with chunk_tokens and overlap_tokens settings.

    Returns:
        Tuple of (chunks, content_sha256) where content_sha256 is the SHA-256
        hash of the raw downloaded bytes for idempotency checking.

    Raises:
        ImportError: If trafilatura or twin_core are not installed.
        ValueError: If the URL cannot be fetched or no text could be extracted.
    """
    if not _HAS_TRAFILATURA:
        raise ImportError(
            "trafilatura is required for URL ingestion. Install with: uv add trafilatura"
        )
    if not _USE_RUST:
        raise ImportError(
            "twin_core Rust extension is required. Run: cd twin_core && uv run maturin develop"
        )

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError(f"Failed to fetch URL: {url}")

    content_hash = hashlib.sha256(
        downloaded.encode("utf-8", errors="replace")
    ).hexdigest()

    text = trafilatura.extract(downloaded, include_metadata=False, output_format="txt")
    if not text or not text.strip():
        raise ValueError(f"No extractable text content at URL: {url}")

    metadata = trafilatura.extract_metadata(downloaded)
    title: str = (
        (metadata.title if metadata and metadata.title else None)
        or urlparse(url).path.strip("/")
        or urlparse(url).netloc
    )
    doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    chunks: list[_Chunk] = []
    chunk_index = 0

    for chunk_text in _rust_split(text.strip(), config.chunk_tokens, config.overlap_tokens):
        if not chunk_text.strip():
            continue
        chunks.append(
            _Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                text=chunk_text,
                source_path=url,
                heading_path=[title],
                chunk_index=chunk_index,
                token_count=_count_tokens(chunk_text),
            )
        )
        chunk_index += 1

    return chunks, content_hash
