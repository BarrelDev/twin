"""Tests for twin/ingestion/url.py."""

import hashlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from twin.config import AppConfig, EmbeddingModel


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path,
        embed_model=EmbeddingModel.NOMIC,
        chunk_tokens=512,
        overlap_tokens=64,
        top_k=5,
    )


_FAKE_HTML = "<html><body><article><p>This is the main article content. It discusses important topics.</p></article></body></html>"
_FAKE_TEXT = "This is the main article content. It discusses important topics."
_FAKE_URL = "https://example.com/article"


def _make_metadata(title: str = "Test Article") -> MagicMock:
    meta = MagicMock()
    meta.title = title
    return meta


# ── ingest_url: basic behaviour ──────────────────────────────────────────────

def test_ingest_url_returns_chunks_and_hash(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        chunks, content_hash = ingest_url(_FAKE_URL, config)

    assert len(chunks) >= 1
    assert isinstance(content_hash, str)
    assert len(content_hash) == 64  # SHA-256 hex digest


def test_ingest_url_content_hash_matches_downloaded_bytes(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        _, content_hash = ingest_url(_FAKE_URL, config)

    expected = hashlib.sha256(_FAKE_HTML.encode("utf-8", errors="replace")).hexdigest()
    assert content_hash == expected


def test_ingest_url_chunk_fields(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        chunks, _ = ingest_url(_FAKE_URL, config)

    chunk = chunks[0]
    assert chunk.source_path == _FAKE_URL
    assert chunk.text.strip() != ""
    assert isinstance(chunk.heading_path, list)
    assert len(chunk.heading_path) == 1
    assert chunk.chunk_index == 0
    assert chunk.token_count >= 1


def test_ingest_url_heading_path_contains_title(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata("My Article")),
    ):
        chunks, _ = ingest_url(_FAKE_URL, config)

    assert chunks[0].heading_path == ["My Article"]


def test_ingest_url_doc_id_is_url_hash(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        chunks, _ = ingest_url(_FAKE_URL, config)

    expected_doc_id = hashlib.sha256(_FAKE_URL.encode()).hexdigest()[:16]
    assert all(c.doc_id == expected_doc_id for c in chunks)


def test_ingest_url_chunk_ids_are_unique(config: AppConfig) -> None:
    long_text = " ".join(["word"] * 2000)  # force multiple chunks
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=long_text),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        chunks, _ = ingest_url(_FAKE_URL, config)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_ingest_url_chunk_indices_are_sequential(config: AppConfig) -> None:
    long_text = " ".join(["word"] * 2000)
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=long_text),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        chunks, _ = ingest_url(_FAKE_URL, config)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_ingest_url_source_path_is_url(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        chunks, _ = ingest_url(_FAKE_URL, config)

    assert all(c.source_path == _FAKE_URL for c in chunks)


# ── ingest_url: fallback title ───────────────────────────────────────────────

def test_ingest_url_falls_back_to_path_when_no_title(config: AppConfig) -> None:
    meta_no_title = MagicMock()
    meta_no_title.title = None
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=meta_no_title),
    ):
        chunks, _ = ingest_url("https://example.com/my-page", config)

    assert chunks[0].heading_path[0] != ""


def test_ingest_url_falls_back_to_domain_when_no_path_or_title(config: AppConfig) -> None:
    meta_no_title = MagicMock()
    meta_no_title.title = None
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=meta_no_title),
    ):
        chunks, _ = ingest_url("https://example.com", config)

    assert chunks[0].heading_path[0] == "example.com"


# ── ingest_url: error cases ───────────────────────────────────────────────────

def test_ingest_url_raises_on_fetch_failure(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with patch("twin.ingestion.url.trafilatura.fetch_url", return_value=None):
        with pytest.raises(ValueError, match="Failed to fetch"):
            ingest_url(_FAKE_URL, config)


def test_ingest_url_raises_on_empty_extraction(config: AppConfig) -> None:
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML),
        patch("twin.ingestion.url.trafilatura.extract", return_value=None),
    ):
        with pytest.raises(ValueError, match="No extractable text"):
            ingest_url(_FAKE_URL, config)


def test_ingest_url_raises_on_missing_trafilatura(config: AppConfig) -> None:
    from twin.ingestion import url as url_module
    original = url_module._HAS_TRAFILATURA
    try:
        url_module._HAS_TRAFILATURA = False
        with pytest.raises(ImportError, match="trafilatura"):
            url_module.ingest_url(_FAKE_URL, config)
    finally:
        url_module._HAS_TRAFILATURA = original


# ── never makes real HTTP requests ───────────────────────────────────────────

def test_ingest_url_never_calls_real_network(config: AppConfig) -> None:
    """Verify that all network calls are mocked — no real HTTP traffic allowed."""
    from twin.ingestion.url import ingest_url
    with (
        patch("twin.ingestion.url.trafilatura.fetch_url", return_value=_FAKE_HTML) as mock_fetch,
        patch("twin.ingestion.url.trafilatura.extract", return_value=_FAKE_TEXT),
        patch("twin.ingestion.url.trafilatura.extract_metadata", return_value=_make_metadata()),
    ):
        ingest_url(_FAKE_URL, config)
        mock_fetch.assert_called_once_with(_FAKE_URL)
