"""Tests for twin/ingestion/pdf.py."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from twin.config import AppConfig, EmbeddingModel

pytestmark = pytest.mark.skipif(not HAS_FITZ, reason="pymupdf not installed")


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path,
        embed_model=EmbeddingModel.NOMIC,
        chunk_tokens=512,
        overlap_tokens=64,
        top_k=5,
    )


def _make_pdf(tmp_path: Path, pages: list[str]) -> Path:
    """Create a minimal PDF with one page per string in pages."""
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    for content in pages:
        page = doc.new_page()
        page.insert_text((50, 50), content)
    doc.save(str(path))
    doc.close()
    return path


# ── parse_pdf: basic behaviour ───────────────────────────────────────────────

def test_parse_pdf_returns_chunks(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Hello world. This is the first page content."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    assert len(chunks) >= 1


def test_parse_pdf_chunk_fields(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Sample content on page one."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    chunk = chunks[0]
    assert chunk.doc_id == path.stem
    assert chunk.source_path == str(path)
    assert chunk.text.strip() != ""
    assert isinstance(chunk.heading_path, list)
    assert isinstance(chunk.chunk_index, int)
    assert chunk.token_count >= 1


def test_parse_pdf_heading_path_contains_page_number(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Page one text.", "Page two text."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    heading_paths = [c.heading_path for c in chunks]
    assert any(hp == ["p.1"] for hp in heading_paths)
    assert any(hp == ["p.2"] for hp in heading_paths)


def test_parse_pdf_source_attribution_format(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Content here."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    assert chunks[0].heading_path[0].startswith("p.")


def test_parse_pdf_chunk_ids_are_unique(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Page one.", "Page two.", "Page three."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_parse_pdf_chunk_indices_are_sequential(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Page one.", "Page two."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_parse_pdf_empty_pages_skipped(tmp_path: Path, config: AppConfig) -> None:
    """PDF with one empty page and one content page produces chunks only for content page."""
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    doc.new_page()  # empty page (no text)
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Content on page two.")
    doc.save(str(path))
    doc.close()

    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    assert len(chunks) >= 1
    assert all(c.heading_path == ["p.2"] for c in chunks)


def test_parse_pdf_multipage_all_chunks_have_correct_source_path(
    tmp_path: Path, config: AppConfig
) -> None:
    path = _make_pdf(tmp_path, ["Page 1.", "Page 2.", "Page 3."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    assert all(c.source_path == str(path) for c in chunks)


# ── parse_pdf: error cases ───────────────────────────────────────────────────

def test_parse_pdf_missing_file_raises(tmp_path: Path, config: AppConfig) -> None:
    from twin.ingestion.pdf import parse_pdf
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "nonexistent.pdf", config)


def test_parse_pdf_raises_on_missing_pymupdf(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["content"])
    from twin.ingestion import pdf as pdf_module
    original = pdf_module._HAS_FITZ
    try:
        pdf_module._HAS_FITZ = False
        with pytest.raises(ImportError, match="pymupdf"):
            pdf_module.parse_pdf(path, config)
    finally:
        pdf_module._HAS_FITZ = original


# ── idempotency: same bytes → same doc_id ───────────────────────────────────

def test_parse_pdf_doc_id_is_stem(tmp_path: Path, config: AppConfig) -> None:
    path = _make_pdf(tmp_path, ["Some content."])
    from twin.ingestion.pdf import parse_pdf
    chunks = parse_pdf(path, config)
    assert all(c.doc_id == path.stem for c in chunks)
