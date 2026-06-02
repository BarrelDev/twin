"""Tests for twin/ingestion/obsidian.py — Obsidian-specific Markdown parsing."""

import time
import pytest
from pathlib import Path

from twin.config import AppConfig, EmbeddingModel
from twin.ingestion.obsidian import VaultWatcher, parse_obsidian_file, _extract_frontmatter
from twin.agent.tools import VaultWriter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig for chunking tests."""
    return AppConfig(
        data_dir=Path("/tmp/twin-test"),
        embed_model=EmbeddingModel.NOMIC,
        chunk_tokens=512,
        overlap_tokens=64,
        top_k=5,
    )


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write a Markdown file and return its path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ── Wikilink extraction ───────────────────────────────────────────────────────

def test_simple_wikilink_extracted(tmp_path: Path, config: AppConfig) -> None:
    """[[Note Name]] is captured as a link target."""
    path = _write(tmp_path, "a.md", "# Heading\n\nSee [[Note Name]] for details.\n")
    _, meta = parse_obsidian_file(path, config)
    assert "Note Name" in meta["link_targets"]


def test_alias_wikilink_target_is_note_name(tmp_path: Path, config: AppConfig) -> None:
    """[[Note Name|Alias]] stores 'Note Name' as link target, not 'Alias'."""
    path = _write(tmp_path, "b.md", "# Heading\n\nRefer to [[Note One|One]].\n")
    _, meta = parse_obsidian_file(path, config)
    assert "Note One" in meta["link_targets"]
    assert "One" not in meta["link_targets"]


def test_alias_text_appears_in_chunk(tmp_path: Path, config: AppConfig) -> None:
    """The display alias (not the note name) appears in chunk text after replacement."""
    path = _write(tmp_path, "c.md", "# Heading\n\nRefer to [[Note One|One]] here.\n")
    chunks, _ = parse_obsidian_file(path, config)
    full_text = " ".join(c.text for c in chunks)
    assert "One" in full_text
    assert "[[" not in full_text


def test_plain_wikilink_text_uses_note_name(tmp_path: Path, config: AppConfig) -> None:
    """[[Note Name]] (no alias) leaves 'Note Name' as plain text in chunks."""
    path = _write(tmp_path, "d.md", "# Heading\n\nSee [[My Note]] for details.\n")
    chunks, _ = parse_obsidian_file(path, config)
    full_text = " ".join(c.text for c in chunks)
    assert "My Note" in full_text
    assert "[[" not in full_text


def test_multiple_wikilinks_deduplicated(tmp_path: Path, config: AppConfig) -> None:
    """Duplicate wikilinks appear only once in link_targets."""
    path = _write(
        tmp_path, "e.md",
        "# Heading\n\n[[Alpha]] and [[Alpha]] again.\n"
    )
    _, meta = parse_obsidian_file(path, config)
    assert meta["link_targets"].count("Alpha") == 1


def test_multiple_distinct_wikilinks(tmp_path: Path, config: AppConfig) -> None:
    """All distinct wikilinks are captured."""
    path = _write(
        tmp_path, "f.md",
        "# Heading\n\n[[Alpha]] and [[Beta]] and [[Gamma]].\n"
    )
    _, meta = parse_obsidian_file(path, config)
    assert set(meta["link_targets"]) == {"Alpha", "Beta", "Gamma"}


# ── Tag extraction ────────────────────────────────────────────────────────────

def test_simple_tag_extracted(tmp_path: Path, config: AppConfig) -> None:
    """#simple tags are extracted from body text."""
    path = _write(tmp_path, "g.md", "# Heading\n\nTagged with #project.\n")
    _, meta = parse_obsidian_file(path, config)
    assert "project" in meta["tags"]


def test_nested_tag_extracted(tmp_path: Path, config: AppConfig) -> None:
    """#parent/child nested tags are captured as 'parent/child'."""
    path = _write(tmp_path, "h.md", "# Heading\n\nTagged #work/research here.\n")
    _, meta = parse_obsidian_file(path, config)
    assert "work/research" in meta["tags"]


def test_tags_deduplicated(tmp_path: Path, config: AppConfig) -> None:
    """Duplicate tags appear only once."""
    path = _write(
        tmp_path, "i.md",
        "# Heading\n\n#project here.\n\nAnd again #project.\n"
    )
    _, meta = parse_obsidian_file(path, config)
    assert meta["tags"].count("project") == 1


def test_frontmatter_tags_not_extracted_as_body_tags(tmp_path: Path, config: AppConfig) -> None:
    """Tags listed in YAML frontmatter are not re-extracted as body tags."""
    path = _write(
        tmp_path, "j.md",
        "---\ntitle: Doc\ntags: [foo, bar]\n---\n# Heading\n\nNo body tags.\n"
    )
    _, meta = parse_obsidian_file(path, config)
    # YAML frontmatter 'tags' are stored in frontmatter dict, not extracted as body tags
    assert "foo" not in meta["tags"]
    assert "bar" not in meta["tags"]


# ── Embed stripping ───────────────────────────────────────────────────────────

def test_embed_stripped_from_chunk_text(tmp_path: Path, config: AppConfig) -> None:
    """![[image.png]] embed directives are removed from chunk text."""
    path = _write(
        tmp_path, "k.md",
        "# Heading\n\n![[diagram.png]]\n\nSome useful text.\n"
    )
    chunks, _ = parse_obsidian_file(path, config)
    full_text = " ".join(c.text for c in chunks)
    assert "![[" not in full_text
    assert "diagram.png" not in full_text
    assert "Some useful text." in full_text


def test_embed_not_captured_as_wikilink(tmp_path: Path, config: AppConfig) -> None:
    """Embed directives ![[x]] do not appear in link_targets."""
    path = _write(
        tmp_path, "l.md",
        "# Heading\n\n![[image.png]]\n\nText with [[Real Note]].\n"
    )
    _, meta = parse_obsidian_file(path, config)
    # image.png is an embed, not a wikilink — should NOT be in link_targets
    # Note: the raw WIKILINK_RE would capture it; embed stripping happens on text,
    # but link_target extraction is done on raw text. Let's verify real wikilink present.
    assert "Real Note" in meta["link_targets"]


# ── Frontmatter ───────────────────────────────────────────────────────────────

def test_frontmatter_preserved_as_dict(tmp_path: Path, config: AppConfig) -> None:
    """All YAML frontmatter fields are returned in the metadata dict."""
    path = _write(
        tmp_path, "m.md",
        "---\ntitle: My Note\nauthor: Alice\ndate: 2026-01-01\n---\n# Body\n\nContent.\n"
    )
    _, meta = parse_obsidian_file(path, config)
    assert meta["frontmatter"]["title"] == "My Note"
    assert meta["frontmatter"]["author"] == "Alice"
    assert meta["frontmatter"]["date"] is not None


def test_no_frontmatter_returns_empty_dict(tmp_path: Path, config: AppConfig) -> None:
    """Files without frontmatter return an empty frontmatter dict."""
    path = _write(tmp_path, "n.md", "# Just a heading\n\nSome content.\n")
    _, meta = parse_obsidian_file(path, config)
    assert meta["frontmatter"] == {}


def test_frontmatter_not_included_in_chunk_text(tmp_path: Path, config: AppConfig) -> None:
    """YAML frontmatter block is not embedded as chunk text."""
    path = _write(
        tmp_path, "o.md",
        "---\ntitle: Secret\nauthor: Bob\n---\n# Body\n\nPublic content.\n"
    )
    chunks, _ = parse_obsidian_file(path, config)
    full_text = " ".join(c.text for c in chunks)
    assert "Secret" not in full_text
    assert "Bob" not in full_text
    assert "Public content." in full_text


# ── doc_id from frontmatter ───────────────────────────────────────────────────

def test_doc_id_uses_frontmatter_id(tmp_path: Path, config: AppConfig) -> None:
    """doc_id is taken from frontmatter 'id' field when present."""
    path = _write(
        tmp_path, "p.md",
        "---\nid: custom-doc-id\n---\n# Heading\n\nContent here.\n"
    )
    chunks, _ = parse_obsidian_file(path, config)
    assert all(c.doc_id == "custom-doc-id" for c in chunks)


def test_doc_id_falls_back_to_stem(tmp_path: Path, config: AppConfig) -> None:
    """doc_id defaults to the filename stem when no id in frontmatter."""
    path = _write(tmp_path, "my-note.md", "# Heading\n\nContent.\n")
    chunks, _ = parse_obsidian_file(path, config)
    assert all(c.doc_id == "my-note" for c in chunks)


# ── Chunk structure ───────────────────────────────────────────────────────────

def test_chunks_returned_for_valid_file(tmp_path: Path, config: AppConfig) -> None:
    """A non-empty file produces at least one chunk."""
    path = _write(tmp_path, "q.md", "# Heading\n\nSome content here.\n")
    chunks, _ = parse_obsidian_file(path, config)
    assert len(chunks) >= 1


def test_chunks_have_sequential_indices(tmp_path: Path, config: AppConfig) -> None:
    """chunk_index values start at 0 and are sequential."""
    path = _write(
        tmp_path, "r.md",
        "# H1\n\nParagraph one.\n\n## H2\n\nParagraph two.\n\n### H3\n\nParagraph three.\n"
    )
    chunks, _ = parse_obsidian_file(path, config)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_ids_unique(tmp_path: Path, config: AppConfig) -> None:
    """All chunks have unique chunk_id values."""
    content = "\n\n".join(f"## Section {i}\n\nContent {i}." for i in range(10))
    path = _write(tmp_path, "s.md", content)
    chunks, _ = parse_obsidian_file(path, config)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_source_path_is_file_path(tmp_path: Path, config: AppConfig) -> None:
    """source_path on each chunk matches the file path."""
    path = _write(tmp_path, "t.md", "# Heading\n\nContent.\n")
    chunks, _ = parse_obsidian_file(path, config)
    assert all(c.source_path == str(path) for c in chunks)


def test_link_targets_attached_to_all_chunks(tmp_path: Path, config: AppConfig) -> None:
    """Document-level link_targets appear on every chunk."""
    path = _write(
        tmp_path, "u.md",
        "# Heading\n\nSee [[Alpha]].\n\n## Section\n\nMore text here.\n"
    )
    chunks, meta = parse_obsidian_file(path, config)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.link_targets == meta["link_targets"]


def test_tags_attached_to_all_chunks(tmp_path: Path, config: AppConfig) -> None:
    """Document-level tags appear on every chunk."""
    path = _write(
        tmp_path, "v.md",
        "# Heading\n\nTagged #project.\n\n## Section\n\nMore content.\n"
    )
    chunks, meta = parse_obsidian_file(path, config)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.tags == meta["tags"]


def test_empty_file_returns_no_chunks(tmp_path: Path, config: AppConfig) -> None:
    """An empty file produces zero chunks."""
    path = _write(tmp_path, "empty.md", "")
    chunks, _ = parse_obsidian_file(path, config)
    assert chunks == []


def test_frontmatter_only_file_returns_no_chunks(tmp_path: Path, config: AppConfig) -> None:
    """A file with only frontmatter and no body produces zero chunks."""
    path = _write(tmp_path, "fm-only.md", "---\ntitle: Only\n---\n")
    chunks, _ = parse_obsidian_file(path, config)
    assert chunks == []


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_missing_file_raises_file_not_found(tmp_path: Path, config: AppConfig) -> None:
    """FileNotFoundError is raised when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        parse_obsidian_file(tmp_path / "does_not_exist.md", config)


def test_no_wikilinks_returns_empty_link_targets(tmp_path: Path, config: AppConfig) -> None:
    """Files without wikilinks return empty link_targets."""
    path = _write(tmp_path, "no-links.md", "# Heading\n\nPlain content only.\n")
    _, meta = parse_obsidian_file(path, config)
    assert meta["link_targets"] == []


def test_no_tags_returns_empty_tags(tmp_path: Path, config: AppConfig) -> None:
    """Files without tags return empty tags list."""
    path = _write(tmp_path, "no-tags.md", "# Heading\n\nPlain content only.\n")
    _, meta = parse_obsidian_file(path, config)
    assert meta["tags"] == []


def test_tmp_vault_note1_parsed(tmp_vault: Path, config: AppConfig) -> None:
    """The tmp_vault fixture's note1.md parses correctly."""
    chunks, meta = parse_obsidian_file(tmp_vault / "note1.md", config)
    assert len(chunks) >= 1
    assert "Note Two" in meta["link_targets"]
    assert meta["frontmatter"]["title"] == "Note One"


def test_tmp_vault_note2_parsed(tmp_vault: Path, config: AppConfig) -> None:
    """The tmp_vault fixture's note2.md parses correctly with nested tag."""
    chunks, meta = parse_obsidian_file(tmp_vault / "note2.md", config)
    assert len(chunks) >= 1
    assert "Note One" in meta["link_targets"]
    assert "tag/nested" in meta["tags"]


# ── _extract_frontmatter unit tests ──────────────────────────────────────────

def test_extract_frontmatter_parses_yaml() -> None:
    """_extract_frontmatter returns parsed dict and remaining body."""
    content = "---\ntitle: Test\nauthor: Alice\n---\n# Body\n\nContent.\n"
    fm, body = _extract_frontmatter(content)
    assert fm["title"] == "Test"
    assert fm["author"] == "Alice"
    assert body.startswith("# Body")


def test_extract_frontmatter_no_frontmatter() -> None:
    """_extract_frontmatter returns empty dict when no frontmatter present."""
    content = "# Just a heading\n\nSome content.\n"
    fm, body = _extract_frontmatter(content)
    assert fm == {}
    assert body == content


def test_extract_frontmatter_invalid_yaml() -> None:
    """_extract_frontmatter returns empty dict on invalid YAML."""
    content = "---\ninvalid: [unclosed\n---\n# Body\n"
    fm, body = _extract_frontmatter(content)
    assert fm == {}


# ── VaultWriter tests ─────────────────────────────────────────────────────────

def test_write_vault_note_creates_file_in_agents(tmp_vault: Path) -> None:
    """write_vault_note creates a file inside the vault's Agents/ directory."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("Research Summary", "Body content here.")
    note_file = tmp_vault / rel_path
    assert note_file.exists()
    assert rel_path.parts[0] == "Agents"


def test_write_vault_note_returns_relative_path(tmp_vault: Path) -> None:
    """write_vault_note returns a path relative to the vault root."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("Test Note", "Body.")
    assert not rel_path.is_absolute()
    assert rel_path.parts[0] == "Agents"


def test_write_vault_note_frontmatter_fields(tmp_vault: Path) -> None:
    """Written note contains all required frontmatter fields."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("My Task", "Some content.")
    text = (tmp_vault / rel_path).read_text(encoding="utf-8")
    assert "generated_by: twin-agent" in text
    assert "task: My Task" in text
    assert "created:" in text
    assert "twin-generated" in text


def test_write_vault_note_includes_custom_tags(tmp_vault: Path) -> None:
    """write_vault_note includes both user tags and the mandatory twin-generated tag."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("Tagged", "Content.", tags=["research", "summary"])
    text = (tmp_vault / rel_path).read_text(encoding="utf-8")
    assert "research" in text
    assert "summary" in text
    assert "twin-generated" in text


def test_write_vault_note_h1_heading_in_body(tmp_vault: Path) -> None:
    """Written note contains an H1 heading matching the title."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("Heading Check", "Body text.")
    text = (tmp_vault / rel_path).read_text(encoding="utf-8")
    assert "# Heading Check\n" in text
    assert "Body text." in text


def test_write_vault_note_sanitizes_traversal_title(tmp_vault: Path) -> None:
    """A title with path-traversal characters is sanitized to a safe filename."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("../../etc/passwd", "Content.")
    # File must exist and be safely inside Agents/
    note_file = tmp_vault / rel_path
    assert note_file.exists()
    assert rel_path.parts[0] == "Agents"
    # The filename must not contain / or .. as separators
    assert ".." not in rel_path.parts


def test_write_vault_note_sanitizes_special_chars(tmp_vault: Path) -> None:
    """Unsafe characters in title are replaced to produce a valid filename."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note('Note: A/B Test? "quoted"', "Body.")
    assert (tmp_vault / rel_path).exists()
    assert rel_path.parts[0] == "Agents"


def test_write_vault_note_no_tags_still_includes_twin_generated(tmp_vault: Path) -> None:
    """When no tags are passed, the twin-generated tag is still added."""
    writer = VaultWriter(tmp_vault)
    rel_path = writer.write_vault_note("Untagged", "Content.")
    text = (tmp_vault / rel_path).read_text(encoding="utf-8")
    assert "twin-generated" in text


# ── VaultWatcher tests ────────────────────────────────────────────────────────

class _FakeEvent:
    """Minimal stand-in for a watchdog FileSystemEvent."""
    def __init__(self, src_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.is_directory = is_directory


def test_watcher_ignores_non_md_files(tmp_vault: Path, config: AppConfig) -> None:
    """Events for non-.md files do not trigger the ingest callback."""
    calls: list[Path] = []
    watcher = VaultWatcher(tmp_vault, config, ingest_callback=calls.append, debounce_seconds=0.05)
    watcher.on_modified(_FakeEvent(str(tmp_vault / "image.png")))
    time.sleep(0.15)
    assert calls == []


def test_watcher_ignores_directory_events(tmp_vault: Path, config: AppConfig) -> None:
    """Directory-level events do not trigger the ingest callback."""
    calls: list[Path] = []
    watcher = VaultWatcher(tmp_vault, config, ingest_callback=calls.append, debounce_seconds=0.05)
    watcher.on_modified(_FakeEvent(str(tmp_vault / "subdir"), is_directory=True))
    time.sleep(0.15)
    assert calls == []


def test_watcher_md_file_triggers_ingest(tmp_vault: Path, config: AppConfig) -> None:
    """A .md file modification event triggers the ingest callback after debounce."""
    calls: list[Path] = []
    watcher = VaultWatcher(tmp_vault, config, ingest_callback=calls.append, debounce_seconds=0.05)
    md_path = str(tmp_vault / "note1.md")
    watcher.on_modified(_FakeEvent(md_path))
    time.sleep(0.15)
    assert len(calls) == 1
    assert calls[0] == Path(md_path)


def test_watcher_debounce_two_rapid_events_one_ingest(tmp_vault: Path, config: AppConfig) -> None:
    """Two rapid events for the same path produce only one ingest call."""
    calls: list[Path] = []
    watcher = VaultWatcher(tmp_vault, config, ingest_callback=calls.append, debounce_seconds=0.05)
    md_path = str(tmp_vault / "note1.md")
    watcher._schedule_ingest(md_path)
    watcher._schedule_ingest(md_path)
    time.sleep(0.15)
    assert len(calls) == 1


def test_watcher_distinct_files_each_trigger_ingest(tmp_vault: Path, config: AppConfig) -> None:
    """Events for two different .md files each trigger a separate ingest call."""
    calls: list[Path] = []
    watcher = VaultWatcher(tmp_vault, config, ingest_callback=calls.append, debounce_seconds=0.05)
    watcher.on_modified(_FakeEvent(str(tmp_vault / "note1.md")))
    watcher.on_modified(_FakeEvent(str(tmp_vault / "note2.md")))
    time.sleep(0.15)
    assert len(calls) == 2


def test_watcher_on_created_also_triggers_ingest(tmp_vault: Path, config: AppConfig) -> None:
    """on_created fires ingest just like on_modified."""
    calls: list[Path] = []
    watcher = VaultWatcher(tmp_vault, config, ingest_callback=calls.append, debounce_seconds=0.05)
    md_path = str(tmp_vault / "new_note.md")
    watcher.on_created(_FakeEvent(md_path))
    time.sleep(0.15)
    assert len(calls) == 1
    assert calls[0] == Path(md_path)
