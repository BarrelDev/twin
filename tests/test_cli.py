"""Tests for the twin CLI commands."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from twin.agent.runtime import AgentOutput
from twin.cli import app
from twin.query.retriever import QueryResult
from twin.rag.pipeline import RAGOutput


runner = CliRunner()


# Ensure ConfigManager.resolve_api_key() succeeds in all CLI tests by default.
# Tests that specifically test the missing-key path must clear this via monkeypatch.
@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-cli-tests")


# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_query_results() -> list[QueryResult]:
    """Two QueryResult chunks for retriever mocking."""
    return [
        QueryResult(
            chunk_id="c1",
            text="Rust ownership is about memory safety.",
            source_path="/notes/rust.md",
            heading_path=["Ownership"],
            score=0.95,
        ),
        QueryResult(
            chunk_id="c2",
            text="The borrow checker enforces these rules.",
            source_path="/notes/rust.md",
            heading_path=["Borrow Checker"],
            score=0.88,
        ),
    ]


@pytest.fixture
def sample_rag_output() -> RAGOutput:
    """Sample RAGOutput for mocking pipeline.query()."""
    return RAGOutput(
        answer="Rust uses ownership to manage memory without a GC.",
        sources=[{"path": "rust.md", "heading_path": ["Ownership"]}],
        context_chunks=[],
    )


@pytest.fixture
def sample_agent_output() -> AgentOutput:
    """Sample AgentOutput with one tool call and a full activity log."""
    return AgentOutput(
        final_answer="Based on your notes, Rust ownership means exclusive control.",
        tool_calls=1,
        activity_log=[
            {
                "iteration": 0,
                "event_type": "tool_call",
                "details": {
                    "tool_name": "search_knowledge_base",
                    "tool_input": {"query": "ownership"},
                },
            },
            {
                "iteration": 0,
                "event_type": "tool_result",
                "details": {"result": "Ownership text from notes"},
            },
            {
                "iteration": 1,
                "event_type": "final_answer",
                "details": {"reason": "final_answer"},
            },
        ],
    )


def _mock_chunk(doc_id: str = "doc1") -> MagicMock:
    """Minimal mock chunk with the attributes ingest reads."""
    chunk = MagicMock()
    chunk.doc_id = doc_id
    chunk.text = "Sample text"
    return chunk


# ──────────────────────────────────────────────
# twin ingest
# ──────────────────────────────────────────────


class TestIngestCommand:
    """Tests for `twin ingest <path>`."""

    def test_nonexistent_path_exits_with_error(self) -> None:
        """Exit code 1 and an error message when the path does not exist."""
        result = runner.invoke(app, ["ingest", "/does/not/exist/abc"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_empty_directory_warns_and_exits_cleanly(self, tmp_path: Path) -> None:
        """Print a warning and exit 0 when the directory has no .md files."""
        result = runner.invoke(app, ["ingest", str(tmp_path)])
        assert result.exit_code == 0
        assert "No .md files found" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.MetadataStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.parse_file")
    def test_ingests_new_files(
        self,
        mock_parse_file: MagicMock,
        mock_build_embedder: MagicMock,
        mock_meta_cls: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """New files are written to the vector store and metadata store."""
        (tmp_path / "note.md").write_text("# Hello\n\nContent here.")

        chunks = [_mock_chunk("doc1"), _mock_chunk("doc1")]
        mock_parse_file.return_value = chunks
        mock_build_embedder.return_value.embed_batch.return_value = [[0.1] * 768] * 2
        mock_meta_cls.return_value.get_hash.return_value = None  # no prior hash

        result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0
        mock_store_cls.return_value.write_chunks.assert_called_once_with(chunks, [[0.1] * 768] * 2)
        mock_meta_cls.return_value.upsert_doc.assert_called_once()

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.MetadataStore")
    @patch("twin.cli.build_embedder")
    def test_skips_unchanged_files(
        self,
        mock_build_embedder: MagicMock,
        mock_meta_cls: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Files whose SHA-256 matches the stored hash are skipped without writing."""
        md_file = tmp_path / "note.md"
        md_file.write_text("# Hello\n\nContent.")
        file_hash = hashlib.sha256(md_file.read_bytes()).hexdigest()
        mock_meta_cls.return_value.get_hash.return_value = file_hash

        result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0
        mock_store_cls.return_value.write_chunks.assert_not_called()
        mock_meta_cls.return_value.upsert_doc.assert_not_called()

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.MetadataStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.parse_file")
    def test_reingest_changed_file(
        self,
        mock_parse_file: MagicMock,
        mock_build_embedder: MagicMock,
        mock_meta_cls: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A file with a different hash is re-ingested, not skipped."""
        (tmp_path / "note.md").write_text("# Updated\n\nNew content.")
        mock_parse_file.return_value = [_mock_chunk("doc1")]
        mock_build_embedder.return_value.embed_batch.return_value = [[0.1] * 768]
        mock_meta_cls.return_value.get_hash.return_value = "stale_hash_abc"  # differs

        result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0
        mock_store_cls.return_value.write_chunks.assert_called_once()
        mock_meta_cls.return_value.upsert_doc.assert_called_once()

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.MetadataStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.parse_file")
    def test_multiple_files_ingested_separately(
        self,
        mock_parse_file: MagicMock,
        mock_build_embedder: MagicMock,
        mock_meta_cls: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Each .md file is parsed and written independently."""
        (tmp_path / "a.md").write_text("# A\n\nContent A.")
        (tmp_path / "b.md").write_text("# B\n\nContent B.")
        mock_parse_file.return_value = [_mock_chunk("doc1")]
        mock_build_embedder.return_value.embed_batch.return_value = [[0.1] * 768]
        mock_meta_cls.return_value.get_hash.return_value = None

        result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0
        assert mock_store_cls.return_value.write_chunks.call_count == 2
        assert mock_meta_cls.return_value.upsert_doc.call_count == 2


# ──────────────────────────────────────────────
# twin query
# ──────────────────────────────────────────────


class TestQueryCommand:
    """Tests for `twin query <q>`."""

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    def test_no_results_warns(
        self,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Print a warning and exit 0 when the knowledge base returns nothing."""
        mock_retriever_cls.return_value.query.return_value = []

        result = runner.invoke(app, ["query", "what is ownership?"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    def test_results_calls_format_results(
        self,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_query_results: list[QueryResult],
    ) -> None:
        """format_results() is called with the returned chunks."""
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.query.return_value = sample_query_results
        mock_retriever.format_results.return_value = "formatted table"

        result = runner.invoke(app, ["query", "what is ownership?"])

        assert result.exit_code == 0
        mock_retriever.format_results.assert_called_once_with(sample_query_results)

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    def test_query_string_is_forwarded_to_retriever(
        self,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """The exact query string is forwarded to retriever.query() with default k=5."""
        mock_retriever_cls.return_value.query.return_value = []

        runner.invoke(app, ["query", "my specific question"])

        mock_retriever_cls.return_value.query.assert_called_once_with(
            "my specific question", k=5
        )


# ──────────────────────────────────────────────
# twin rag
# ──────────────────────────────────────────────


class TestRagCommand:
    """Tests for `twin rag <query>`."""

    @patch("twin.llm.anthropic.Claude")
    def test_missing_api_key_exits_with_error(
        self, mock_claude_cls: MagicMock
    ) -> None:
        """Exit code 1 and informative message when ANTHROPIC_API_KEY is absent."""
        mock_claude_cls.side_effect = ValueError("ANTHROPIC_API_KEY not set")

        result = runner.invoke(app, ["rag", "what is ownership?"])

        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.rag.pipeline.RAGPipeline")
    def test_answer_appears_in_output(
        self,
        mock_pipeline_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_rag_output: RAGOutput,
    ) -> None:
        """The synthesized answer text appears in the output."""
        mock_pipeline_cls.return_value.query.return_value = sample_rag_output

        result = runner.invoke(app, ["rag", "what is ownership?"])

        assert result.exit_code == 0
        assert sample_rag_output.answer in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.rag.pipeline.RAGPipeline")
    def test_sources_appear_in_output(
        self,
        mock_pipeline_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_rag_output: RAGOutput,
    ) -> None:
        """Source filenames appear below the answer."""
        mock_pipeline_cls.return_value.query.return_value = sample_rag_output

        result = runner.invoke(app, ["rag", "what is ownership?"])

        assert result.exit_code == 0
        assert "rust.md" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.rag.pipeline.RAGPipeline")
    def test_empty_sources_omits_sources_section(
        self,
        mock_pipeline_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Sources section is omitted when the pipeline returns no sources."""
        output = RAGOutput(answer="Some answer.", sources=[], context_chunks=[])
        mock_pipeline_cls.return_value.query.return_value = output

        result = runner.invoke(app, ["rag", "question"])

        assert result.exit_code == 0
        assert "Sources" not in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.rag.pipeline.RAGPipeline")
    def test_top_k_option_forwarded_to_pipeline(
        self,
        mock_pipeline_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_rag_output: RAGOutput,
    ) -> None:
        """--top-k is forwarded as k= to pipeline.query()."""
        mock_pipeline_cls.return_value.query.return_value = sample_rag_output

        runner.invoke(app, ["rag", "question", "--top-k", "3"])

        mock_pipeline_cls.return_value.query.assert_called_once_with("question", k=3)

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.rag.pipeline.RAGPipeline")
    def test_pipeline_constructed_with_retriever_and_llm(
        self,
        mock_pipeline_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_rag_output: RAGOutput,
    ) -> None:
        """RAGPipeline receives the Retriever and Claude instances."""
        mock_pipeline_cls.return_value.query.return_value = sample_rag_output

        runner.invoke(app, ["rag", "question"])

        mock_pipeline_cls.assert_called_once_with(
            mock_retriever_cls.return_value,
            mock_claude_cls.return_value,
        )


# ──────────────────────────────────────────────
# twin agent
# ──────────────────────────────────────────────


class TestAgentCommand:
    """Tests for `twin agent <task>`."""

    @patch("twin.llm.anthropic.Claude")
    def test_missing_api_key_exits_with_error(
        self, mock_claude_cls: MagicMock
    ) -> None:
        """Exit code 1 and informative message when ANTHROPIC_API_KEY is absent."""
        mock_claude_cls.side_effect = ValueError("ANTHROPIC_API_KEY not set")

        result = runner.invoke(app, ["agent", "summarize my notes"])

        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.agent.runtime.AgentRuntime")
    @patch("twin.agent.tools.ToolDispatcher")
    def test_final_answer_appears_in_output(
        self,
        mock_dispatcher_cls: MagicMock,
        mock_runtime_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_agent_output: AgentOutput,
    ) -> None:
        """The agent's final answer text appears in the output."""
        mock_runtime_cls.return_value.execute.return_value = sample_agent_output

        result = runner.invoke(app, ["agent", "summarize rust ownership"])

        assert result.exit_code == 0
        assert sample_agent_output.final_answer in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.agent.runtime.AgentRuntime")
    @patch("twin.agent.tools.ToolDispatcher")
    def test_tool_call_count_appears_in_output(
        self,
        mock_dispatcher_cls: MagicMock,
        mock_runtime_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_agent_output: AgentOutput,
    ) -> None:
        """The tool call count is displayed after the answer panel."""
        mock_runtime_cls.return_value.execute.return_value = sample_agent_output

        result = runner.invoke(app, ["agent", "task"])

        assert result.exit_code == 0
        assert "Tool calls made: 1" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.agent.runtime.AgentRuntime")
    @patch("twin.agent.tools.ToolDispatcher")
    def test_verbose_shows_activity_log(
        self,
        mock_dispatcher_cls: MagicMock,
        mock_runtime_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_agent_output: AgentOutput,
    ) -> None:
        """--verbose prints the Activity Log section with tool call and final answer entries."""
        mock_runtime_cls.return_value.execute.return_value = sample_agent_output

        result = runner.invoke(app, ["agent", "task", "--verbose"])

        assert result.exit_code == 0
        assert "Activity Log" in result.output
        assert "tool_call" in result.output
        assert "done" in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.agent.runtime.AgentRuntime")
    @patch("twin.agent.tools.ToolDispatcher")
    def test_no_verbose_hides_activity_log(
        self,
        mock_dispatcher_cls: MagicMock,
        mock_runtime_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_agent_output: AgentOutput,
    ) -> None:
        """Activity log is hidden by default (no --verbose flag)."""
        mock_runtime_cls.return_value.execute.return_value = sample_agent_output

        result = runner.invoke(app, ["agent", "task"])

        assert result.exit_code == 0
        assert "Activity Log" not in result.output

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.agent.runtime.AgentRuntime")
    @patch("twin.agent.tools.ToolDispatcher")
    def test_max_iter_forwarded_to_runtime(
        self,
        mock_dispatcher_cls: MagicMock,
        mock_runtime_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_agent_output: AgentOutput,
    ) -> None:
        """--max-iter is forwarded as max_iterations= to AgentRuntime."""
        mock_runtime_cls.return_value.execute.return_value = sample_agent_output

        runner.invoke(app, ["agent", "task", "--max-iter", "3"])

        mock_runtime_cls.assert_called_once_with(
            mock_claude_cls.return_value,
            mock_dispatcher_cls.return_value,
            max_iterations=3,
        )

    @patch("twin.cli.VectorStore")
    @patch("twin.cli.build_embedder")
    @patch("twin.cli.Retriever")
    @patch("twin.llm.anthropic.Claude")
    @patch("twin.agent.runtime.AgentRuntime")
    @patch("twin.agent.tools.ToolDispatcher")
    def test_task_string_forwarded_to_runtime(
        self,
        mock_dispatcher_cls: MagicMock,
        mock_runtime_cls: MagicMock,
        mock_claude_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_build_embedder: MagicMock,
        mock_store_cls: MagicMock,
        sample_agent_output: AgentOutput,
    ) -> None:
        """The task string is forwarded verbatim to runtime.execute()."""
        mock_runtime_cls.return_value.execute.return_value = sample_agent_output

        runner.invoke(app, ["agent", "summarize everything about decorators"])

        mock_runtime_cls.return_value.execute.assert_called_once_with(
            "summarize everything about decorators"
        )
