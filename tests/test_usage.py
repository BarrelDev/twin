"""Tests for twin/usage.py — UsageLogger and format_session_summary."""

import json
import pytest
from pathlib import Path

from twin.usage import UsageLogger, UsageRecord, format_session_summary


# ── Helpers ──────────────────────────────────────────────────────────────────

def _record(**kwargs) -> UsageRecord:
    """Create a UsageRecord with sensible defaults, overridden by kwargs."""
    defaults: dict = {
        "timestamp": "2026-06-01T12:00:00+00:00",
        "command": "rag",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "estimated_cost_usd": 0.001,
    }
    defaults.update(kwargs)
    return UsageRecord(**defaults)


# ── UsageLogger ───────────────────────────────────────────────────────────────

class TestUsageLoggerLog:

    def test_log_creates_jsonl_file(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        logger.log(_record())
        assert (tmp_path / "usage.jsonl").exists()

    def test_log_writes_valid_json_line(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        logger.log(_record(command="agent", provider="openai"))
        line = (tmp_path / "usage.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["command"] == "agent"
        assert parsed["provider"] == "openai"

    def test_log_appends_multiple_records(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        logger.log(_record(command="rag"))
        logger.log(_record(command="agent"))
        lines = (tmp_path / "usage.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["command"] == "rag"
        assert json.loads(lines[1])["command"] == "agent"

    def test_log_preserves_none_cost(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        logger.log(_record(estimated_cost_usd=None))
        line = (tmp_path / "usage.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["estimated_cost_usd"] is None

    def test_log_persists_all_fields(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        r = _record(
            timestamp="2026-01-15T08:30:00+00:00",
            command="agent",
            provider="gemini",
            model="gemini-2.0-flash",
            prompt_tokens=200,
            completion_tokens=75,
            estimated_cost_usd=0.0025,
        )
        logger.log(r)
        parsed = json.loads((tmp_path / "usage.jsonl").read_text().strip())
        assert parsed["timestamp"] == "2026-01-15T08:30:00+00:00"
        assert parsed["command"] == "agent"
        assert parsed["provider"] == "gemini"
        assert parsed["model"] == "gemini-2.0-flash"
        assert parsed["prompt_tokens"] == 200
        assert parsed["completion_tokens"] == 75
        assert parsed["estimated_cost_usd"] == 0.0025


class TestUsageLoggerReadAll:

    def test_read_all_returns_empty_list_when_no_file(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        assert logger.read_all() == []

    def test_read_all_roundtrips_fields(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        r = _record(prompt_tokens=123, completion_tokens=45, estimated_cost_usd=0.002)
        logger.log(r)
        records = logger.read_all()
        assert len(records) == 1
        assert records[0].prompt_tokens == 123
        assert records[0].completion_tokens == 45
        assert records[0].estimated_cost_usd == 0.002

    def test_read_all_returns_usage_record_instances(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        logger.log(_record())
        records = logger.read_all()
        assert all(isinstance(r, UsageRecord) for r in records)

    def test_read_all_roundtrips_none_cost(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        logger.log(_record(estimated_cost_usd=None))
        records = logger.read_all()
        assert records[0].estimated_cost_usd is None

    def test_read_all_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "usage.jsonl"
        r = _record()
        path.write_text(f"\n{json.dumps(r.__dict__)}\n\n", encoding="utf-8")
        logger = UsageLogger(tmp_path)
        records = logger.read_all()
        assert len(records) == 1

    def test_read_all_returns_records_in_file_order(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        for i in range(3):
            logger.log(_record(prompt_tokens=i * 10))
        records = logger.read_all()
        assert [r.prompt_tokens for r in records] == [0, 10, 20]

    def test_read_all_multiple_records(self, tmp_path: Path) -> None:
        logger = UsageLogger(tmp_path)
        for cmd in ("rag", "agent", "rag"):
            logger.log(_record(command=cmd))
        records = logger.read_all()
        assert len(records) == 3
        assert [r.command for r in records] == ["rag", "agent", "rag"]


# ── format_session_summary ────────────────────────────────────────────────────

class TestFormatSessionSummary:

    def test_empty_records_returns_empty_string(self) -> None:
        assert format_session_summary([]) == ""

    def test_single_paid_call(self) -> None:
        records = [_record(prompt_tokens=1000, completion_tokens=200, estimated_cost_usd=0.003)]
        summary = format_session_summary(records)
        assert "1 call" in summary
        assert "1,200 tokens" in summary
        assert "$0.003" in summary

    def test_multiple_calls_plural(self) -> None:
        records = [_record(), _record()]
        summary = format_session_summary(records)
        assert "2 calls" in summary

    def test_aggregates_tokens_across_calls(self) -> None:
        records = [
            _record(prompt_tokens=500, completion_tokens=100, estimated_cost_usd=0.001),
            _record(prompt_tokens=600, completion_tokens=200, estimated_cost_usd=0.002),
        ]
        summary = format_session_summary(records)
        assert "1,400 tokens" in summary

    def test_aggregates_cost_across_calls(self) -> None:
        records = [
            _record(prompt_tokens=500, completion_tokens=100, estimated_cost_usd=0.001),
            _record(prompt_tokens=600, completion_tokens=200, estimated_cost_usd=0.002),
        ]
        summary = format_session_summary(records)
        assert "$0.003" in summary

    def test_ollama_shows_local_no_cost(self) -> None:
        records = [_record(provider="ollama", estimated_cost_usd=None, prompt_tokens=800, completion_tokens=200)]
        summary = format_session_summary(records)
        assert "1 call" in summary
        assert "1,000 tokens" in summary
        assert "local (no cost)" in summary

    def test_streaming_shows_unavailable(self) -> None:
        records = [_record(prompt_tokens=0, completion_tokens=0, estimated_cost_usd=None)]
        summary = format_session_summary(records)
        assert "1 call" in summary
        assert "streaming" in summary

    def test_mixed_paid_and_streaming_shows_totals(self) -> None:
        records = [
            _record(prompt_tokens=500, completion_tokens=100, estimated_cost_usd=0.002),
            _record(prompt_tokens=0, completion_tokens=0, estimated_cost_usd=None),
        ]
        summary = format_session_summary(records)
        assert "2 calls" in summary
        # Has tokens from the first call
        assert "600 tokens" in summary

    def test_three_calls_summary(self) -> None:
        records = [_record(prompt_tokens=400, completion_tokens=100, estimated_cost_usd=0.001)] * 3
        summary = format_session_summary(records)
        assert "3 calls" in summary
        assert "1,500 tokens" in summary
        assert "$0.003" in summary
