"""Cost and token tracking for Twin LLM calls."""

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timezone


@dataclass
class UsageRecord:
    """One record per LLM call appended to ~/.twin/usage.jsonl."""

    timestamp: str
    """ISO 8601 timestamp of the call."""

    command: str
    """CLI command that triggered the call: 'rag' or 'agent'."""

    provider: str
    """Provider name: 'anthropic', 'openai', 'gemini', 'ollama', 'openrouter'."""

    model: str
    """Model identifier used for the call."""

    prompt_tokens: int
    """Number of prompt (input) tokens. 0 for streaming calls where counts are unavailable."""

    completion_tokens: int
    """Number of completion (output) tokens. 0 for streaming calls."""

    estimated_cost_usd: float | None
    """Estimated cost in USD. None for Ollama (local) or streaming calls."""


class UsageLogger:
    """Appends UsageRecord entries to a JSONL file and reads them back."""

    def __init__(self, data_dir: Path) -> None:
        """
        Initialize the logger.

        Args:
            data_dir: Directory where usage.jsonl lives (usually ~/.twin).
        """
        self._path = data_dir / "usage.jsonl"

    def log(self, record: UsageRecord) -> None:
        """
        Append one record to the JSONL file.

        Args:
            record: Usage record to persist.
        """
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    def read_all(self) -> list[UsageRecord]:
        """
        Read all records from the JSONL file.

        Returns:
            List of UsageRecord in file order; empty list if file does not exist.
        """
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(UsageRecord(**json.loads(line)))
        return records


def format_session_summary(records: list[UsageRecord]) -> str:
    """
    Format a one-line session summary for display at the end of a CLI command.

    Examples::

        "3 calls · 1,240 tokens · ~$0.003"
        "2 calls · 890 tokens · local (no cost)"
        "1 call · (streaming, usage unavailable)"

    Args:
        records: Usage records accumulated during the session.

    Returns:
        Formatted summary string, or empty string if records is empty.
    """
    if not records:
        return ""

    n = len(records)
    calls_str = f"{n} call{'s' if n != 1 else ''}"
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in records)
    costs = [r.estimated_cost_usd for r in records if r.estimated_cost_usd is not None]

    if total_tokens == 0 and not costs:
        return f"{calls_str} · (streaming, usage unavailable)"

    tokens_str = f"{total_tokens:,} tokens"

    if not costs:
        return f"{calls_str} · {tokens_str} · local (no cost)"

    return f"{calls_str} · {tokens_str} · ~${sum(costs):.3f}"
