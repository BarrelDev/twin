from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timezone

@dataclass
class UsageRecord:
    """One record per LLM call appended to ~/.twin/usage.jsonl."""
    timestamp: str          # ISO 8601
    command: str            # "rag" | "agent"
    provider: str           # "anthropic" | "openai" | ...
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float | None  # None for Ollama

class UsageLogger:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "usage.jsonl"
    
    def log(self, record: UsageRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__) + "\n")
    
    def read_all(self) -> list[UsageRecord]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text().splitlines():
            if line.strip():
                records.append(UsageRecord(**json.loads(line)))
        return records