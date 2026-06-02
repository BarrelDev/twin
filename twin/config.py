import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_DEFAULT_DATA_DIR = str(Path.home() / ".twin")

_EMBEDDING_DIMS: dict[str, int] = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


class EmbeddingModel(str, Enum):
    NOMIC = "nomic-ai/nomic-embed-text-v1.5"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


@dataclass
class ModelInfo:
    """Metadata about a model available from a provider."""

    model_id: str
    name: str
    supports_tools: bool


@dataclass
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    data_dir: Path
    embed_model: EmbeddingModel
    chunk_tokens: int
    overlap_tokens: int
    top_k: int

    @property
    def embed_dim(self) -> int:
        """Dimensionality of the configured embedding model."""
        return _EMBEDDING_DIMS[self.embed_model.value]

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Load configuration from environment variables.

        Reads TWIN_* variables, falling back to legacy SECONDBRAIN_* names,
        then to hardcoded defaults.

        Returns:
            AppConfig populated from environment.
        """
        def _get(twin_key: str, legacy_key: str, default: str) -> str:
            return (
                os.environ.get(twin_key)
                or os.environ.get(legacy_key)
                or default
            )

        return cls(
            data_dir=Path(
                _get("TWIN_DATA_DIR", "SECONDBRAIN_DATA_DIR", _DEFAULT_DATA_DIR)
            ),
            embed_model=EmbeddingModel(
                _get("TWIN_EMBED_MODEL", "SECONDBRAIN_EMBED_MODEL", EmbeddingModel.NOMIC.value)
            ),
            chunk_tokens=int(_get("TWIN_CHUNK_TOKENS", "SECONDBRAIN_CHUNK_TOKENS", "512")),
            overlap_tokens=int(_get("TWIN_OVERLAP", "SECONDBRAIN_OVERLAP", "64")),
            top_k=int(_get("TWIN_TOP_K", "SECONDBRAIN_TOP_K", "5")),
        )