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

        Falls back to sensible defaults when variables are absent.

        Returns:
            AppConfig populated from environment.
        """
        return cls(
            data_dir=Path(os.environ.get("SECONDBRAIN_DATA_DIR", _DEFAULT_DATA_DIR)),
            embed_model=EmbeddingModel(
                os.environ.get("SECONDBRAIN_EMBED_MODEL", EmbeddingModel.NOMIC.value)
            ),
            chunk_tokens=int(os.environ.get("SECONDBRAIN_CHUNK_TOKENS", "512")),
            overlap_tokens=int(os.environ.get("SECONDBRAIN_OVERLAP", "64")),
            top_k=int(os.environ.get("SECONDBRAIN_TOP_K", "5")),
        )