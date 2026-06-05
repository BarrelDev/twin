from sentence_transformers import SentenceTransformer

from twin.config import AppConfig

_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class Embedder:
    """Sentence-transformers wrapper with nomic-embed-text prefix handling."""

    def __init__(self, model_name: str, dim: int) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document passages.

        Uses the 'search_document' prefix required by nomic-embed-text.
        Batches internally for efficiency.

        Args:
            texts: List of passage strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []
        prefixed = [_DOCUMENT_PREFIX + t for t in texts]
        vectors = self._model.encode(prefixed, batch_size=32, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Uses the 'search_query' prefix required by nomic-embed-text.

        Args:
            query: Query string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        prefixed = _QUERY_PREFIX + query
        vector = self._model.encode(prefixed, show_progress_bar=False)
        return vector.tolist()


def build_embedder(config: AppConfig | None = None) -> Embedder:
    """
    Build an Embedder from AppConfig.

    Args:
        config: AppConfig to use. Loads from environment if None.

    Returns:
        Configured Embedder instance.
    """
    if config is None:
        config = AppConfig.from_env()
    return Embedder(model_name=config.embed_model.value, dim=config.embed_dim)
