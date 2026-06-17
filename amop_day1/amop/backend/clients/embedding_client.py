"""
Local sentence-transformer embedding client.
Model is loaded once and cached in memory for the lifetime of the process.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings()
    logger.info("Loading embedding model", extra={"model": settings.embedding_model})
    return SentenceTransformer(settings.embedding_model)


def encode(texts: list[str]) -> list[list[float]]:
    """Return a list of float vectors, one per input text."""
    model = _load_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
