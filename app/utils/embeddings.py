from functools import lru_cache
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    SentenceTransformer = None  # type: ignore


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    """
    Lazily load and cache the sentence-transformers model.

    This avoids re-loading the model on every request.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install it with: pip install sentence-transformers"
        )
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embedding(text: str) -> np.ndarray:
    """Return a single embedding vector for the given text."""
    model = _get_model()
    # sentence-transformers already returns a numpy array
    return model.encode(text or "", show_progress_bar=False, normalize_embeddings=True)


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Return embeddings for a list of texts as a 2D numpy array."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if a is None or b is None:
        return 0.0
    # Vectors from sentence-transformers are already L2-normalized when
    # normalize_embeddings=True, so dot product is cosine similarity.
    return float(np.dot(a, b))

