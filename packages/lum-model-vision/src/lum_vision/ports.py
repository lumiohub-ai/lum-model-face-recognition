"""Ports the host application implements.

Keeping these as protocols is what lets the package stay free of any database
driver: the caller supplies the storage, the package only consumes the data.
"""

from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Source of the known-face embeddings used for recognition.

    The application's pgvector store already satisfies this as-is. Anything
    returning a ``(names, embeddings)`` pair will do — see
    :class:`InMemoryEmbeddingProvider`.
    """

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Return ``(names, embeddings)`` with embeddings shaped (N, dim)."""
        ...


class InMemoryEmbeddingProvider:
    """An :class:`EmbeddingProvider` backed by arrays already in memory.

    Useful for tests, notebooks, and any caller that loads embeddings itself.
    """

    def __init__(self, names: List[str], embeddings: np.ndarray) -> None:
        if len(names) != len(embeddings):
            raise ValueError(
                f"names/embeddings length mismatch: {len(names)} vs {len(embeddings)}"
            )
        self._names = list(names)
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        return list(self._names), self._embeddings
