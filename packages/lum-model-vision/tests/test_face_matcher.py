"""FaceMatcher works against any embedding provider — no database involved."""

import numpy as np
import pytest

from lum_vision import FaceMatcher, InMemoryEmbeddingProvider


@pytest.fixture
def matcher():
    """Three orthogonal unit embeddings, one per identity."""
    names = ["alice", "bob", "carol"]
    embs = np.eye(3, dtype=np.float32)
    return FaceMatcher(InMemoryEmbeddingProvider(names, embs), match_threshold=0.3)


def test_best_match_picks_nearest_identity(matcher):
    query = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)
    idx, score = matcher.get_best_match(matcher.compute_similarities(query))

    assert matcher.db_names[idx] == "alice"
    assert score == pytest.approx(0.9939, abs=1e-3)


def test_similarities_shape_matches_query_and_db(matcher):
    query = np.random.rand(4, 3).astype(np.float32)
    assert matcher.compute_similarities(query).shape == (4, 3)


def test_query_is_normalized_so_magnitude_does_not_change_ranking(matcher):
    small = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
    large = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)

    assert matcher.compute_similarities(small) == pytest.approx(
        matcher.compute_similarities(large)
    )


def test_zero_vector_does_not_divide_by_zero(matcher):
    sims = matcher.compute_similarities(np.zeros((1, 3), dtype=np.float32))
    assert np.all(np.isfinite(sims))


def test_empty_database_returns_empty_similarity_matrix():
    matcher = FaceMatcher(InMemoryEmbeddingProvider([], np.empty((0, 3), dtype=np.float32)))
    assert matcher.compute_similarities(np.zeros((2, 3), dtype=np.float32)).shape == (2, 0)


def test_name_index_is_built():
    names = ["alice", "bob"]
    matcher = FaceMatcher(InMemoryEmbeddingProvider(names, np.eye(2, dtype=np.float32)))
    assert matcher.name_to_index == {"alice": 0, "bob": 1}


def test_reload_picks_up_provider_changes():
    class GrowingProvider:
        def __init__(self):
            self.calls = 0

        def get_all_embeddings(self):
            self.calls += 1
            n = self.calls
            return [f"user{i}" for i in range(n)], np.eye(3, dtype=np.float32)[:n]

    matcher = FaceMatcher(GrowingProvider())
    assert len(matcher.db_names) == 1

    matcher.reload_embeddings()
    assert len(matcher.db_names) == 2
    assert matcher.name_to_index == {"user0": 0, "user1": 1}


def test_provider_missing_the_protocol_method_is_rejected():
    with pytest.raises(TypeError, match="get_all_embeddings"):
        FaceMatcher(object())


def test_in_memory_provider_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        InMemoryEmbeddingProvider(["a", "b"], np.eye(3, dtype=np.float32))
