"""A minimal IVF-style index: KMeans centroids + inverted lists.

Qdrant has no IVF option for dense vectors, so we build this IVF-style
index ourselves and compare it to Qdrant's HNSW.

    python ivf.py          # build the index and print cluster sizes
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from embed import VECTORS_PATH

N_CLUSTERS = 64
RANDOM_SEED = 42
NPROBE_VALUES = (1, 8)


def build_ivf_index(
    vectors: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Cluster vectors and group document ids by nearest centroid."""
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(vectors)
    inverted_lists: dict[int, list[int]] = defaultdict(list)
    for doc_id, label in enumerate(labels):
        inverted_lists[int(label)].append(int(doc_id))
    return {
        "centroids": kmeans.cluster_centers_.astype(np.float32),
        "inverted_lists": dict(inverted_lists),
        "vectors": vectors.astype(np.float32),
        "labels": labels.astype(int),
        "n_clusters": n_clusters,
    }


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity; matches our Qdrant cosine collection (unit MiniLM)."""
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query
    return dots / np.clip(query_norm * row_norms, 1e-12, None)


def search_ivf(query_vector: np.ndarray, nprobe: int, k: int, index: dict) -> list[tuple[int, float]]:
    """Find nprobe nearest centroids, exact-scan those lists, return top-k."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    centroid_scores = _cosine_scores(query, index["centroids"])
    probed = np.argsort(-centroid_scores)[:nprobe]

    candidate_ids: list[int] = []
    for cluster_id in probed:
        candidate_ids.extend(index["inverted_lists"].get(int(cluster_id), []))

    if not candidate_ids:
        return []

    candidate_ids = np.asarray(candidate_ids, dtype=int)
    candidate_vectors = index["vectors"][candidate_ids]
    scores = _cosine_scores(query, candidate_vectors)
    top = np.argsort(-scores)[:k]
    return [(int(candidate_ids[i]), float(scores[i])) for i in top]


def print_index_summary(index: dict) -> None:
    sizes = [len(index["inverted_lists"].get(i, [])) for i in range(index["n_clusters"])]
    print(f"IVF clusters: {index['n_clusters']}")
    print(f"  list sizes: min={min(sizes)}  max={max(sizes)}  mean={sum(sizes) / len(sizes):.1f}")


if __name__ == "__main__":
    vectors = np.load(VECTORS_PATH)
    index = build_ivf_index(vectors)
    print_index_summary(index)
    print(f"Loaded {len(vectors)} vectors from {VECTORS_PATH}")
