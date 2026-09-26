"""Create Qdrant collections and upsert the Part 1 vectors.

Part 2: three collections, same points, different distance metrics.
Part 3 will add HNSW variants in this same file later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from data import OUTPUT_PATH
from embed import VECTORS_PATH

QDRANT_URL = "http://localhost:6333"
VECTOR_SIZE = 384
UPSERT_BATCH = 256

METRIC_COLLECTIONS = {
    "news_cosine": models.Distance.COSINE,
    "news_euclid": models.Distance.EUCLID,
    "news_dot": models.Distance.DOT,
}


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def load_points() -> tuple[list[dict], np.ndarray]:
    if not OUTPUT_PATH.exists() or not VECTORS_PATH.exists():
        raise FileNotFoundError("Run python embed.py first so data/metadata.json and data/vectors.npy exist")
    documents = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    vectors = np.load(VECTORS_PATH)
    if len(documents) != len(vectors):
        raise ValueError("metadata.json and vectors.npy are out of sync — re-run embed.py")
    return documents, vectors


def recreate_metric_collections(client: QdrantClient) -> None:
    for name, distance in METRIC_COLLECTIONS.items():
        print(f"Creating collection {name} ({distance})")
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=distance),
        )


def upsert_documents(
    client: QdrantClient,
    collection_name: str,
    documents: list[dict],
    vectors: np.ndarray,
) -> None:
    """Write the same (id, vector, payload) into one collection, in batches."""
    total = len(documents)
    for start in range(0, total, UPSERT_BATCH):
        batch_docs = documents[start : start + UPSERT_BATCH]
        batch_vecs = vectors[start : start + UPSERT_BATCH]
        points = [
            models.PointStruct(
                id=doc["id"],
                vector=vec.tolist(),
                payload={"category": doc["category"], "text": doc["text"]},
            )
            for doc, vec in zip(batch_docs, batch_vecs)
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"  {collection_name}: upserted {min(start + UPSERT_BATCH, total)}/{total}")


def main() -> None:
    documents, vectors = load_points()
    print(f"Loaded {len(documents)} documents, vectors {vectors.shape}")
    print(f"First vector L2 norm: {float(np.linalg.norm(vectors[0])):.4f}")

    client = get_client()
    recreate_metric_collections(client)
    for name in METRIC_COLLECTIONS:
        upsert_documents(client, name, documents, vectors)
        info = client.get_collection(name)
        print(f"{name}: {info.points_count} points")


if __name__ == "__main__":
    main()
