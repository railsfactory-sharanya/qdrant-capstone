"""Embed the sampled documents and the five test queries with MiniLM.

On a fresh checkout, run:

    python embed.py

That rebuilds metadata if needed, then writes vectors to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import OUTPUT_PATH, load_documents, print_summary, save_metadata

MODEL_NAME = "all-MiniLM-L6-v2"
QUERIES_PATH = Path("queries.md")
VECTORS_PATH = Path("data/vectors.npy")
QUERY_VECTORS_PATH = Path("data/query_vectors.npy")


def load_or_build_metadata(path: Path = OUTPUT_PATH) -> list[dict]:
    """Reuse data/metadata.json, or run the Part 1 sampler if it is missing."""
    if path.exists():
        documents = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded {len(documents)} documents from {path}")
        return documents

    print(f"{path} not found — sampling 20 Newsgroups")
    documents = load_documents()
    save_metadata(documents, path)
    print_summary(documents)
    return documents


def load_queries(path: Path = QUERIES_PATH) -> list[str]:
    """Read numbered queries from queries.md (lines like '1. some text')."""
    queries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line[0].isdigit() or ". " not in line:
            continue
        queries.append(line.split(". ", 1)[1])
    if len(queries) != 5:
        raise ValueError(f"Expected 5 queries in {path}, found {len(queries)}")
    return queries


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def print_vector_stats(name: str, vectors: np.ndarray) -> None:
    first_norm = float(np.linalg.norm(vectors[0]))
    print(f"{name}: shape={vectors.shape} dtype={vectors.dtype} first_norm={first_norm:.4f}")


if __name__ == "__main__":
    documents = load_or_build_metadata()
    queries = load_queries()

    print(f"Loading model {MODEL_NAME} (first run downloads it into ~/.cache)")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(documents)} documents")
    vectors = embed_texts([doc["text"] for doc in documents], model)
    print(f"Embedding {len(queries)} queries")
    query_vectors = embed_texts(queries, model)

    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    np.save(QUERY_VECTORS_PATH, query_vectors)

    print_vector_stats("documents", vectors)
    print_vector_stats("queries", query_vectors)
    print("Queries:")
    for i, query in enumerate(queries):
        print(f"  {i}: {query}")
    print(f"Wrote {VECTORS_PATH} and {QUERY_VECTORS_PATH}")
