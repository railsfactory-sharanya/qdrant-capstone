"""Run the five test queries through each search method and save results.

Part 2: cosine / Euclidean / dot-product collections.
Later parts will add exact search, HNSW, and IVF in this same script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from embed import QUERY_VECTORS_PATH, load_queries
from qdrant_setup import get_client

RESULTS_DIR = Path("results")
DISTANCE_RESULTS_PATH = RESULTS_DIR / "distance_metrics.json"
SNIPPET_CHARS = 240


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def search_top5(client: QdrantClient, collection_name: str, query_vector: np.ndarray) -> list[dict]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=5,
        with_payload=True,
    )
    rows = []
    for point in response.points:
        payload = point.payload or {}
        rows.append(
            {
                "id": int(point.id),
                "score": float(point.score),
                "category": payload.get("category"),
                "text": snippet(str(payload.get("text", ""))),
            }
        )
    return rows


def ids_of(rows: list[dict]) -> list[int]:
    return [row["id"] for row in rows]


def find_rank_differences(record: dict) -> list[str]:
    """Return metric pairs whose top-5 id order is not identical."""
    cosine = ids_of(record["cosine"])
    euclid = ids_of(record["euclid"])
    dot = ids_of(record["dot"])
    diffs = []
    if cosine != euclid:
        diffs.append("cosine vs euclid")
    if cosine != dot:
        diffs.append("cosine vs dot")
    if euclid != dot:
        diffs.append("euclid vs dot")
    return diffs


def run_distance_metrics(client: QdrantClient) -> list[dict]:
    queries = load_queries()
    query_vectors = np.load(QUERY_VECTORS_PATH)
    if len(queries) != len(query_vectors):
        raise ValueError("queries.md and query_vectors.npy are out of sync — re-run embed.py")

    results = []
    metric_names = {"news_cosine": "cosine", "news_euclid": "euclid", "news_dot": "dot"}

    for index, query in enumerate(queries):
        print(f"\n=== Query {index + 1}: {query} ===")
        record = {"query": query, "cosine": [], "euclid": [], "dot": []}
        for collection, metric in metric_names.items():
            hits = search_top5(client, collection, query_vectors[index])
            record[metric] = hits
            print(f"\n{metric} (higher is closer except euclid, where lower is closer):")
            for rank, hit in enumerate(hits, start=1):
                print(
                    f"  {rank}. id={hit['id']:4d}  score={hit['score']:.4f}  "
                    f"{hit['category']}"
                )
                print(f"      {hit['text'][:120]}")
        diffs = find_rank_differences(record)
        record["rank_order_differs"] = diffs
        if diffs:
            print(f"\nRank order differs: {', '.join(diffs)}")
        else:
            print("\nRank order is identical across cosine, euclid, and dot")
        results.append(record)
    return results


def main() -> None:
    if not QUERY_VECTORS_PATH.exists():
        raise FileNotFoundError("Run python embed.py first")

    client = get_client()
    results = run_distance_metrics(client)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DISTANCE_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {DISTANCE_RESULTS_PATH}")

    print("\n--- Rank-order summary ---")
    any_diff = False
    for record in results:
        diffs = record["rank_order_differs"]
        mark = ", ".join(diffs) if diffs else "same order"
        if diffs:
            any_diff = True
        print(f"  {record['query'][:60]}...  {mark}" if len(record["query"]) > 60 else f"  {record['query']}  {mark}")
    if not any_diff:
        print(
            "\nNo ranking swaps. MiniLM vectors are unit length, so cosine, "
            "dot, and Euclidean rankings are mathematically tied. Write that in comparison.md."
        )


if __name__ == "__main__":
    main()
