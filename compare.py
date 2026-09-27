"""Run the five test queries through each search method and save results.

Part 2: cosine / Euclidean / dot-product collections.
Part 3: exact search vs default HNSW vs under-tuned HNSW at several ef values.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from embed import QUERY_VECTORS_PATH, load_queries
from qdrant_setup import HNSW_DEFAULT_COLLECTION, HNSW_UNTUNED_COLLECTION, get_client

RESULTS_DIR = Path("results")
DISTANCE_RESULTS_PATH = RESULTS_DIR / "distance_metrics.json"
HNSW_RESULTS_PATH = RESULTS_DIR / "hnsw.json"
SNIPPET_CHARS = 240
EF_VALUES = (16, 64, 128)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def search_top5(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    search_params: models.SearchParams | None = None,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=5,
        with_payload=True,
        search_params=search_params,
    )
    latency_ms = (time.perf_counter() - started) * 1000
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
    return rows, latency_ms


def ids_of(rows: list[dict]) -> list[int]:
    return [row["id"] for row in rows]


def overlap_at_5(approx_ids: list[int], exact_ids: list[int]) -> float:
    return len(set(approx_ids) & set(exact_ids)) / 5


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


def print_hits(title: str, hits: list[dict], latency_ms: float | None = None) -> None:
    extra = f"  ({latency_ms:.2f} ms)" if latency_ms is not None else ""
    print(f"\n{title}{extra}:")
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. id={hit['id']:4d}  score={hit['score']:.4f}  {hit['category']}")
        print(f"      {hit['text'][:120]}")


def run_distance_metrics(client: QdrantClient, queries: list[str], query_vectors: np.ndarray) -> list[dict]:
    results = []
    metric_names = {"news_cosine": "cosine", "news_euclid": "euclid", "news_dot": "dot"}

    for index, query in enumerate(queries):
        print(f"\n=== Query {index + 1}: {query} ===")
        record = {"query": query, "cosine": [], "euclid": [], "dot": []}
        for collection, metric in metric_names.items():
            hits, _latency = search_top5(client, collection, query_vectors[index])
            record[metric] = hits
            print_hits(f"{metric} (higher is closer except euclid, where lower is closer)", hits)
        diffs = find_rank_differences(record)
        record["rank_order_differs"] = diffs
        if diffs:
            print(f"\nRank order differs: {', '.join(diffs)}")
        else:
            print("\nRank order is identical across cosine, euclid, and dot")
        results.append(record)
    return results


def hnsw_configs() -> list[dict]:
    configs = [
        {
            "name": "exact",
            "collection": HNSW_DEFAULT_COLLECTION,
            "params": models.SearchParams(exact=True),
        }
    ]
    for label, collection in (
        ("hnsw_default", HNSW_DEFAULT_COLLECTION),
        ("hnsw_untuned", HNSW_UNTUNED_COLLECTION),
    ):
        for ef in EF_VALUES:
            configs.append(
                {
                    "name": f"{label}_ef{ef}",
                    "collection": collection,
                    "params": models.SearchParams(hnsw_ef=ef),
                }
            )
    return configs


def run_hnsw(client: QdrantClient, queries: list[str], query_vectors: np.ndarray) -> dict:
    warmup_vector = query_vectors[0]
    search_top5(
        client,
        HNSW_DEFAULT_COLLECTION,
        warmup_vector,
        search_params=models.SearchParams(exact=True),
    )

    exact_ids_by_query: list[list[int]] = []
    blocks = []

    for config in hnsw_configs():
        per_query = []
        print(f"\n===== {config['name']} ({config['collection']}) =====")
        for index, query in enumerate(queries):
            hits, latency_ms = search_top5(
                client,
                config["collection"],
                query_vectors[index],
                search_params=config["params"],
            )
            ids = ids_of(hits)
            if config["name"] == "exact":
                exact_ids_by_query.append(ids)
                overlap = 1.0
            else:
                overlap = overlap_at_5(ids, exact_ids_by_query[index])
            print_hits(f"{query}  overlap={overlap:.2f}", hits, latency_ms)
            per_query.append(
                {
                    "query": query,
                    "ids": ids,
                    "scores": [hit["score"] for hit in hits],
                    "categories": [hit["category"] for hit in hits],
                    "latency_ms": round(latency_ms, 3),
                    "overlap": overlap,
                }
            )
        mean_overlap = sum(row["overlap"] for row in per_query) / len(per_query)
        mean_latency = sum(row["latency_ms"] for row in per_query) / len(per_query)
        print(f"mean overlap={mean_overlap:.2f}  mean latency={mean_latency:.2f} ms")
        blocks.append(
            {
                "name": config["name"],
                "collection": config["collection"],
                "mean_overlap": round(mean_overlap, 4),
                "mean_latency_ms": round(mean_latency, 3),
                "per_query": per_query,
            }
        )

    return {"configs": blocks}


def main() -> None:
    if not QUERY_VECTORS_PATH.exists():
        raise FileNotFoundError("Run python embed.py first")

    queries = load_queries()
    query_vectors = np.load(QUERY_VECTORS_PATH)
    if len(queries) != len(query_vectors):
        raise ValueError("queries.md and query_vectors.npy are out of sync — re-run embed.py")

    client = get_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n######## Part 2: distance metrics ########")
    distance_results = run_distance_metrics(client, queries, query_vectors)
    DISTANCE_RESULTS_PATH.write_text(json.dumps(distance_results, indent=2), encoding="utf-8")
    print(f"\nWrote {DISTANCE_RESULTS_PATH}")

    print("\n######## Part 3: exact vs HNSW ########")
    hnsw_results = run_hnsw(client, queries, query_vectors)
    HNSW_RESULTS_PATH.write_text(json.dumps(hnsw_results, indent=2), encoding="utf-8")
    print(f"\nWrote {HNSW_RESULTS_PATH}")

    print("\n--- HNSW summary (mean over 5 queries) ---")
    print(f"{'config':<22} {'overlap@5':>10} {'latency_ms':>12}")
    for block in hnsw_results["configs"]:
        print(f"{block['name']:<22} {block['mean_overlap']:>10.2f} {block['mean_latency_ms']:>12.2f}")


if __name__ == "__main__":
    main()
