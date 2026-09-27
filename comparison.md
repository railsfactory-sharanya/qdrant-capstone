# Comparison notes

## Part 2: Distance metrics

Setup: 6,000 MiniLM embeddings (384-d), three Qdrant collections
(`news_cosine`, `news_euclid`, `news_dot`) with **identical** point ids,
vectors, and payload (`category`, `text`). Only the distance setting
changes. Five queries from `queries.md`. Full top-5 tables:
`results/distance_metrics.json`.

### What each metric measures

| Metric | Formula idea | Qdrant score | Better means |
|--------|----------------|--------------|--------------|
| Cosine | angle only: `(a·b) / (‖a‖ ‖b‖)` | similarity | higher (max 1) |
| Dot product | alignment **and** length: `a·b = ‖a‖ ‖b‖ cos(θ)` | similarity | higher |
| Euclidean | straight-line distance `‖a−b‖` | distance | **lower** |

Do not compare raw numbers across metrics. Compare **id order**.

### Did the ranking change?

On all five queries, cosine, Euclidean, and dot product returned the
**same top-5 ids in the same order**.

That is not a bug. Part 1 measured `‖v‖ = 1.0000` for MiniLM output.
When every vector is unit length:

- `a·b = cos(θ)` so **cosine and dot product are the same ranking**
- `‖a−b‖² = 2 − 2 (a·b)` so **Euclidean order is the same ranking too**
  (smaller distance = larger cosine)

Check on query 1, rank 1 (id 982): cosine score `0.4384`, Euclidean
score `1.0598`.  
`sqrt(2 - 2×0.4384) ≈ 1.0598`. The two scores are the same geometry,
different units.

**When cosine vs dot would change the top hit**,
even with identical stored vectors: if a document were *longer* in
vector space (‖v‖ > 1), dot product would boost it and cosine would
ignore the extra length. Our model already divided length out, so we
did not see that swap. Metric choice and index choice are still
separate decisions — here the index is the same (Qdrant default HNSW)
and the metric knob did not move rank because of unit norms.

Example that *did* retrieve the right neighborhood (same under all
three metrics): query *how public-key encryption and cryptography work*
→ top hit id 1898, `sci.crypt`, a public-key FAQ. Query *who is likely
to win the hockey playoffs this season* → all top-5 are
`rec.sport.hockey`.

## Part 3: HNSW vs exact search

Same 6,000 cosine vectors. Ground truth is Qdrant search with
`exact: true` on `news_cosine`. Default HNSW is that same collection
(`m=16`, `ef_construct=100`). Under-tuned HNSW is `news_hnsw_untuned`
(`m=4`, `ef_construct=16`). Search-time `ef` is 16, 64, and 128.
`full_scan_threshold` is 20 on both so Qdrant actually uses the graph
(its default threshold is 10,000, above our collection size).

Overlap@5 = share of the exact top-5 ids that the method also returned
(order does not matter). Latency is one timed `query_points` call per
query, then averaged. Full numbers: `results/hnsw.json`.

| Method | mean overlap@5 vs exact | mean latency (ms) |
|--------|-------------------------|-------------------|
| exact brute-force | 1.00 | 4.22 |
| HNSW default ef=16 | 1.00 | 3.17 |
| HNSW default ef=64 | 1.00 | 2.50 |
| HNSW default ef=128 | 1.00 | 2.39 |
| HNSW untuned ef=16 | 1.00 | 7.30 |
| HNSW untuned ef=64 | 1.00 | 3.04 |
| HNSW untuned ef=128 | 1.00 | 2.77 |

Every HNSW config matched exact top-5 on all five queries. 6,000 points
is small: even a thin graph (`m=4`) still reached the true neighbors.
Latency on this laptop is noisy (local REST, one shot after a short
warmup) and does not show a clean “higher `ef` = slower” line.

What *would* make the under-tuned collection behave more like default:
raise **`m`** (more neighbors per node) and/or **`ef_construct`**
(better graph at build time). Raising search `ef` explores more of a
weak graph and can help, but it does not rebuild connectivity. On this
dataset we did not need that help.
