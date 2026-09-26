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
