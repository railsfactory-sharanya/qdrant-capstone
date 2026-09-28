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

## Part 4: IVF-style index

Qdrant has no IVF toggle, so `ivf.py` builds one: KMeans with **64**
clusters, each centroid plus the document ids assigned to it (inverted
lists). Search (`search_ivf`): take the `nprobe` nearest centroids, then
exact cosine only on those lists, return top-5. Cosine matches the
Qdrant exact collection (MiniLM unit vectors). Full numbers:
`results/ivf.json`.

| Method | mean overlap@5 vs exact | mean latency (ms) |
|--------|-------------------------|-------------------|
| IVF nprobe=1 | 0.48 | 0.40 |
| IVF nprobe=8 | 1.00 | 0.97 |

`nprobe=1` missed true neighbors on four of five queries (overlaps
0.20, 0.20, 0.40, 0.60, 1.00). Example: *how public-key encryption and
cryptography work* — exact top hit is id 1898 (`sci.crypt` public-key
FAQ). At `nprobe=1` that id never appears (it lives in an un-probed
list); at `nprobe=8` it is rank 1 again and overlap is 1.00.

Raising `nprobe` scans more inverted lists, so recall moves toward
brute force and latency goes up (0.40 ms → 0.97 ms). Same speed /
accuracy idea as raising HNSW `ef`, different data structure. IVF does
not appear in the Qdrant dashboard — it is in-process Python.

## Part 5: Combined comparison

All numbers below are means over the same five queries. Overlap is
versus Qdrant cosine exact search (`exact: true`). Latency is one timed
search call per query (local REST; noisy). Sources: `results/hnsw.json`,
`results/ivf.json`.

| Method | mean overlap@5 vs exact | mean latency (ms) |
|--------|-------------------------|-------------------|
| exact brute-force | 1.00 | 4.22 |
| HNSW default ef=16 | 1.00 | 3.17 |
| HNSW default ef=64 | 1.00 | 2.50 |
| HNSW default ef=128 | 1.00 | 2.39 |
| HNSW untuned ef=16 | 1.00 | 7.30 |
| HNSW untuned ef=64 | 1.00 | 3.04 |
| HNSW untuned ef=128 | 1.00 | 2.77 |
| IVF nprobe=1 | 0.48 | 0.40 |
| IVF nprobe=8 | 1.00 | 0.97 |

### Speed / accuracy as we gave the index more work

Approximate indexes skip vectors on purpose. HNSW skips nodes it does
not visit on the graph; IVF skips whole inverted lists. Giving the
method more work means walking more of the graph (higher search `ef`)
or scanning more clusters (higher `nprobe`). In the indexing unit that
is the usual recall-vs-latency tradeoff: more candidates → closer to
brute force → slower.

On this 6,000-vector collection the two families did not behave the
same. Every HNSW configuration already matched exact top-5 (overlap
1.00), including the under-tuned graph (`m=4`) at `ef=16`. Extra `ef`
therefore had no overlap left to improve. Latency also did not show a
clean “higher `ef` = slower” line; one HTTP call on localhost jittered
by a few milliseconds. That is a size finding: a small cosine set is
easy for even a thin HNSW graph.

IVF is where the tradeoff showed up. With `nprobe=1` we only exact-scan
one KMeans list, so the true neighbor can sit in an un-probed cluster.
Mean overlap was 0.48 (four of five queries missed at least one exact
id). Raising `nprobe` to 8 added seven more lists; overlap went to 1.00
and mean latency from 0.40 ms to 0.97 ms. The crypto query is the
clearest case: exact top-1 id 1898 is absent at `nprobe=1` and rank 1
again at `nprobe=8`.

So the method that got closer to exact as we gave it more work was
**IVF**. That matches the unit: more inverted lists examined, more of
the dataset scored, closer to a full scan. HNSW would show the same
curve on a larger or harder graph; here it was already at the top of
the curve.

### Metric choice vs index choice (Part 2)

The spec asks for a Part 2 case where rank order changed with the
distance metric while the index and dataset stayed identical. We did
not observe that. Cosine, Euclidean, and dot product returned the same
top-5 ids on every query because MiniLM vectors are unit length, so
`dot = cosine` and Euclidean distance is a monotone function of cosine.
That is still the right place to separate the two decisions: in Part 2
we turned only the **metric** knob and ranking did not move; in Part 4
we turned only the **index** knob (`nprobe`) on the same cosine vectors
and overlap moved from 0.48 to 1.00. Metric and index are independent.
If a future model left vectors unnormalized, the Part 2 knob could
change top-1 even with this same HNSW setup.
