# Qdrant Capstone: Compare Search Algorithms

Compare cosine / Euclidean / dot-product distance, HNSW vs exact search, and a from-scratch IVF-style index on a locally embedded 20 Newsgroups collection.

## Prerequisites

- Python 3.10+
- Docker Desktop running

## Part 0: Run Qdrant locally

From this directory, one command:

```bash
docker compose up -d
```

That pulls `qdrant/qdrant` if needed and starts the container, equivalent to:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

Use **either** `docker compose up -d` **or** the `docker pull` / `docker run` pair, not both.

Confirm it is up: open [http://localhost:6333/dashboard](http://localhost:6333/dashboard). You should see the Qdrant web UI with no collections yet.

The `./qdrant_storage` volume persists data across restarts. Stop with `docker compose down`.

## Python environment

Create and use a virtualenv **in this folder**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv/` is gitignored. Recreate it the same way on a fresh checkout.
