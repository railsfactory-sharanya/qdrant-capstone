"""Load and sample the 20 Newsgroups dataset for later embedding."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

SAMPLE_SIZE = 6000
RANDOM_SEED = 42
MIN_CHARS = 40
OUTPUT_PATH = Path("data/metadata.json")


def load_documents(
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    min_chars: int = MIN_CHARS,
) -> list[dict]:
    """Download 20 Newsgroups, clean, sample, and return id/text/category rows."""
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    cleaned: list[tuple[str, str]] = []
    for text, label_index in zip(dataset.data, dataset.target):
        text = text.strip()
        if len(text) < min_chars:
            continue
        category = dataset.target_names[label_index]
        cleaned.append((text, category))

    if sample_size > len(cleaned):
        raise ValueError(
            f"Need {sample_size} documents but only {len(cleaned)} remain after cleaning"
        )

    rng = random.Random(seed)
    sampled = rng.sample(cleaned, sample_size)

    documents = []
    for doc_id, (text, category) in enumerate(sampled):
        documents.append({"id": doc_id, "text": text, "category": category})
    return documents


def save_metadata(documents: list[dict], path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(documents, indent=2), encoding="utf-8")
    return path


def print_summary(documents: list[dict]) -> None:
    counts = Counter(doc["category"] for doc in documents)
    print(f"Saved {len(documents)} documents")
    print("Category counts:")
    for category, count in sorted(counts.items()):
        print(f"  {count:4d}  {category}")


if __name__ == "__main__":
    documents = load_documents()
    output = save_metadata(documents)
    print_summary(documents)
    print(f"Wrote {output}")
