from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path("dataset")
MAX_QUERIES = 600
MAX_PASSAGES_PER_QUERY = 4
HIDDEN_RATIO = 0.3
SEED = 42


def text_to_doc_id(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def build_splits(
    max_queries: int = MAX_QUERIES,
    max_passages_per_query: int = MAX_PASSAGES_PER_QUERY,
    hidden_ratio: float = HIDDEN_RATIO,
    seed: int = SEED,
):
    ds = load_dataset("ms_marco", "v1.1", split="train")
    rnd = random.Random(seed)
    sample_size = min(max_queries, len(ds))
    indices = rnd.sample(range(len(ds)), sample_size)
    rows = ds.select(indices)

    corpus = {}
    all_queries = []

    for i, item in enumerate(rows):
        question = str(item.get("query", "")).strip()
        passages = item.get("passages") or {}
        passage_texts = passages.get("passage_text") or []
        selected_flags = passages.get("is_selected") or []
        if not question or not passage_texts:
            continue

        # Keep a bounded subset of passages per query to control corpus size and runtime.
        pairs: list[tuple[str, int]] = []
        for idx, text in enumerate(passage_texts[:max_passages_per_query]):
            clean_text = str(text).strip()
            if not clean_text:
                continue
            flag = int(selected_flags[idx]) if idx < len(selected_flags) else 0
            pairs.append((clean_text, flag))
        if not pairs:
            continue

        positive_doc_ids: list[str] = []
        for passage_text, flag in pairs:
            doc_id = text_to_doc_id(passage_text)
            corpus[doc_id] = passage_text
            if flag > 0:
                positive_doc_ids.append(doc_id)

        # Some rows may not have an explicit selected passage; keep one weak positive to stay trainable.
        if not positive_doc_ids:
            fallback_doc_id = text_to_doc_id(pairs[0][0])
            positive_doc_ids.append(fallback_doc_id)

        qid = f"q_{i:05d}"
        all_queries.append((qid, question, positive_doc_ids))

    rnd.shuffle(all_queries)
    hidden_count = max(1, int(len(all_queries) * hidden_ratio))
    hidden_part = all_queries[:hidden_count]
    public_part = all_queries[hidden_count:]

    queries_public = {}
    qrels_public = {}
    queries_hidden = {}
    qrels_hidden = {}

    for qid, question, positive_doc_ids in public_part:
        queries_public[qid] = question
        qrels_public[qid] = {doc_id: 1 for doc_id in positive_doc_ids}

    for qid, question, positive_doc_ids in hidden_part:
        queries_hidden[qid] = question
        qrels_hidden[qid] = {doc_id: 1 for doc_id in positive_doc_ids}

    return corpus, queries_public, qrels_public, queries_hidden, qrels_hidden


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    parser.add_argument("--max-passages-per-query", type=int, default=MAX_PASSAGES_PER_QUERY)
    parser.add_argument("--hidden-ratio", type=float, default=HIDDEN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    corpus, queries_public, qrels_public, queries_hidden, qrels_hidden = build_splits(
        max_queries=args.max_queries,
        max_passages_per_query=args.max_passages_per_query,
        hidden_ratio=args.hidden_ratio,
        seed=args.seed,
    )

    write_json(DATA_DIR / "corpus.json", corpus)
    write_json(DATA_DIR / "queries_public.json", queries_public)
    write_json(DATA_DIR / "qrels_public.json", qrels_public)
    write_json(DATA_DIR / "queries_hidden.json", queries_hidden)
    write_json(DATA_DIR / "qrels_hidden.json", qrels_hidden)

    print(
        json.dumps(
            {
                "data_dir": str(DATA_DIR),
                "corpus_docs": len(corpus),
                "queries_public": len(queries_public),
                "queries_hidden": len(queries_hidden),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
