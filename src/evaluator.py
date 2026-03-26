from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace.baseline import TfidfBaselineRetriever

DATA_DIR = Path("dataset")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mrr_at_10(qrels: dict[str, dict[str, int]], ranking: dict[str, list[str]]) -> float:
    if not qrels:
        return 0.0
    total = 0.0
    for qid, rel_docs in qrels.items():
        rr = 0.0
        for rank, doc_id in enumerate(ranking.get(qid, [])[:10], start=1):
            if rel_docs.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(qrels)


def ndcg_at_10(qrels: dict[str, dict[str, int]], ranking: dict[str, list[str]]) -> float:
    if not qrels:
        return 0.0
    total = 0.0
    for qid, rel_docs in qrels.items():
        docs = ranking.get(qid, [])[:10]
        dcg = 0.0
        for rank, doc_id in enumerate(docs, start=1):
            rel = int(rel_docs.get(doc_id, 0))
            dcg += (2**rel - 1) / math.log2(rank + 1)
        ideal_labels = sorted((int(x) for x in rel_docs.values()), reverse=True)[:10]
        idcg = 0.0
        for rank, rel in enumerate(ideal_labels, start=1):
            idcg += (2**rel - 1) / math.log2(rank + 1)
        total += 0.0 if idcg == 0 else dcg / idcg
    return total / len(qrels)


def score_metrics(qrels: dict[str, dict[str, int]], ranking: dict[str, list[str]]) -> dict:
    return {"mrr@10": mrr_at_10(qrels, ranking), "ndcg@10": ndcg_at_10(qrels, ranking)}


def parse_solution(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def has_function_signature(tree: ast.Module, fn_name: str) -> bool:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            args = [a.arg for a in node.args.args]
            return args == ["query", "top_k", "corpus"]
    return False


def load_module(solution_path: Path):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pipeline(fn, corpus: dict, queries: dict, top_k: int) -> tuple[dict, list[str]]:
    ranking = {}
    errors = []
    corpus_ids = set(corpus.keys())
    for qid, text in queries.items():
        try:
            doc_ids = fn(text, top_k, corpus)
        except Exception as e:
            errors.append(f"runtime error on {qid}: {e}")
            continue
        if not isinstance(doc_ids, list):
            errors.append(f"{qid}: return type must be list[str]")
            continue
        seen = set()
        out = []
        for doc_id in doc_ids:
            if not isinstance(doc_id, str):
                continue
            if doc_id not in corpus_ids:
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
            if len(out) == top_k:
                break
        if len(out) < top_k:
            errors.append(f"{qid}: returned fewer than top_k doc ids")
        ranking[qid] = out
    return ranking, errors


def run_fixed_baseline(
    corpus: dict[str, str], queries: dict[str, str], top_k: int
) -> tuple[dict[str, list[str]], list[str]]:
    try:
        retriever = TfidfBaselineRetriever.from_corpus(corpus)
    except Exception as e:
        return {}, [f"baseline init failed: {e}"]

    ranking = {}
    errors = []
    corpus_ids = set(corpus.keys())
    for qid, text in queries.items():
        try:
            doc_ids = retriever.retrieve(text, top_k)
        except Exception as e:
            errors.append(f"baseline runtime error on {qid}: {e}")
            continue
        if not isinstance(doc_ids, list):
            errors.append(f"baseline {qid}: return type must be list[str]")
            continue
        seen = set()
        out = []
        for doc_id in doc_ids:
            if not isinstance(doc_id, str):
                continue
            if doc_id not in corpus_ids:
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
            if len(out) == top_k:
                break
        if len(out) < top_k:
            errors.append(f"baseline {qid}: returned fewer than top_k doc ids")
        ranking[qid] = out
    return ranking, errors


def fail_payload(baseline_score: dict | None = None) -> dict:
    return {
        "status": "failed",
        "baseline_score": baseline_score or {"mrr@10": 0.0, "ndcg@10": 0.0},
        "search_score": None,
    }


def evaluate_solution(solution_path: Path, top_k: int = 10) -> dict:
    if not DATA_DIR.exists():
        return fail_payload()

    corpus = load_json(DATA_DIR / "corpus.json")
    hidden_queries = load_json(DATA_DIR / "queries_hidden.json")
    hidden_qrels = load_json(DATA_DIR / "qrels_hidden.json")

    baseline_ranking, baseline_errors = run_fixed_baseline(corpus, hidden_queries, top_k)
    if baseline_errors:
        return fail_payload()

    baseline_score = score_metrics(hidden_qrels, baseline_ranking)

    if not solution_path.exists():
        return fail_payload(baseline_score=baseline_score)

    try:
        tree = parse_solution(solution_path)
    except Exception:
        return fail_payload(baseline_score=baseline_score)
    if not has_function_signature(tree, "retrieve"):
        return fail_payload(baseline_score=baseline_score)
    if not has_function_signature(tree, "baseline_retrieve"):
        return fail_payload(baseline_score=baseline_score)

    try:
        module = load_module(solution_path)
    except Exception:
        return fail_payload(baseline_score=baseline_score)

    search_ranking, search_errors = run_pipeline(module.retrieve, corpus, hidden_queries, top_k)
    if search_errors:
        return fail_payload(baseline_score=baseline_score)

    search_score = score_metrics(hidden_qrels, search_ranking)
    passed = (
        search_score["mrr@10"] >= baseline_score["mrr@10"] + 0.05
        and search_score["ndcg@10"] >= baseline_score["ndcg@10"] + 0.05
    )
    return {
        "status": "passed" if passed else "failed",
        "baseline_score": baseline_score,
        "search_score": search_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="workspace/solution.py")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = evaluate_solution(Path(args.solution), top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
