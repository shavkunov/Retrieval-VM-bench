import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_base_vectorizer = None
_base_doc_ids = None
_base_matrix = None


def _init_baseline(corpus: dict[str, str]):
    global _base_vectorizer, _base_doc_ids, _base_matrix
    _base_doc_ids = list(corpus.keys())
    docs = [corpus[doc_id] for doc_id in _base_doc_ids]
    _base_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        smooth_idf=True,
        max_features=80000,
        min_df=1,
        max_df=0.95,
    )
    _base_matrix = _base_vectorizer.fit_transform(docs)


def baseline_retrieve(query: str, top_k: int, corpus: dict[str, str]) -> list[str]:
    global _base_vectorizer, _base_doc_ids, _base_matrix
    if _base_vectorizer is None or _base_doc_ids is None or _base_matrix is None:
        _init_baseline(corpus)
    query_vec = _base_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _base_matrix).flatten()
    order = np.argsort(-scores)
    out = []
    seen = set()
    for idx in order:
        doc_id = _base_doc_ids[idx]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if len(out) == top_k:
            break
    if len(out) < top_k:
        for doc_id in _base_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
            if len(out) == top_k:
                break
    return out


from rank_bm25 import BM25Okapi
import re

_bm25 = None
_bm25_doc_ids = None
_bm25_corpus_tokens = None

def _tokenize(text):
    # Simple tokenizer: lowercase and split on non-alphanumeric
    return re.findall(r"\b\w+\b", text.lower())

def _init_bm25(corpus: dict[str, str]):
    global _bm25, _bm25_doc_ids, _bm25_corpus_tokens
    _bm25_doc_ids = list(corpus.keys())
    docs = [corpus[doc_id] for doc_id in _bm25_doc_ids]
    _bm25_corpus_tokens = [_tokenize(doc) for doc in docs]
    _bm25 = BM25Okapi(_bm25_corpus_tokens)

def retrieve(query: str, top_k: int, corpus: dict[str, str]) -> list[str]:
    global _bm25, _bm25_doc_ids, _bm25_corpus_tokens
    if _bm25 is None or _bm25_doc_ids is None or _bm25_corpus_tokens is None:
        _init_bm25(corpus)
    if corpus is None or len(corpus) == 0:
        return []
    query_tokens = _tokenize(query)
    scores = _bm25.get_scores(query_tokens)
    order = np.argsort(-scores)
    out = []
    seen = set()
    for idx in order:
        doc_id = _bm25_doc_ids[idx]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if len(out) == top_k:
            break
    if len(out) < top_k:
        for doc_id in _bm25_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
            if len(out) == top_k:
                break
    return out



