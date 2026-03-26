from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TfidfBaselineRetriever:
    doc_ids: list[str]
    matrix: np.ndarray
    vectorizer: TfidfVectorizer

    @classmethod
    def from_corpus(cls, corpus: dict[str, str]) -> "TfidfBaselineRetriever":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            smooth_idf=True,
            max_features=80000,
            min_df=1,
            max_df=0.95,
        )
        doc_ids = list(corpus.keys())
        matrix = vectorizer.fit_transform([corpus[d] for d in doc_ids])
        return cls(doc_ids=doc_ids, matrix=matrix, vectorizer=vectorizer)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [self.doc_ids[i] for i in order]
