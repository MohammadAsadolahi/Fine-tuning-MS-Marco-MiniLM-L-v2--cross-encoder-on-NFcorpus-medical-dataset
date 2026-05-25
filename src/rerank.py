"""
Cross-encoder reranking over a first-stage retriever's results.

Given retriever scores per query, take the top_k documents, score each (query, document)
pair with the cross-encoder, and return the reranked scores.
"""

from __future__ import annotations

from typing import Dict


def rerank(
    model,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    results: Dict[str, Dict[str, float]],
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    pair_ids, sentence_pairs = [], []
    for query_id, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        for doc_id, _ in sorted_docs:
            pair_ids.append((query_id, doc_id))
            doc = corpus[doc_id]
            text = (doc.get("title", "") + "\n" + doc.get("text", "")).strip()
            sentence_pairs.append([queries[query_id], text])

    scores = [float(s) for s in model.predict(sentence_pairs)]

    reranked: Dict[str, Dict[str, float]] = {qid: {} for qid in results}
    for (qid, doc_id), score in zip(pair_ids, scores):
        reranked[qid][doc_id] = score
    return reranked
