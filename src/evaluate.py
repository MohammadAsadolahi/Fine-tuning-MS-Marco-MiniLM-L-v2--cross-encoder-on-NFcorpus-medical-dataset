"""
Retrieval evaluation utilities (NDCG/MAP/Recall/P@k) using pytrec_eval.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Compute NDCG@k, MAP@k, Recall@k, P@k for the given k_values.

    `results` may contain more than k documents per query — pytrec_eval handles cut-offs.
    """
    import pytrec_eval

    ndcg, _map, recall, precision = {}, {}, {}, {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_str = "map_cut." + ",".join(str(k) for k in k_values)
    ndcg_str = "ndcg_cut." + ",".join(str(k) for k in k_values)
    recall_str = "recall." + ",".join(str(k) for k in k_values)
    p_str = "P." + ",".join(str(k) for k in k_values)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_str, ndcg_str, recall_str, p_str})
    scores = evaluator.evaluate(results)

    for qid in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[qid][f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"] += scores[qid][f"map_cut_{k}"]
            recall[f"Recall@{k}"] += scores[qid][f"recall_{k}"]
            precision[f"P@{k}"] += scores[qid][f"P_{k}"]

    n = max(len(scores), 1)
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / n, 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / n, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / n, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / n, 5)

    return ndcg, _map, recall, precision
