"""
NFCorpus loading + hard-negative mining for cross-encoder fine-tuning.

The training pipeline is a classic "first-stage retriever -> hard negatives -> cross-encoder"
setup:

    1. A bi-encoder embeds every document and every training query.
    2. For each training query we FAISS-search the top-K documents.
    3. Any retrieved doc that is NOT in the ground-truth qrels for that query becomes a
       hard negative (label = 0). The ground-truth positives get label = 1.
    4. Those (query, doc, label) triples become InputExample(s) that the cross-encoder is
       fine-tuned on with BCE loss.

NFCorpus comes from the BEIR benchmark and is publicly downloadable. See `download_nfcorpus`.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import InputExample


def download_nfcorpus(target_dir: str = "data") -> str:
    """Download and extract NFCorpus from the official BEIR release.

    Returns the path to the extracted `nfcorpus/` directory.
    """
    import zipfile
    import urllib.request

    os.makedirs(target_dir, exist_ok=True)
    out = os.path.join(target_dir, "nfcorpus.zip")
    if not os.path.exists(out):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, out)
    extracted = os.path.join(target_dir, "nfcorpus")
    if not os.path.exists(extracted):
        with zipfile.ZipFile(out) as z:
            z.extractall(target_dir)
    return extracted


def load_nfcorpus_split(nfcorpus_dir: str, split: str = "train"):
    """Load a BEIR-format split. Returns (corpus, queries, qrels)."""
    from beir.datasets.data_loader import GenericDataLoader
    return GenericDataLoader(nfcorpus_dir).load(split=split)


def build_faiss_index(corpus_embeddings: Dict[str, np.ndarray]):
    """Build a flat L2 FAISS index over the corpus embeddings.

    Returns (index, id_map) where id_map[i] is the doc_id for FAISS row i.
    """
    import faiss

    first_vec = next(iter(corpus_embeddings.values()))
    dim = int(first_vec.shape[-1])
    index = faiss.IndexFlatL2(dim)
    id_map: Dict[int, str] = {}
    for doc_id, vec in corpus_embeddings.items():
        id_map[index.ntotal] = doc_id
        index.add(np.expand_dims(vec.astype(np.float32), axis=0))
    return index, id_map


def mine_hard_negatives(
    qrels: Dict[str, Dict[str, int]],
    queries_embeddings: Dict[str, np.ndarray],
    index,
    id_map: Dict[int, str],
    top_k: int = 300,
) -> Dict[str, Dict[str, float]]:
    """For each query, search top_k docs; anything not in qrels becomes a candidate negative."""
    results: Dict[str, Dict[str, float]] = {}
    for query_id in qrels:
        vec = np.expand_dims(queries_embeddings[str(query_id)].astype(np.float32), axis=0)
        D, I = index.search(vec, top_k)
        per_query: Dict[str, float] = {}
        for idx, dist in zip(I[0], D[0]):
            doc_id = id_map[int(idx)]
            per_query[doc_id] = float(1 - dist)
        results[query_id] = per_query
    return results


def build_training_pairs(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    retriever_results: Dict[str, Dict[str, float]],
) -> List[InputExample]:
    """Convert (qrels + retriever results) into labelled (query, doc) pairs.

    Positives = qrels (label 1). Negatives = retrieved docs not in qrels (label 0).
    """
    samples: List[InputExample] = []
    for query_id, rels in qrels.items():
        q_text = queries[query_id]
        for doc_id in rels:
            samples.append(InputExample(texts=[q_text, corpus[doc_id]["text"]], label=1))
        for doc_id in retriever_results.get(query_id, {}):
            if doc_id not in rels:
                samples.append(InputExample(texts=[q_text, corpus[doc_id]["text"]], label=0))
    return samples


def shuffle_samples(samples: List[InputExample], seed: int = 13, passes: int = 3) -> List[InputExample]:
    rng = random.Random(seed)
    samples = list(samples)
    for _ in range(passes):
        rng.shuffle(samples)
    return samples


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
