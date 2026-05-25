"""
Build corpus + query embeddings with the Instructor bi-encoder, for first-stage retrieval.

Outputs two pickles:
    {doc_id: np.ndarray}  ->  corpus_embeddings.pkl
    {qid:    np.ndarray}  ->  query_embeddings.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict

import numpy as np

from data import load_nfcorpus_split


CORPUS_INSTRUCTION = "Represent the medical document for retrieval:"
QUERY_INSTRUCTION = "Represent the medical query for retrieving relevant documents:"


def encode_all(model, texts_with_inst):
    return model.encode(texts_with_inst).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nfcorpus_dir", required=True)
    p.add_argument("--out_dir", default="data/embeddings")
    p.add_argument("--instructor_model", default="hkunlp/instructor-xl")
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = p.parse_args()

    from InstructorEmbedding import INSTRUCTOR

    model = INSTRUCTOR(args.instructor_model)

    corpus, queries, _ = load_nfcorpus_split(args.nfcorpus_dir, args.split)
    os.makedirs(args.out_dir, exist_ok=True)

    corpus_inputs = [[CORPUS_INSTRUCTION, c.get("title", "") + "\n" + c.get("text", "")] for c in corpus.values()]
    corpus_ids = list(corpus.keys())
    print(f"Encoding {len(corpus_inputs)} documents ...")
    corpus_vecs = encode_all(model, corpus_inputs)
    corpus_embeds: Dict[str, np.ndarray] = dict(zip(corpus_ids, corpus_vecs))
    with open(os.path.join(args.out_dir, "corpus_embeddings.pkl"), "wb") as f:
        pickle.dump(corpus_embeds, f)

    query_inputs = [[QUERY_INSTRUCTION, q] for q in queries.values()]
    qids = list(queries.keys())
    print(f"Encoding {len(query_inputs)} queries ...")
    q_vecs = encode_all(model, query_inputs)
    q_embeds: Dict[str, np.ndarray] = dict(zip(qids, q_vecs))
    with open(os.path.join(args.out_dir, "query_embeddings.pkl"), "wb") as f:
        pickle.dump(q_embeds, f)

    print("Done.")


if __name__ == "__main__":
    main()
