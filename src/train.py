"""
Fine-tune a cross-encoder on NFCorpus.

This is a sanitized, public-domain version of the original training pipeline. It uses
only the public NFCorpus dataset (BEIR), the public `cross-encoder/ms-marco-MiniLM-L-4-v2`
base model, and the public `hkunlp/instructor-xl` bi-encoder (or any embedding model you
plug in) for the first-stage retriever.

Usage:
    python src/train.py \\
        --nfcorpus_dir data/nfcorpus \\
        --base_model cross-encoder/ms-marco-MiniLM-L-4-v2 \\
        --output_dir models/ms-marco-MiniLM-L-4-v2-nfcorpus \\
        --batch_size 164 --lr 5e-6 --warmup 5000 --steps_per_chunk 16384

The bi-encoder used for hard-negative mining is expected to have produced two pickle
files: `corpus_embeddings.pkl` and `query_embeddings.pkl`, each a `dict[str, np.ndarray]`.
See `src/embed_corpus.py` for a reference implementation using Instructor.
"""

from __future__ import annotations

import argparse
import os
import pickle

import torch
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader

from data import (
    build_faiss_index,
    build_training_pairs,
    load_nfcorpus_split,
    mine_hard_negatives,
    shuffle_samples,
)
from evaluate import evaluate
from rerank import rerank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nfcorpus_dir", required=True)
    p.add_argument("--corpus_embeddings", required=True, help="pickle of {doc_id: np.ndarray}")
    p.add_argument("--query_embeddings", required=True, help="pickle of {query_id: np.ndarray}")
    p.add_argument("--base_model", default="cross-encoder/ms-marco-MiniLM-L-4-v2")
    p.add_argument("--output_dir", default="models/ms-marco-MiniLM-L-4-v2-nfcorpus")
    p.add_argument("--batch_size", type=int, default=164)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--steps_per_chunk", type=int, default=16384)
    p.add_argument("--retrieve_top_k", type=int, default=300)
    p.add_argument("--eval_top_k", type=int, default=100)
    p.add_argument("--use_amp", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] Loading NFCorpus train / test splits")
    train_corpus, train_queries, train_qrels = load_nfcorpus_split(args.nfcorpus_dir, "train")
    test_corpus, test_queries, test_qrels = load_nfcorpus_split(args.nfcorpus_dir, "test")

    print("[2/5] Loading precomputed embeddings + building FAISS index")
    with open(args.corpus_embeddings, "rb") as f:
        corpus_embeddings = pickle.load(f)
    with open(args.query_embeddings, "rb") as f:
        query_embeddings = pickle.load(f)
    index, id_map = build_faiss_index(corpus_embeddings)
    print(f"  -> indexed {index.ntotal} documents")

    print("[3/5] Mining hard negatives")
    train_results = mine_hard_negatives(
        train_qrels, query_embeddings, index, id_map, top_k=args.retrieve_top_k
    )
    train_samples = build_training_pairs(train_corpus, train_queries, train_qrels, train_results)
    train_samples = shuffle_samples(train_samples)
    print(f"  -> {len(train_samples)} (query, doc, label) pairs")

    print(f"[4/5] Loading base model {args.base_model}")
    torch.cuda.empty_cache()
    model = CrossEncoder(args.base_model, num_labels=1)

    # First-stage retrieval results for the test split (used for eval reranking)
    test_results = mine_hard_negatives(
        test_qrels, query_embeddings, index, id_map, top_k=args.eval_top_k
    )
    k_eval = [1, 3, 5, 10, 100]
    print("Before fine-tuning, reranking test split:")
    print(evaluate(test_qrels, rerank(model, test_corpus, test_queries, test_results, args.eval_top_k), k_eval))

    print("[5/5] Fine-tuning in chunks")
    n = len(train_samples)
    chunks = (n + args.steps_per_chunk - 1) // args.steps_per_chunk
    for i in range(chunks):
        start = i * args.steps_per_chunk
        end = min(start + args.steps_per_chunk, n)
        chunk = train_samples[start:end]
        loader = DataLoader(chunk, shuffle=True, batch_size=args.batch_size)
        torch.cuda.empty_cache()
        model.fit(
            train_dataloader=loader,
            optimizer_params={"lr": args.lr},
            epochs=1,
            warmup_steps=args.warmup,
            output_path=args.output_dir,
            use_amp=args.use_amp,
        )
        model.save(args.output_dir)
        print(f"  checkpoint {start}/{n}")
        metrics = evaluate(
            test_qrels,
            rerank(model, test_corpus, test_queries, test_results, args.eval_top_k),
            k_eval,
        )
        print("  ", metrics)


if __name__ == "__main__":
    main()
