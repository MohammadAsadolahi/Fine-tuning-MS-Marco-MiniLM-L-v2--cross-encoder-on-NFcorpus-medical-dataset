# Tutorial — Cross-Encoder Reranking for Medical RAG

This tutorial walks you through the **why** and the **how** of fine-tuning a small
MS-MARCO cross-encoder on NFCorpus, and then plugging it into a two-stage retrieval
pipeline for a medical RAG system.

> Audience: an ML engineer who has built a vector store before and now wants better
> top-k precision than the bi-encoder alone gives.

---

## 1. Why a cross-encoder at all?

A **bi-encoder** (DPR, Sentence-BERT, Instructor, GTE, BGE, …) encodes the query and
each document into independent vectors. Similarity is dot-product / cosine. That's fast
and scalable but the model never *jointly* attends to the query and the document, so
fine distinctions are lost.

A **cross-encoder** runs the *(query, document)* pair through the transformer together —
every query token can attend to every document token. That gives much better ranking
quality, but the model has to run once per pair, so we can only afford to run it on a
shortlist (the bi-encoder's top-K). Hence the two-stage pattern.

```
                ┌──────────────────┐
   query ─────► │   bi-encoder     │ ──► top-100 docs (fast, recall-oriented)
                └──────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  cross-encoder   │ ──► top-5 (slow, precision-oriented)
                └──────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │       LLM        │ ──► grounded answer
                └──────────────────┘
```

We use `cross-encoder/ms-marco-MiniLM-L-4-v2` because it has a great
quality / latency trade-off — 4 transformer layers, ~80 MB, ~10 ms / pair on a single
modern GPU.

---

## 2. Why NFCorpus?

NFCorpus (Boteva et al. 2016) is the medical IR task in
[BEIR](https://github.com/beir-cellar/beir). It has:

- 3,633 documents (titles + abstracts from NIH / PubMed-style sources)
- ~2,600 training queries with graded relevance judgements
- 323 test queries

It is small enough to iterate on a laptop GPU, and it is genuinely *medical* — exactly
the distribution shift the original MS-MARCO cross-encoder is weakest at.

Download:

```python
from src.data import download_nfcorpus
download_nfcorpus("data")   # -> data/nfcorpus/
```

The resulting directory has BEIR's standard layout:

```
data/nfcorpus/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
```

---

## 3. Step-by-step pipeline

### 3.1 First-stage embeddings

We use [`hkunlp/instructor-xl`](https://huggingface.co/hkunlp/instructor-xl) as the
bi-encoder, with explicit instructions so it knows the domain:

```python
CORPUS_INSTRUCTION = "Represent the medical document for retrieval:"
QUERY_INSTRUCTION  = "Represent the medical query for retrieving relevant documents:"
```

`src/embed_corpus.py` produces `corpus_embeddings.pkl` and `query_embeddings.pkl` —
both `dict[str, np.ndarray]` keyed by BEIR document / query IDs.

You can swap in any other bi-encoder (`BAAI/bge-large-en-v1.5`, `intfloat/e5-large`, …)
— just produce the two pickles in the same format.

### 3.2 Build the FAISS index

```python
from src.data import build_faiss_index
index, id_map = build_faiss_index(corpus_embeddings)
```

We use `IndexFlatL2` because 3.6 k docs is tiny — exact search costs nothing. For
larger corpora use `IndexIVFFlat` or `IndexHNSWFlat`.

### 3.3 Hard-negative mining

For every training query, retrieve the **top-300** docs from the bi-encoder. Anything
not in the ground-truth qrels for that query becomes a hard negative.

Why 300? It needs to be (a) large enough to find genuinely confusing negatives the
cross-encoder will struggle with, and (b) small enough that we don't drown the
positives. Empirically 100–500 works; 300 was the sweet spot in this dataset.

```python
from src.data import mine_hard_negatives, build_training_pairs

results = mine_hard_negatives(train_qrels, query_embeddings, index, id_map, top_k=300)
samples = build_training_pairs(train_corpus, train_queries, train_qrels, results)
# samples is a list of sentence_transformers.InputExample with label in {0, 1}
```

In our run this produced **~835 k** labelled pairs.

### 3.4 Fine-tune in chunks

We do **NOT** train one giant epoch. We slice the shuffled training pool into chunks
of 16384 examples and after each chunk we

- save the cross-encoder
- rerank the test split's top-100
- compute NDCG/MAP/Recall/P@k

That gives us a clean learning curve and lets us pick the *best* checkpoint instead of
the *last* one. The relevant excerpt from `src/train.py`:

```python
for i in range(chunks):
    chunk = train_samples[i*16384 : (i+1)*16384]
    loader = DataLoader(chunk, shuffle=True, batch_size=164)
    model.fit(
        train_dataloader = loader,
        optimizer_params = {"lr": 5e-6},
        epochs           = 1,
        warmup_steps     = 5000,
        output_path      = "models/ms-marco-MiniLM-L-4-v2-nfcorpus",
        use_amp          = True,
    )
    model.save("models/ms-marco-MiniLM-L-4-v2-nfcorpus")
    print(evaluate(test_qrels,
                   rerank(model, test_corpus, test_queries, test_results, top_k=100),
                   k_values=[1, 3, 5, 10, 100]))
```

Hyper-parameters that worked well for L-4-v2:

| parameter        | value     |
|------------------|-----------|
| batch size       | 164       |
| learning rate    | 5e-6      |
| warmup steps     | 5000      |
| mixed precision  | yes (AMP) |
| chunk size       | 16384     |
| optimizer        | AdamW (st default) |
| epochs / chunk   | 1         |

### 3.5 Use the model

```python
from sentence_transformers import CrossEncoder

ce = CrossEncoder("models/ms-marco-MiniLM-L-4-v2-nfcorpus")
scores = ce.predict([
    ("diabetes treatment", "Type 1 and 2 diabetes mellitus: a review …"),
    ("diabetes treatment", "Impact of salt intake on hypertension …"),
])
```

Score is a single logit per pair — higher = more relevant. There is no fixed cutoff;
the model is meant for *ranking*, not classification.

To plug into a RAG pipeline:

```python
from src.rerank import rerank

# `bi_encoder_results[qid]` is a dict {doc_id: bi-encoder score} of the top 100
reranked = rerank(ce, corpus, queries, bi_encoder_results, top_k=100)
top5 = sorted(reranked[qid].items(), key=lambda kv: -kv[1])[:5]
```

---

## 4. What changes vs. the off-the-shelf cross-encoder?

Looking at `reports/training_metrics.csv`, the most useful gain is at the **top of the
list** (NDCG@1, P@1), which is exactly what matters for an LLM prompt that only sees a
handful of documents:

- NDCG@1: 0.4242 → 0.4427 ( +1.9 absolute / +4.4% relative )
- P@1:    0.4396 → 0.4582 ( +1.9 absolute / +4.3% relative )

NDCG@10 and NDCG@100 are essentially unchanged — once you go beyond the top few results,
the bi-encoder's ordering already dominates. That's the *signature* of cross-encoder
reranking: it pushes the right answer to position 1.

---

## 5. Common pitfalls

- **Don't shuffle qrels into the negative pool.** `build_training_pairs` checks
  `doc_id not in qrels[qid]` before labelling something as a negative — losing this
  check silently destroys training (the model gets told the right answer is wrong).
- **Watch the LR.** Cross-encoders need a *small* LR. `5e-6` is the sweet spot for
  L-2 / L-4. If you crank it to `1e-4` you'll see all metrics collapse within an epoch.
- **AMP can cause NaNs on tiny GPUs.** If you see NaN losses, set `use_amp=False` and
  cut the batch size in half.
- **`results[query_id]` must keep more than `top_k` entries** if you want to rerank
  exactly `top_k`. We retrieve 300 at training time but only 100 at eval time.

---

## 6. Where this came from

The recipe was developed for a hybrid retrieval stage of a medical-RAG system.
The version published here keeps the public datasets (NFCorpus), the public base model
(`cross-encoder/ms-marco-MiniLM-L-4-v2`), and the public bi-encoder (`hkunlp/instructor-xl`)
— no proprietary corpora, queries, or evaluation sets are included.
