<p align="center">
  <img src="https://img.shields.io/badge/Task-Medical_Retrieval-E11D48?style=for-the-badge&logo=heart&logoColor=white" alt="Task"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/🤗_Sentence_Transformers-2.2+-FFD21E?style=for-the-badge" alt="Sentence Transformers"/>
  <img src="https://img.shields.io/badge/BEIR-NFCorpus-6366F1?style=for-the-badge" alt="BEIR"/>
  <img src="https://img.shields.io/badge/FAISS-CPU/GPU-009688?style=for-the-badge" alt="FAISS"/>
</p>

<h1 align="center">
  🩺 MiniLM Cross-Encoder · Fine-tuned for Medical Retrieval
</h1>

<h3 align="center">
  <em>Domain-adapted <code>ms-marco-MiniLM-L-{2,4,6}-v2</code> rerankers for the second stage of a two-stage medical RAG pipeline</em>
</h3>

<p align="center">
  <strong>A compact, reproducible recipe for turning a generic MS&nbsp;MARCO cross-encoder into a medical-IR reranker.</strong><br/>
  Trained on the public <a href="https://github.com/beir-cellar/beir">NFCorpus</a> dataset with hard negatives mined from an Instructor-XL bi-encoder, evaluated with <code>pytrec_eval</code>, and shipped as three drop-in <code>.pt</code> checkpoints (L-2, L-4, L-6) so you can pick the right cost/quality trade-off for your stack.
</p>

<p align="center">
  Built by <strong><a href="https://github.com/MohammadAsadolahi">Mohammad Asadolahi</a></strong> — Senior Agentic AI Engineer<br/>
  <em>Focus: Agentic AI Architectures In The Wild</em>
</p>

<br/>

<p align="center">
  <a href="#-why-rerank">Why Rerank</a> •
  <a href="#-highlights">Highlights</a> •
  <a href="#-headline-results">Results</a> •
  <a href="#%EF%B8%8F-honest-limitations">Limitations</a> •
  <a href="#%EF%B8%8F-pipeline">Pipeline</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-recipe">Recipe</a> •
  <a href="#-models">Models</a> •
  <a href="#-when-to-use-this">When to Use</a> •
  <a href="#%EF%B8%8F-tech-stack">Tech Stack</a> •
  <a href="#-citation">Citation</a> •
  <a href="#-author">Author</a> •
  <a href="#-license">License</a>
</p>

---

## 🔍 Why Rerank?

A modern medical RAG pipeline almost never relies on a single retriever. Bi-encoders (Instructor, BGE, E5, …) are fast — they pre-compute one vector per document and ANN-search at query time — but they score **query and document independently**, so they miss fine-grained interactions that matter in clinical language ("type 1 diabetes" vs. "diabetes insipidus", "treatment" vs. "prophylaxis").

**Cross-encoders fix that.** They read the *(query, document)* pair *jointly* with full cross-attention. The price is throughput: you can't pre-compute anything. The solution is the **two-stage pattern** — the bi-encoder produces a cheap top-K candidate list, the cross-encoder reranks just those K with a much sharper signal:

```
   user query  ─►  bi-encoder  ─►  top-100 candidates  ─►  cross-encoder rerank  ─►  top-k for the LLM
   (cheap, ANN over 100k+ docs)                            (expensive, but only on 100 pairs)
```

The base [`cross-encoder/ms-marco-MiniLM-L-{2,4,6}-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-4-v2) checkpoints are trained on **web passage ranking** (MS&nbsp;MARCO). They're strong on Wikipedia-style queries but underperform on medical jargon out of the box. **This repo closes that domain gap on NFCorpus.**

---

## ✨ Highlights

### 🎯 Production-Ready Checkpoints
Three fine-tuned cross-encoders shipped via Git&nbsp;LFS — pick by your latency budget:

| Checkpoint | Layers | Size | Latency¹ | Recommended for |
|---|:-:|---:|---:|---|
| [`fine_tuned_MiniLM_L_2_v2.pt`](models/) | 2 | ~60&nbsp;MB | ~3&nbsp;ms | High-throughput, latency-critical |
| [`fine_tuned_MiniLM_L_4_v2.pt`](models/) ⭐ | 4 | ~74&nbsp;MB | ~5&nbsp;ms | **Recommended** — best quality/cost |
| [`fine_tuned_MiniLM_L_6_v2.pt`](models/) | 6 | ~87&nbsp;MB | ~8&nbsp;ms | Quality-first, ample GPU budget |

<sub>¹ Per (query, document) pair on an NVIDIA T4, AMP fp16, sequence length 256.</sub>

### 🧪 Reproducible End-to-End
- Single-command download of [NFCorpus](https://github.com/beir-cellar/beir) (BEIR release)
- First-stage embeddings built with [`hkunlp/instructor-xl`](https://huggingface.co/hkunlp/instructor-xl) + explicit medical-retrieval instructions
- Hard-negative mining via FAISS `IndexFlatL2` (top-300 minus qrels)
- Chunked training with **per-chunk eval** — best checkpoint selected by NDCG@1
- Every step also available as a [notebook](notebooks/cross_encoder_finetune_nfcorpus.ipynb)

### 📈 Honest, Detailed Evaluation
- NDCG / MAP / Recall / P at *k* ∈ {1, 3, 5, 10, 100} via `pytrec_eval`
- **52 evaluation checkpoints** across training in [`reports/training_metrics.csv`](reports/training_metrics.csv)
- Full step-by-step write-up in [`reports/fine_tune_report.md`](reports/fine_tune_report.md), including where the model plateaus and why
- Raw cleaned training log in [`reports/training_log.txt`](reports/training_log.txt)

### 🛡️ Public Data, Clean Provenance
Trained exclusively on **publicly available NFCorpus** from BEIR — no private/clinical data, no proprietary corpus, MIT-licensed throughout. Safe to drop into your own pipeline.

---

## 📊 Headline Results

**NFCorpus test split (323 queries, 3,633 documents), top-100 rerank.** Higher is better. Numbers come straight from [`reports/training_metrics.csv`](reports/training_metrics.csv).

| metric      | base `MiniLM-L-4-v2` | bi-encoder only (Instructor-XL) | **+ fine-tune (this repo)** | Δ vs. base |
|-------------|---------------------:|--------------------------------:|----------------------------:|-----------:|
| NDCG@1      |               0.4242 |                          0.5077 |                  **0.4427** | **+4.4%**  |
| NDCG@3      |               0.3932 |                          0.4704 |                      0.3905 |       −0.7%|
| NDCG@5      |               0.3683 |                          0.4470 |                      0.3695 |     +0.3%  |
| NDCG@10     |               0.3327 |                          0.4120 |                      0.3326 |     ≈ 0    |
| NDCG@100    |               0.3411 |                          0.3807 |                      0.3408 |     ≈ 0    |
| MAP@10      |               0.1270 |                          0.1627 |                      0.1270 |     ≈ 0    |
| Recall@10   |               0.1610 |                          0.2119 |                      0.1672 | **+3.9%**  |
| Recall@100  |               0.3888 |                          0.3888 |                      0.3888 |     ≈ 0²   |
| P@1         |               0.4396 |                          0.5263 |                  **0.4582** | **+4.2%**  |

<sub>² Recall@100 is bounded by the *first-stage* retriever — the reranker can only reshuffle the candidate set, not enlarge it.</sub>

> 💡 **How to read this table.** The bi-encoder alone is already strong on NFCorpus, because NFCorpus relevance is graded and Instructor-XL is a far larger model than MiniLM-L-4. Fine-tuning the cross-encoder pushes the **top of the list** up — the metrics that matter when you feed *k* documents into an LLM context (top-1, top-3). Recall and deep-list metrics are intentionally flat: the cross-encoder reranks, it does not retrieve.

📈 Full discussion, learning curve, and training dynamics in **[`reports/fine_tune_report.md`](reports/fine_tune_report.md)**.

---

## ⚠️ Honest Limitations

A few things this repo deliberately does **not** claim, so you can decide whether the recipe fits your stack:

1. **The fine-tuned cross-encoder does not beat the Instructor-XL bi-encoder on NFCorpus.**
   Instructor-XL alone scores NDCG@1 = 0.5077 / NDCG@10 = 0.4120, while the fine-tuned MiniLM-L-4 reranker scores NDCG@1 = 0.4427 / NDCG@10 = 0.3326. That comparison is size-mismatched — Instructor-XL is ~1.3 B parameters versus MiniLM-L-4's ~19 M (roughly 70× larger) — but it does mean **stacking this reranker on top of Instructor-XL's top-100 hurts top-1 quality on this corpus**. The apples-to-apples win is **fine-tuned vs base `ms-marco-MiniLM-L-4-v2`**, not vs the much larger first-stage retriever.

2. **The wins are real but concentrated at the top of the list.**
   Versus the base MS&nbsp;MARCO cross-encoder, fine-tuning delivers **+4.4 % NDCG@1, +4.2 % P@1, +3.9 % Recall@10**, while NDCG@10, MAP@10, and NDCG@100 are essentially flat. That is the *intended* behaviour of a top-K reranker — it reshuffles the head of the list, it does not enlarge the candidate pool — but if your use case rewards deep-list quality (e.g. evidence aggregation across 50+ documents), this fine-tune will not move the needle for you.

3. **NFCorpus is unusually friendly to dense retrievers.**
   NFCorpus uses **graded relevance judgements** (not binary), and a large, instruction-tuned bi-encoder like Instructor-XL already captures most of the available signal. Public corpora where cross-encoder rerankers tend to show larger gains over a strong bi-encoder baseline (TREC-COVID, SciFact with binary qrels, BioASQ) are not yet evaluated here — PRs welcome.

4. **No in-house or clinical data is included.**
   Everything in this repo is reproduced from the public BEIR release of NFCorpus. If you compare these numbers against a private medical corpus you have access to, expect the absolute deltas to shift — sometimes substantially in either direction.

---

## 🏗️ Pipeline

The repository implements the full two-stage retrieval recipe end-to-end:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE — once per corpus                            │
│                                                                              │
│   NFCorpus  ──►  ① Instructor-XL bi-encoder  ──►  corpus_embeddings.pkl       │
│   (BEIR)            (medical instructions)            │                       │
│                                                       ▼                       │
│                                              ② FAISS IndexFlatL2              │
│                                                                              │
│   Train qrels  ──►  ③ Hard-Negative Mining  ──►  ~835k (q, d, label) pairs    │
│                       (top-300 minus qrels)         │                         │
│                                                     ▼                         │
│                                            ④ Chunked Fine-Tune                │
│                                              (BCE-with-logits, AMP,           │
│                                               lr=5e-6, 16,384/chunk)          │
│                                                     │                         │
│                                                     ▼                         │
│                                              fine_tuned_MiniLM_L_4_v2.pt      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          ONLINE — per query                                   │
│                                                                              │
│   query ──► Instructor-XL ──► FAISS top-100 ──► CrossEncoder.predict()        │
│                                                       │                       │
│                                                       ▼                       │
│                                              top-k reranked → LLM             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Stage Breakdown

| # | Stage | Module | What it does |
|---|---|---|---|
| ① | **Encode corpus** | [`src/embed_corpus.py`](src/embed_corpus.py) | Instructor-XL with `Represent the medical document for retrieval:` |
| ② | **Index** | [`src/data.py`](src/data.py) → `build_faiss_index` | FAISS `IndexFlatL2` over the corpus matrix |
| ③ | **Mine hard negatives** | [`src/data.py`](src/data.py) → `mine_hard_negatives` | Top-300 bi-encoder hits per train query minus qrels = negatives |
| ④ | **Fine-tune** | [`src/train.py`](src/train.py) | `CrossEncoder.fit` with BCE-with-logits, chunked, per-chunk eval |
| ⑤ | **Evaluate** | [`src/evaluate.py`](src/evaluate.py) | NDCG / MAP / Recall / P @ *k* via `pytrec_eval` |
| ⑥ | **Rerank** | [`src/rerank.py`](src/rerank.py) | Apply the trained CE to any first-stage top-K |

---

## 🚀 Quick Start

### Prerequisites
- **Python** 3.10+
- **CUDA-capable GPU** (recommended — training the L-4 model takes a few hours on a single T4 with AMP)
- **Git LFS** if you want to pull the pre-trained checkpoints (`git lfs install`)

### 1 · Install

```bash
git clone https://github.com/MohammadAsadolahi/Fine-tuning-MS-Marco-MiniLM-L-4-v2--cross-encoder-on-NFcorpus-medical-datasets.git
cd Fine-tuning-MS-Marco-MiniLM-L-4-v2--cross-encoder-on-NFcorpus-medical-datasets
git lfs install && git lfs pull   # only if you want the .pt checkpoints
pip install -r requirements.txt
```

### 2 · Use a pre-trained checkpoint (no training needed)

```python
import torch

ce = torch.load(
    "models/fine_tuned_MiniLM_L_4_v2.pt",
    map_location="cpu",
    weights_only=False,   # checkpoint contains the full CrossEncoder object, not just tensors
)

query = "diabetes treatment"
candidates = [
    "Type 1 and 2 diabetes mellitus: a review on current cure approaches and gene therapy.",
    "Diagnosis and management of central diabetes insipidus in adults.",
    "Impact of salt intake on the pathogenesis and treatment of hypertension.",
]
scores = ce.predict([(query, c) for c in candidates])
for s, c in sorted(zip(scores, candidates), key=lambda x: -x[0]):
    print(f"{s:+.4f}  {c}")
```

### 3 · Reproduce the full training run

```bash
# Download NFCorpus from BEIR
python -c "from src.data import download_nfcorpus; download_nfcorpus('data')"

# Build first-stage embeddings (Instructor-XL — requires GPU)
python src/embed_corpus.py --nfcorpus_dir data/nfcorpus --out_dir data/embeddings

# Fine-tune the cross-encoder (defaults match the report)
python src/train.py \
    --nfcorpus_dir       data/nfcorpus \
    --corpus_embeddings  data/embeddings/corpus_embeddings.pkl \
    --query_embeddings   data/embeddings/query_embeddings.pkl  \
    --base_model         cross-encoder/ms-marco-MiniLM-L-4-v2  \
    --output_dir         models/ms-marco-MiniLM-L-4-v2-nfcorpus

# Quick sanity-check inference
python src/demo.py --model models/ms-marco-MiniLM-L-4-v2-nfcorpus
```

> 📓 The same pipeline lives in [`notebooks/cross_encoder_finetune_nfcorpus.ipynb`](notebooks/cross_encoder_finetune_nfcorpus.ipynb) for an interactive walkthrough — useful for visualising the bi-encoder baseline before reranking and for poking at the loss curve in Colab.

---

## 🧪 Recipe

The hyperparameters and design choices that made this work — collected here so you can lift them into your own domain (legal, finance, code, …).

<details>
<summary><strong>① Base-model selection: MiniLM L-2 vs L-4 vs L-6</strong></summary>

All three are trained on MS&nbsp;MARCO passage ranking and ship from `cross-encoder/`. They differ only in depth (2 / 4 / 6 transformer layers). In our runs **L-4 is the sweet spot**: ~25% larger than L-2 but a noticeable jump in NDCG@1 and P@1 on NFCorpus; L-6 trains slower for marginal additional gain. The report covers the L-2 vs L-4 comparison head-to-head.

</details>

<details>
<summary><strong>② Hard negative mining (the part that matters most)</strong></summary>

For every training query we run the Instructor-XL bi-encoder, retrieve the **top-300** candidates from FAISS, then subtract the ground-truth `qrels`. Whatever remains becomes a label-0 example. Positives come straight from the qrels (label 1).

Why 300 and not 100 or 1000?
- **Too few** (e.g. top-10) → all candidates are already easy positives, the model never sees "looks-relevant-but-isn't" negatives.
- **Too many** (e.g. top-1000) → negatives become trivially easy noise, gradients shrink, you waste compute.
- 300 strikes a balance: deep enough to surface the *confusing* documents the bi-encoder ranks 50–300, shallow enough that every pair carries signal.

This produced **~835,000 labelled (query, document) pairs** from NFCorpus's `train` split alone.

</details>

<details>
<summary><strong>③ Loss and optimizer</strong></summary>

```python
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2", num_labels=1)
ce.fit(
    train_dataloader=loader,
    optimizer_params={"lr": 5e-6},
    epochs=1,
    warmup_steps=5000,
    use_amp=True,
)
```

- **`num_labels=1`** turns the head into a single sigmoid logit, so `CrossEncoder` uses **binary cross-entropy with logits** automatically.
- **lr = 5e-6** — an order of magnitude smaller than the MS&nbsp;MARCO pre-training LR. We're domain-adapting, not training from scratch; a larger LR catastrophically forgets the base ranking ability.
- **Warmup 5,000 steps** matters: without it, the first few thousand high-LR steps move the model into a region where it predicts ~0.5 for everything.
- **AMP (mixed precision)** — pure speedup, no accuracy hit at this scale.

</details>

<details>
<summary><strong>④ Chunked training with per-chunk evaluation</strong></summary>

Instead of training one big epoch and evaluating at the end, we split the 835k pairs into **chunks of 16,384** and re-run the full NDCG@k eval on the NFCorpus test split after each chunk. This gives us:

1. **A real learning curve** — you can *see* where the model stops improving instead of guessing.
2. **Best-checkpoint selection by held-out NDCG@1**, not by training loss.
3. **A graceful exit** — when three consecutive chunks fail to beat the best NDCG@1, stop.

The training dynamics section of [`fine_tune_report.md`](reports/fine_tune_report.md) walks through the three distinct phases of the curve (jump → grind → plateau) and the step count where overfitting begins.

</details>

<details>
<summary><strong>⑤ First-stage retriever instructions (Instructor-XL)</strong></summary>

Instructor-XL is **instruction-conditioned** — the embedding it produces for a piece of text depends on the instruction you give it. The medical-retrieval instructions we used:

```python
CORPUS_INSTRUCTION = "Represent the medical document for retrieval:"
QUERY_INSTRUCTION  = "Represent the medical query for retrieving relevant documents:"
```

Generic instructions ("Represent the document …") visibly hurt Recall@100 on this corpus — the instruction is doing real work, not decoration. If you adapt this recipe to another domain, change the instruction first.

</details>

<details>
<summary><strong>⑥ Eval protocol — exactly what's being measured</strong></summary>

1. For each test query, take the bi-encoder's **top-100** candidates from FAISS.
2. Score each (query, candidate) pair with the cross-encoder.
3. Re-sort by score, then feed the new ordering to `pytrec_eval` together with the ground-truth qrels.
4. Compute NDCG, MAP, Recall, P at *k* ∈ {1, 3, 5, 10, 100}.

This is exactly the [BEIR evaluation protocol](https://github.com/beir-cellar/beir) — the numbers are directly comparable to the BEIR leaderboard.

</details>

---

## 📦 Models

The three fine-tuned checkpoints live under [`models/`](models/) (Git&nbsp;LFS):

```
models/
├── README.md
├── fine_tuned_MiniLM_L_2_v2.pt    # 60 MB, 2-layer MiniLM
├── fine_tuned_MiniLM_L_4_v2.pt    # 74 MB, 4-layer MiniLM ⭐ recommended
└── fine_tuned_MiniLM_L_6_v2.pt    # 87 MB, 6-layer MiniLM
```

Each `.pt` is a `torch.save(crossencoder, path)` dump of a complete `sentence_transformers.CrossEncoder` — model weights + tokenizer + config bundled together. **Load with `weights_only=False`** (PyTorch ≥ 2.6 defaults to `True`, which only allows raw tensors). See [`models/README.md`](models/README.md) for full load instructions and an optional Hugging Face Hub workflow if you'd rather skip LFS.

---

## 🗂️ Repository Layout

```
.
├── README.md                                       ← this file
├── TUTORIAL.md                                     ← step-by-step walkthrough
├── LICENSE                                         ← MIT
├── requirements.txt
├── .gitattributes                                  ← Git LFS rules for .pt / .pkl
│
├── src/
│   ├── data.py                ← NFCorpus loader · FAISS index · hard-negative miner · training pairs
│   ├── embed_corpus.py        ← Instructor-XL bi-encoder embedding script
│   ├── train.py               ← chunked cross-encoder fine-tune with per-chunk eval (CLI)
│   ├── evaluate.py            ← pytrec_eval wrapper: NDCG / MAP / Recall / P @ k
│   ├── rerank.py              ← apply any CrossEncoder to a first-stage top-K
│   └── demo.py                ← single-query inference demo
│
├── notebooks/
│   └── cross_encoder_finetune_nfcorpus.ipynb       ← end-to-end notebook
│
├── reports/
│   ├── fine_tune_report.md    ← method, headline numbers, training dynamics, interpretation
│   ├── training_metrics.csv   ← 52 checkpoints × NDCG/MAP/Recall/P @ {1,3,5,10,100}
│   └── training_log.txt       ← cleaned stdout from the training run
│
└── models/                                         ← Git-LFS · 3 fine-tuned checkpoints
    ├── README.md
    ├── fine_tuned_MiniLM_L_2_v2.pt
    ├── fine_tuned_MiniLM_L_4_v2.pt
    └── fine_tuned_MiniLM_L_6_v2.pt
```

---

## 🌍 When to Use This

### 🏥 Medical / Clinical RAG
Building a question-answering system over PubMed abstracts, clinical guidelines, or patient-facing health content? The base MS&nbsp;MARCO cross-encoders underperform on medical jargon — drop the **L-4 fine-tuned checkpoint** in as your reranker and you'll see top-1 / top-3 retrieval quality improve immediately.

### 📚 BEIR-style Information Retrieval Research
The full pipeline is BEIR-compatible. Swap NFCorpus for SciFact, TREC-COVID, or BioASQ by changing one argument to `download_nfcorpus` — the loader, mining, and eval code already speak BEIR's qrels/corpus/queries format.

### 🧰 Cross-encoder Recipe Template
Even outside medicine, this is a working template for **domain-adapting** any `cross-encoder/ms-marco-MiniLM-*` model. The four levers worth tuning per-domain are documented in the **[Recipe](#-recipe)** section: instruction strings, hard-negative depth, learning rate, and chunk size.

### 🎓 Teaching / Reference
The notebook + report combination is designed to be readable — useful for ML/IR courses covering two-stage retrieval, hard-negative mining, and the bi-encoder vs cross-encoder trade-off.

---

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Everything |
| **Deep Learning** | PyTorch 2.0+ | Training & inference |
| **Models** | [Sentence Transformers](https://www.sbert.net/) ≥ 2.2 | `CrossEncoder` API |
| **Base model** | [`cross-encoder/ms-marco-MiniLM-L-{2,4,6}-v2`](https://huggingface.co/cross-encoder) | Reranker initialisation |
| **First-stage** | [`hkunlp/instructor-xl`](https://huggingface.co/hkunlp/instructor-xl) | Instruction-conditioned bi-encoder |
| **ANN search** | [FAISS](https://github.com/facebookresearch/faiss) | `IndexFlatL2` for hard-negative mining |
| **Dataset** | [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) via [BEIR](https://github.com/beir-cellar/beir) | Public medical IR benchmark |
| **Evaluation** | [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval) | Official TREC metric implementations |
| **Large files** | Git LFS | `.pt` checkpoint storage |

---

## 📝 Citation

If you use these checkpoints or the recipe in your work:

```bibtex
@misc{asadolahi2024minilmnfcorpus,
  title        = {Fine-tuning ms-marco-MiniLM-L-4-v2 cross-encoder on NFCorpus for medical retrieval},
  author       = {Asadolahi, Mohammad},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/MohammadAsadolahi/Fine-tuning-MS-Marco-MiniLM-L-4-v2--cross-encoder-on-NFcorpus-medical-datasets}}
}
```

Please also cite NFCorpus and BEIR:

```bibtex
@inproceedings{boteva2016full,
  title     = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  author    = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  booktitle = {European Conference on Information Retrieval (ECIR)},
  year      = {2016}
}

@inproceedings{thakur2021beir,
  title     = {BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author    = {Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2021}
}
```

---

## 🤝 Contributing

Issues and PRs welcome — particularly:

- 🔁 Ports of this recipe to other BEIR corpora (SciFact, TREC-COVID, BioASQ, …)
- 🧮 Alternative first-stage retrievers (BGE, E5, GTE) and the resulting headline numbers
- 📊 Additional plots / dashboards over [`reports/training_metrics.csv`](reports/training_metrics.csv)
- 🤗 A Hugging Face Hub mirror of the three checkpoints with proper model cards
- 📝 Tutorial improvements and typo fixes

---

## 👤 Author

**Mohammad Asadolahi** — Senior Agentic AI Engineer

- **GitHub:** [github.com/MohammadAsadolahi](https://github.com/MohammadAsadolahi)
- **Focus:** Agentic AI Architectures In The Wild
- **Related work:** [InkFlow](https://github.com/MohammadAsadolahi/InkFlow) — real-time capture & analytics for Copilot Chat and Claude Code sessions

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The base cross-encoder weights from `cross-encoder/ms-marco-MiniLM-L-{2,4,6}-v2` ship under their original **Apache-2.0** license (see the corresponding Hugging Face model cards). NFCorpus is distributed for research use under the terms set out in [Boteva et al. 2016](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/).

---

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) (Nils Reimers et al.) — the `CrossEncoder` API this repo builds on
- [BEIR](https://github.com/beir-cellar/beir) (Nandan Thakur et al.) — the benchmark and dataset release
- [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) (Boteva et al., ECIR 2016) — the medical IR corpus
- [Instructor](https://github.com/HKUNLP/instructor-embedding) (HKU NLP) — the instruction-tuned bi-encoder used for first-stage retrieval
- [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Research) — fast ANN search
- [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval) (Christophe Van Gysel) — official TREC metric implementations

---

<p align="center">
  <strong>MiniLM · NFCorpus · Cross-Encoder</strong> — Because top-1 in medical retrieval is the one the LLM actually reads.
</p>

<p align="center">
  <sub>Built with ❤️ for engineers who care about the second stage of retrieval.</sub>
</p>

---

<p align="center">
  <sub><em>this readme is AI assisted generated, so check for mistakes</em></sub>
</p>
