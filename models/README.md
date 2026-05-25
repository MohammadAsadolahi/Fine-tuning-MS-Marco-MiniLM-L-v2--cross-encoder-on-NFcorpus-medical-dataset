# Fine-tuned cross-encoder checkpoints

Three checkpoints from the runs documented in [`reports/fine_tune_report.md`](../reports/fine_tune_report.md):

| file | base model | size |
|---|---|---:|
| `fine_tuned_MiniLM_L_2_v2.pt` | [`cross-encoder/ms-marco-MiniLM-L-2-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2) | ~60 MB |
| `fine_tuned_MiniLM_L_4_v2.pt` | [`cross-encoder/ms-marco-MiniLM-L-4-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-4-v2) | ~74 MB |
| `fine_tuned_MiniLM_L_6_v2.pt` | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | ~87 MB |

The `*-L-4-v2.pt` checkpoint is the recommended production model — see the report.

## How they were saved

Each file is a Python `torch.save(model, path)` dump of a `sentence_transformers.CrossEncoder`
instance. The format is the standard PyTorch zip archive (you'll see `PK\x03\x04` at the
start with `xxd`).

## How to load

```python
import torch
ce = torch.load("models/fine_tuned_MiniLM_L_4_v2.pt", map_location="cpu", weights_only=False)
scores = ce.predict([("diabetes treatment", "Type 1 and 2 diabetes mellitus ...")])
```

> Note on `weights_only=False`: PyTorch ≥ 2.6 defaults to `weights_only=True`, which only
> deserialises tensors. These artifacts contain a pickled `CrossEncoder` object (model +
> tokenizer + config), so you have to opt in.

## Git LFS

These files are tracked through Git LFS (see [`.gitattributes`](../.gitattributes) at the
repo root). Install LFS once with:

```bash
git lfs install
```

before cloning or pushing, otherwise Git will store the raw blobs and you'll exceed
GitHub's 100 MB single-file limit on some clients.

If you don't want to use LFS, host the checkpoints on Hugging Face Hub instead:

```bash
huggingface-cli upload <your-org>/ms-marco-MiniLM-L-4-v2-nfcorpus \
    models/fine_tuned_MiniLM_L_4_v2.pt
```
