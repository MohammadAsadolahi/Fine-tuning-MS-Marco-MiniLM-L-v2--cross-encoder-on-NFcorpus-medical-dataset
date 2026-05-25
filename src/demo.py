"""
Quick demo: score a single query against a few candidate medical documents using either
the base cross-encoder or our fine-tuned model.

    python src/demo.py --model cross-encoder/ms-marco-MiniLM-L-4-v2
    python src/demo.py --model models/ms-marco-MiniLM-L-4-v2-nfcorpus
"""

from __future__ import annotations

import argparse

from sentence_transformers import CrossEncoder


QUERY = "diabetes treatment"

ARTICLES = [
    "Type 1 and 2 diabetes mellitus: A review on current cure approach and gene therapy as potential intervention. "
    "Type 1 and type 2 diabetes mellitus is a serious and lifelong condition commonly characterised by abnormally "
    "elevated blood glucose levels due to a failure in insulin production or a decrease in insulin sensitivity.",
    "Diabetes mellitus and its chronic complications. Diabetes mellitus is a major cause of morbidity and mortality, "
    "and it is a major risk factor for early onset of coronary heart disease. Complications of diabetes are "
    "retinopathy, nephropathy, and peripheral neuropathy.",
    "Diagnosis and Management of Central Diabetes Insipidus in Adults. Central diabetes insipidus (CDI) is a clinical "
    "syndrome which results from loss or impaired function of vasopressinergic neurons in the hypothalamus.",
    "Adipsic diabetes insipidus. Adipsic diabetes insipidus (ADI) is a rare but devastating disorder of water balance "
    "with significant associated morbidity and mortality.",
    "Nephrogenic diabetes insipidus: a comprehensive overview. Nephrogenic diabetes insipidus (NDI) is characterized "
    "by the inability to concentrate urine that results in polyuria and polydipsia.",
    "Impact of Salt Intake on the Pathogenesis and Treatment of Hypertension. Excessive dietary salt intake is "
    "associated with an increased risk for hypertension and other cardiovascular pathologies.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-4-v2")
    args = p.parse_args()

    model = CrossEncoder(args.model)
    pairs = [[QUERY, a] for a in ARTICLES]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, ARTICLES), key=lambda x: -x[0])
    print(f"\nQuery: {QUERY}\nModel: {args.model}\n")
    for s, a in ranked:
        print(f"score={s:+.4f}  {a[:120]}...")


if __name__ == "__main__":
    main()
