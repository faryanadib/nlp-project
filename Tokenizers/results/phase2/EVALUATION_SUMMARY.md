# Phase 2 — Evaluation of the existing model on new data (NO fine-tuning)

**Date:** 2026-06-12
**Decision:** We deliberately **skipped fine-tuning**. The goal is to measure how
the existing IMDB-fine-tuned model (`fabriceyhc/bert-base-uncased-imdb`) performs
on a *different-domain* dataset, so we have a realistic estimate of the accuracy
our app will have when it crawls real reviews from various sources. Re-fine-tuning
would defeat that purpose.

**How:** `evaluate.py` accepts `--model`, so no code change was needed:
```
python phase2_new_dataset/evaluate.py --model fabriceyhc/bert-base-uncased-imdb
```

## Results on Rotten Tomatoes test set (1,066 reviews)

| Metric | Value |
|--------|-------|
| Accuracy | **0.8096** |
| Precision | 0.7603 |
| Recall | 0.9043 |
| **F1** | **0.8260** |

### Confusion matrix (0 = neg, 1 = pos)
```
            pred 0   pred 1
  true 0     381      152
  true 1      51      482
```

### Baselines (accuracy / F1)
| Predictor | Acc / F1 |
|-----------|----------|
| random | 0.5131 / 0.5136 |
| majority class | 0.5000 / 0.0000 |
| **our model** | **0.8096 / 0.8260** |

Prediction class share: neg 40.5% / pos 59.5%

## Interpretation

- The IMDB model **transfers well** to the Rotten Tomatoes domain: 81% accuracy,
  far above the 50–51% baselines.
- It is **biased toward positive** (high recall 0.90, lower precision 0.76): it
  predicts "positive" 59.5% of the time and misclassifies 152/533 true-negatives
  as positive.
- **Practical takeaway for the app:** expect ~81% accuracy on crawled reviews,
  with a tendency to over-call negatives as positive. This directly motivates
  **Phase 3 (sarcasm detection)** — subtly-negative / sarcastic reviews are
  exactly the ones this model gets wrong.
- The overall "FAILED" line comes only from re-running the Phase 1 calibration
  check (the same known overconfidence). The actual Phase 2 metrics pass:
  F1 0.826 > 0.75 threshold, no class collapse.

## Files
- `evaluate_original_model_output.log` — full terminal output
- `DATASET_SUMMARY.md` — dataset details
