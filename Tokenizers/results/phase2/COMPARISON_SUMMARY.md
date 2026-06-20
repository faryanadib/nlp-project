# Phase 2 — Full vs Sample Comparison (existing IMDB model, NO fine-tuning)

**Date:** 2026-06-12
**Model:** `fabriceyhc/bert-base-uncased-imdb` (used as-is, no training)
**Dataset:** Rotten Tomatoes (Cornell Movie Review) — see `phase2_new_dataset/data/DATASET_SOURCE.md`

We evaluated the same ready-made model three ways to get a robust real-world
accuracy estimate for the app, and to make the numbers directly comparable.

## Results

| Evaluation set | Rows | Accuracy | Precision | Recall | F1 | pos-share |
|----------------|------|----------|-----------|--------|----|-----------|
| Original test split | 1,066 | 0.8096 | 0.7603 | 0.9043 | 0.8260 | 59.5% |
| **Full dataset** | **10,662** | **0.8011** | 0.7527 | 0.8968 | **0.8185** | 59.6% |
| Random sample (seed=42) | 1,066 | 0.7824 | 0.7378 | 0.8645 | 0.7961 | 57.6% |

### Confusion matrices (rows = true, cols = pred; 0 = neg, 1 = pos)

Full dataset (10,662):
```
          pred 0   pred 1
  true 0    3760     1571
  true 1     550     4781
```
Random sample (1,066):
```
          pred 0   pred 1
  true 0     381      161
  true 1      71      453
```

## Interpretation

- **Stable ~80% accuracy** across all three views (78.2% – 81.0%). The full-dataset
  number (80.1%) is the most reliable single estimate; the random 1,066 sample
  (78.2%) confirms it on a slice the same size as the original IMDB-comparison run.
- **Consistent positive bias** everywhere: the model predicts "positive" ~58–60%
  of the time and misses many true-negatives (1,571 / 5,331 = 29% of negatives
  flipped to positive on the full set). High recall (~0.90), lower precision (~0.75).
- **Practical takeaway:** when the app crawls real reviews, expect ~80% accuracy
  with a lean toward calling things positive. Subtle / sarcastic negative reviews
  are the main failure mode — exactly what **Phase 3** targets.

## Files
- `evaluate_full_and_sample_output.log` — full terminal output
- `evaluate_original_model_output.log` — original 1,066 test-split run
- data: `rotten_tomatoes_full.csv`, `rotten_tomatoes_sample_1066.csv`, `DATASET_SOURCE.md`
