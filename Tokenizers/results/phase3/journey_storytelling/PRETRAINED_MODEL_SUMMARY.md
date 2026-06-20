# Phase 3 — Using a PRE-TRAINED sarcasm model (no training from scratch)

**Date:** 2026-06-12
**Decision:** Instead of training a BERT sarcasm classifier from scratch
(`train_sarcasm.py`), we looked for an existing fine-tuned model and re-tested it
on our data — mirroring the Phase 2 philosophy of measuring ready models.

**Model chosen:** `helinivan/english-sarcasm-detector`
- Architecture: `BertForSequenceClassification` (same base the project would have built).
- Fine-tuned on the **News Headlines Dataset for Sarcasm Detection** (Misra) — the
  exact dataset we use here.
- Labels: `LABEL_0` = not sarcastic, `LABEL_1` = sarcastic (verified on 4 known
  Onion/HuffPost headlines, all correct).

## Dataset
`raquiba/Sarcasm_News_Headline` (HF mirror of Misra v2) → written to
`phase3_sarcasm/data/Sarcasm_Headlines_Dataset_v2.json` (28,619 rows, 47.6% sarcastic).

## Results

| Evaluation set | Rows | Accuracy | Precision | Recall | F1 |
|----------------|------|----------|-----------|--------|----|
| **Full dataset** | 28,619 | **0.9455** | 0.9715 | 0.9124 | **0.9410** |
| Random sample (seed=42) | 1,066 | 0.9409 | 0.9505 | 0.9170 | 0.9335 |

### Confusion matrices (rows = true, cols = pred; 0 = not, 1 = sarcastic)
Full:
```
          pred 0   pred 1
  true 0  14620      365
  true 1   1194    12440
```
Sample:
```
          pred 0   pred 1
  true 0    561       23
  true 1     40      442
```

Baselines: random ~0.50 / majority ~0.52 — the model (0.945) is far above both.

## ⚠️ Important caveat — in-domain contamination

This model was **trained on the same dataset** we test it on, so these numbers are
in-domain and almost certainly optimistic. Unlike Phase 2 (IMDB model tested on a
*different* dataset = genuine cross-domain), here train and test share the same
distribution and likely overlap. Therefore:
- ~94% is an **upper bound**, not a realistic estimate of generalization.
- On real movie reviews (the app's actual input), expect lower performance,
  because the domain shifts from **news headlines → movie reviews**.

## Files
- `evaluate_pretrained_sarcasm_output.log` — full terminal output
