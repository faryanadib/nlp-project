# Phase 1 — Sanity Check Results

**Date:** 2026-06-12
**Model:** `fabriceyhc/bert-base-uncased-imdb`
**Device:** MPS (Apple Silicon)
**Environment:** Python 3.11.15, torch 2.12.0, transformers 5.11.0

## Result: 7 / 8 checks PASSED

| # | Check | Status | Detail |
|---|-------|--------|--------|
| 1 | Output shape & type | ✅ PASS | shape (6, 2) OK, prob sums within 1e-4 |
| 2 | Prediction direction | ✅ PASS | all 6 clear-cut sentences correct |
| 3 | Edge cases | ✅ PASS | all 4 edge cases handled, no crash/NaN |
| 4 | Confidence (neutral<clear) | ✅ PASS | clear 1.000 vs neutral 0.980 |
| 5 | Reproducibility | ✅ PASS | identical logits across 3 runs |
| 6 | Probability calibration | ❌ FAIL | conf mean 0.993, range 0.052 — overconfident |
| 7 | Batch consistency | ✅ PASS | max logit deviation 4.77e-07 |
| 8 | Tokenization (trunc/pad) | ✅ PASS | truncated to 512, padded batch (2,512) |

## Interpretation

The single failure (**Probability calibration**) is a **genuine property of the
pre-trained model**, not a code bug. `fabriceyhc/bert-base-uncased-imdb` produces
near-certain probabilities (mean confidence 0.993) on almost every input, so the
spread of confidences falls below the 0.15 threshold the check requires. The
`pytest` run reproduces the identical outcome (1 failed, 7 passed), confirming the
test harness itself is correct and deterministic.

This is exactly the kind of issue Phase 2 (fine-tuning on a new dataset) is meant
to probe — whether a freshly fine-tuned model is better calibrated.

## Files
- `sanity_check_output.log` — full terminal output of the CLI script
- `pytest_output.log` — full terminal output of the pytest run
