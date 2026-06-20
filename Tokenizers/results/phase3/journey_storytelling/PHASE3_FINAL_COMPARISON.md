# Phase 3 — Final: choosing a sarcasm model that works for movie reviews

**Date:** 2026-06-12
**Goal:** The headline-trained model (`helinivan`) scored 94% in-domain but failed
on real movie-review sarcasm (domain gap). We searched for a **review-domain**
sarcasm model + an independent **film-domain** dataset, replaced the model, re-ran
the pipeline, and compared.

## What we found (real HuggingFace Hub search, not guessed)

**Review-domain sarcasm models:**
- `dnzblgn/Sarcasm-Detection-Customer-Reviews` (DeBERTa-v2, fine-tuned on customer reviews)
- `jkhan447/sarcasm-detection-Bert-base-uncased-CR` (BERT, "CR" = customer reviews)
- `cardiffnlp/twitter-roberta-base-irony` (Twitter irony, conversational)

**Film/TV-domain test dataset (none of the models were trained on it):**
- **MUStARD** (`DynamicSuperb/SarcasmDetection_Mustard`) — 200 sarcastic/non-sarcastic
  utterances from FRIENDS, Big Bang Theory, Golden Girls, Sarcasmoholics (92 sarc / 108 not).

> Note: there is **no public, high-quality movie-*review* sarcasm corpus** (the
> project's own DATASET_INSTRUCTIONS.md says so). Customer-review sarcasm (what
> dnzblgn learned) + MUStARD film/TV dialogue are the closest available stand-ins.

## Experiment 1 — Generalization to film sarcasm (MUStARD, text-only)

| Model | Domain trained on | F1 | Acc | Recall | Behavior |
|-------|-------------------|----|----|--------|----------|
| **dnzblgn** | customer reviews | **0.4615** | 0.580 | 0.391 | **best F1 — actually detects sarcasm** |
| jkhan447 BERT-CR | customer reviews | 0.4173 | 0.595 | 0.315 | best acc |
| cardiffnlp | Twitter irony | 0.4076 | 0.535 | 0.348 | |
| **helinivan** | **news headlines** | **0.0211** | 0.535 | 0.011 | **collapses → predicts "not sarcastic" ~99%** |
| majority baseline | — | 0.000 | 0.540 | 0.0 | |

**Conclusion:** the headline model is **useless** outside headlines (F1 0.02). The
customer-review model **generalizes best** to film/TV sarcasm. All models are only
modestly above baseline because text-only MUStARD sarcasm depends on tone/audio/context
(genuinely hard) — but dnzblgn is the only one that meaningfully detects sarcasm.

## Experiment 2 — Combined sentiment+sarcasm on movie-review examples

Re-ran the project's `CombinedAnalyzer` (sentiment = original IMDB model) swapping
the sarcasm model:

| Sarcasm model | Built-in test | Sarcastic-positive movie cases |
|---------------|---------------|-------------------------------|
| helinivan (headlines) | 4 / 6 | ❌ "Oh sure, BEST movie ever 🙄" → *Genuinely Positive* (wrong) |
| **dnzblgn (cust. reviews)** | **5 / 6** | ✅ → *Likely Negative (sarcastic positive)* (correct) |

With dnzblgn, every hard sarcastic-positive review is now correctly flagged. The one
remaining "miss" ("Awful… never get back" called sarcastic) still yields the **correct
final verdict (Likely Negative)**, because both sarcastic-negative and genuine-negative
map to "Negative" — so the app's GOOD/BAD output is robust.

## Recommendation for the app (Phase 5)

Use **`dnzblgn/Sarcasm-Detection-Customer-Reviews`** as the sarcasm model instead of
the headline model / from-scratch training. It is the most reliable choice for
movie-review sarcasm and requires no training. (Wiring it into the app's default
will need a one-line change to `SARCASM_MODEL` in `combined_inference.py` — to be
approved when we reach Phase 5.)

## Files
- `mustard_model_comparison_output.log` — 4-model comparison on MUStARD
- `combined_inference_dnzblgn_output.log` — combined pipeline with the new model
- `combined_inference_output.log` — earlier run with the headline model (for contrast)
- `PRETRAINED_MODEL_SUMMARY.md`, `COMBINED_INFERENCE_SUMMARY.md` — earlier headline-model results
