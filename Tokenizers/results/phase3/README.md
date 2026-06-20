# Phase 3 — Sarcasm detection for movie reviews (READ THIS FIRST)

## ⭐ FINAL RESULT (present this to the professor)

We built a sarcasm detector that actually works on movie-domain text by
fine-tuning BERT on **9,527 real r/movies (Reddit) comments**.

| Sarcasm model | F1 on held-out movie test (1,906) |
|---------------|-----------------------------------|
| **BERT fine-tuned on movie Reddit** ⭐ | **0.7211** |
| RoBERTa fine-tuned on movie Reddit | 0.7082 (no gain — task ceiling) |
| dnzblgn (best ready model, customer reviews) | 0.3847 |
| helinivan (news-headline model) | 0.1132 |
| baseline | 0.000 |

**Headline / ready models fail on movie sarcasm; our in-domain fine-tune nearly
doubles F1.** A bigger base model (RoBERTa) did not help — the ceiling is the lack
of conversational context, not the model size.

**Final model:** `phase3_sarcasm/checkpoints/movie_reddit_model/` (BERT).

## Files at this level = FINAL deliverables
- `README.md` — this file
- `MOVIE_MODEL_VS_READY.md` — the full final comparison (BERT vs RoBERTa vs ready models)
- `train_movie_sarcasm.py` — training code (BERT/RoBERTa via env var)
- `train_movie_sarcasm_output.log` — BERT training + comparison
- `train_movie_roberta_output.log` — RoBERTa training + comparison
- `combined_inference_movie_model_output.log` — sentiment+sarcasm pipeline with the final model

## `journey_storytelling/` = the process (good for telling the story, not the final answer)
How we got here, step by step:
1. `PRETRAINED_MODEL_SUMMARY.md` — tried a news-headline model: 94% in-domain but…
2. `COMBINED_INFERENCE_SUMMARY.md` — …it failed on real movie-review sarcasm (domain gap).
3. `PHASE3_FINAL_COMPARISON.md` — switched to a customer-review model (dnzblgn); better, still weak on film.
4. `REDDIT_SARCASM_COMPARISON.md` — found the SARC Reddit data has a real r/movies slice.
5. `mustard_model_comparison_output.log`, `reddit_comparison_output.log`,
   `combined_inference_*_output.log` — the supporting runs.

→ which led to fine-tuning on the movie Reddit data = the FINAL result above.
