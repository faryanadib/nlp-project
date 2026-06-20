# Phase 3b — Fine-tuned movie sarcasm model vs ready models

**Date:** 2026-06-12
**Why:** Ready sarcasm models top out at F1 ~0.4 on movie/conversational sarcasm.
The user asked to *also* train a model on the real movie-Reddit data we collected
and compare, then pick the best for the app.

## Training setup (standalone script `train_movie_sarcasm.py`)
- Data: **9,527 unique** movie-subreddit comments from SARC (`marcbishara/sarcasm-on-reddit`,
  the danofer Kaggle dataset), subreddits: r/movies, DC_Cinematic, marvelstudios,
  TrueFilm, flicks, scifi, horror, … — balanced (4,831 not / 4,696 sarcastic).
- Split: train 6,858 / val 763 / **test 1,906** (stratified, seed 42, test saved to
  `phase3_sarcasm/data/reddit_movie_test.csv`).
- Model: `bert-base-uncased`, 3 epochs, lr 2e-5, max_len 128, batch 32 (MPS, ~19 min).
- Best checkpoint by val loss = **epoch 2** (val acc 0.71), saved to
  `phase3_sarcasm/checkpoints/movie_reddit_model`.

## Result — held-out movie test set (1,906 real movie comments)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **NEW BERT (movie Reddit)** ⭐ FINAL | **0.7130** | 0.6918 | 0.7529 | **0.7211** |
| NEW RoBERTa (movie Reddit) | 0.7151 | 0.7148 | 0.7018 | 0.7082 |
| dnzblgn (customer reviews) | 0.5367 | 0.5565 | 0.2939 | 0.3847 |
| helinivan (news headlines) | 0.5068 | 0.4959 | 0.0639 | 0.1132 |
| majority baseline | 0.5073 | — | — | 0.000 |

**The in-domain fine-tuned model nearly doubles F1 over the best ready model
(0.72 vs 0.38)** and is the only one well above the 0.51 baseline. This is the
reliable, in-domain result we were after.

### Does a stronger base model (RoBERTa) help? — No.

We also fine-tuned **RoBERTa-base** on the identical train/test split. Result:
acc 0.7151 / F1 0.7082 — **statistically tied with BERT** (BERT's F1 is even a
hair higher). Swapping to a stronger base model did **not** break the ~0.71–0.72
ceiling. The bottleneck is the **task/data**, not the model: text-only
conversational sarcasm lacks the tone / parent-comment context that disambiguates
it. To go higher we would need to add that **context**, not a bigger model.
→ **BERT is kept as the final model** (equal/better F1, lighter, already wired in).
RoBERTa checkpoint kept at `phase3_sarcasm/checkpoints/movie_reddit_roberta` as
evidence that "bigger base model didn't help".

## Combined inference (sentiment + sarcasm) on handcrafted movie examples

Both the new model and dnzblgn pass 5/6 of the built-in examples and both now
correctly flag sarcastic-positive reviews ("Oh sure, BEST movie ever 🙄" → Likely
Negative). The difference is the *type* of remaining error:

| Sarcasm model | The one miss | Harm to app verdict |
|---------------|--------------|---------------------|
| dnzblgn | calls a genuine *negative* ("Awful… never get back") sarcastic | none — verdict stays Negative |
| **NEW movie model** | calls a genuine *positive* ("Absolutely loved the soundtrack") sarcastic | flips a true Positive → Negative (worse) |

So the new model is slightly **over-eager** on enthusiastic positives, but on the
large realistic test set its overall accuracy is far higher.

## Recommendation
Use the **NEW movie-Reddit BERT model** as the app's sarcasm detector — it is
dramatically more accurate on real movie comments (F1 0.72). Keep dnzblgn as a
documented fallback. Its only weakness (occasionally flagging very enthusiastic
genuine praise as sarcastic) is minor compared with its overall gain.

To wire it into the app it must live at the path `combined_inference.py` expects
(`phase3_sarcasm/checkpoints/best_model`) or that constant must point to
`movie_reddit_model` — a one-line change to approve in Phase 5.

## Files
- `train_movie_sarcasm.py` — training + comparison script
- `train_movie_sarcasm_output.log` — full training + ranking log
- `combined_inference_movie_model_output.log` — combined pipeline with the new model
- `phase3_sarcasm/data/reddit_movie_test.csv` — held-out test set
- `phase3_sarcasm/checkpoints/movie_reddit_model/` — the trained checkpoint
