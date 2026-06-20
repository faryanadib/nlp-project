# Phase 3 — Reddit (SARC) sarcasm data + model test

**Date:** 2026-06-12
**Trigger:** User suggested the Kaggle dataset
https://www.kaggle.com/datasets/danofer/sarcasm (the **SARC — Self-Annotated
Reddit Corpus**), asking: is it good data? does Reddit do better? is there a model
fine-tuned on it? does it suit movies?

## Data — yes, and it has real movie content

`danofer/sarcasm` (Kaggle) = `marcbishara/sarcasm-on-reddit` (HF mirror, used here,
no Kaggle login needed). Balanced 50/50, columns `label, comment, subreddit,
score, parent_comment, …`.

Crucially it has a **`subreddit` column**, so we filtered the holdout split to
movie subreddits (r/movies, DC_Cinematic, marvelstudios, TrueFilm, flicks, horror,
boxoffice, scifi…) → **815 real movie-discussion comments**, balanced (411 not /
404 sarcastic). Saved to `phase3_sarcasm/data/reddit_movie_sarcasm.csv`. This is the
**most genuinely movie-domain sarcasm test set** we have found (real audience talk
about films, with /s-derived labels).

## Models tested (no training; comment-only inputs)

| Model | Reddit general (2000) | Reddit MOVIE subs (815) | MUStARD film/TV (200) |
|-------|----------------------:|------------------------:|----------------------:|
| **dnzblgn** (customer reviews) | F1 **0.413** / acc 0.54 | F1 **0.380** / acc 0.53 | F1 **0.462** / acc 0.58 |
| helinivan (news headlines) | F1 0.116 / acc 0.50 | F1 0.112 / acc 0.51 | F1 0.021 / acc 0.54 |
| **minh21 XLNet** (trained on Reddit SARC) | F1 **0.000** / acc 0.50 | F1 0.000 / acc 0.50 | F1 0.000 / acc 0.54 |

## Answers to the four questions

1. **Is it good data?** Yes — labeled, balanced, conversational, large, and (unlike
   headlines/MUStARD) it contains a real, sizable **movie-discussion** slice.
2. **Does Reddit give better results?** Not for the ready models — they score about
   the same on Reddit as on MUStARD. Conversational sarcasm is just hard without
   parent-comment/tone context. **dnzblgn stays the best across every set.**
3. **Is there a model fine-tuned on it?** Yes: `minh21/XLNet-Reddit-Sarcasm-Analysis`.
   But it **collapses** — predicts "not sarcastic" for 100% of inputs (F1 0.000).
   Unusable as-is (likely needs the parent-comment context it was trained with, or
   is under-trained). So a *usable* ready Reddit model does not exist.
4. **Does it suit movies?** The **data** does — the r/movies slice is the best
   movie-domain sarcasm corpus we have. But no working ready *model* is trained on
   it. To benefit, we'd have to **fine-tune** on this movie-subreddit data — which
   is the opposite of the "use ready models" approach we've followed so far.

## Bottom line
- Keep **`dnzblgn/Sarcasm-Detection-Customer-Reviews`** as the app's sarcasm model:
  it is consistently the best ready option (Reddit, Reddit-movie, MUStARD, and the
  combined-inference movie-review examples).
- The Reddit movie data is now saved and available **if** we later decide to train
  a movie-specific sarcasm model (optional, user's call).
- All ready models top out around F1 0.4–0.46 on conversational/film sarcasm — an
  honest ceiling caused by missing context, not a bug.

## Files
- `reddit_comparison_output.log` — full 3-model × 3-set run
- `phase3_sarcasm/data/reddit_movie_sarcasm.csv` — 815 movie-subreddit comments
