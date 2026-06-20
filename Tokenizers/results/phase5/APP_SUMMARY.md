# Phase 5 — Web app (FINAL, end-to-end working)

**Date:** 2026-06-12
**Status:** ✅ e2e test passes (4/4) and the Streamlit UI runs live.

## What it does
Enter a movie name → the Phase 4 crawler collects reviews from **TMDB + Letterboxd
+ Trakt** → each review is scored by the combined pipeline (sentiment + **our
movie-Reddit sarcasm model**) → the dashboard shows the verdict before AND after
sarcasm, the top reviews, and a per-source crawler report.

## Dashboard features (v2)
- **Two-stage verdict**: "1️⃣ Raw sentiment (first model only)" with confidence +
  pie, then "2️⃣ After sarcasm correction" with confidence + pie, side by side, plus
  a banner stating whether sarcasm **changed** the verdict.
- **Top reviews**: 👍 positive, 👎 negative (sarcastic ones flagged), and a new
  **🙄 Most sarcastic** section (ranked by sarcasm confidence). Shows 5 each, with
  the source of every review.
- **Per-source metrics** (counts per TMDB/Letterboxd/Trakt) + sarcastic %.
- **🕑 History tab**: the last 3 analyses of the session (movie, before/after
  verdicts + confidence, sarcastic count).
- **Crawler report**: each source shows the reason on failure (e.g. "no match or
  no reviews on TMDB", "rate-limited / blocked", "skipped — no key") instead of a
  bare "FAILED".

## Crawler robustness (v2)
- Each source now returns a **reason** string, surfaced in the dashboard.
- **TV-series fallback**: TMDB and Trakt try a movie match first, then a TV/show
  match — so series like *Game of Thrones* return reviews too.
- `fetch()` retries **up to 3 times with growing backoff** (and longer waits on
  403/429 rate-limits).

## Wiring (one approved code edit)
`combined_inference.py` `SARCASM_MODEL` was pointed at our winning model:
`phase3_sarcasm/checkpoints/movie_reddit_model` (BERT, F1 0.72). Sentiment uses the
ready IMDB model via the existing fallback (no change).

## End-to-end test (`test_pipeline_e2e.py`)
```
[PASS] App imports without errors
       Sentiment model: fabriceyhc/bert-base-uncased-imdb
       Sarcasm model:   .../phase3_sarcasm/checkpoints/movie_reddit_model
[PASS] Pipeline returns reviews — 146 reviews for 'Inception' (TMDB 8 + Letterboxd 66 + Trakt 80)
[PASS] Verdict is GOOD/BAD/MIXED — verdict=GOOD, confidence=66%
[PASS] Sarcasm stats present — 20 sarcastic reviews
SUMMARY: 4/4 checks passed — ✅ PASSED
```
Inception → **GOOD, 66% confidence**, 20/146 reviews flagged sarcastic (and
down-weighted to negative by the verdict logic).

## Running the live UI
```bash
export TMDB_API_KEY=<TMDB v3 key>
export TRAKT_CLIENT_ID=<Trakt app Client ID>
streamlit run phase5_app/app.py
```
Verified live on http://localhost:8502 (health endpoint returns "ok").

## Files
- `test_e2e_output.log` — end-to-end test output
- `streamlit_server.log` — Streamlit startup log
