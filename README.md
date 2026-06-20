# Movie Review Analyzer — Sentiment + Sarcasm Pipeline

A five-phase NLP project that, given a movie (or TV-series) name, crawls real
reviews from the web and judges whether the title is **GOOD / BAD / MIXED** by
combining a **sentiment** model with a **movie-domain sarcasm** model — so a review
that *sounds* positive but is sarcastic is correctly counted as negative.

> The full project lives in [`Tokenizers/`](Tokenizers). (The `Week01`–`Week05`
> folders are earlier course exercises.)

> **Guiding principle:** prefer ready, pre-trained models and only fine-tune when
> they genuinely fall short — so the numbers reflect what a real app would actually
> get, not an over-fitted lab score.

---

## Setup

The system Python (3.14) has no PyTorch wheels, so use **Python 3.11**:

```bash
cd Tokenizers
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The crawler's two API sources need free keys, passed as environment variables:

```bash
export TMDB_API_KEY=<your TMDB v3 API key>        # themoviedb.org → Settings → API
export TRAKT_CLIENT_ID=<your Trakt app Client ID> # trakt.tv/oauth/applications → New Application
```

Letterboxd needs no key. Without the keys the crawler still runs on Letterboxd
alone (the API sources skip gracefully). All commands below assume you are inside
`Tokenizers/`.

---

## Phase 1 — Sanity-check the sentiment model

Model: `fabriceyhc/bert-base-uncased-imdb` (BERT fine-tuned on IMDB).

```bash
python phase1_sanity_check/sanity_check.py
pytest phase1_sanity_check/test_sanity.py -v      # same 8 checks as unit tests
```

**Result:** 7/8 pass. The one failure — *probability calibration* — is a **real
property of the model** (it is over-confident, mean confidence 0.993), not a bug.
→ [`Tokenizers/results/phase1/`](Tokenizers/results/phase1)

---

## Phase 2 — How well does the ready model do on a *different* dataset?

Dataset: **Rotten Tomatoes** (`cornell-movie-review-data/rotten_tomatoes`).
We **deliberately did NOT fine-tune** — the goal is to estimate the real-world
accuracy the app will have on freshly-crawled reviews.

```bash
python phase2_new_dataset/download_and_prepare.py
python phase2_new_dataset/evaluate.py --model fabriceyhc/bert-base-uncased-imdb
```

**Result:** ~**80% accuracy / F1 0.82** (full 10,662-review set). The model transfers
well but is **biased toward positive** and misses subtle/sarcastic negatives — which
is exactly what Phase 3 addresses. → [`Tokenizers/results/phase2/`](Tokenizers/results/phase2)

---

## Phase 3 — A sarcasm model that actually works on movie reviews

Most ready sarcasm models are trained on **news headlines** and collapse on
conversational movie sarcasm, so we fine-tuned our own on real movie-discussion data.

| Sarcasm model | Domain | F1 on held-out movie test |
|---------------|--------|---------------------------|
| `helinivan/english-sarcasm-detector` | news headlines | 0.11 (collapses) |
| `dnzblgn/Sarcasm-Detection-Customer-Reviews` | customer reviews | 0.38 (best ready) |
| **Our BERT, fine-tuned on r/movies Reddit** | **movie comments** | **0.72** ⭐ |
| RoBERTa, same data | movie comments | 0.71 (no gain — task ceiling) |

Training data: **9,527 movie-subreddit comments** from the SARC corpus
(`marcbishara/sarcasm-on-reddit`), filtered by subreddit.

```bash
python results/phase3/train_movie_sarcasm.py             # reproduce the winning BERT model
python phase3_sarcasm/combined_inference.py --test       # sentiment + sarcasm pipeline
```

**Final sarcasm model:** `phase3_sarcasm/checkpoints/movie_reddit_model` (BERT).
→ [`Tokenizers/results/phase3/`](Tokenizers/results/phase3) (final) and its
`journey_storytelling/` subfolder (how we got there)

---

## Phase 4 — Multi-source review crawler

Most movie sites are now JS-rendered or bot-walled, so we crawl **three reliable
sources**:

| Source | Type | Auth |
|--------|------|------|
| **TMDB** | official API (JSON) | free `TMDB_API_KEY` |
| **Letterboxd** | server-rendered HTML (slug-based) | none |
| **Trakt** | official API (JSON) | free `TRAKT_CLIENT_ID` |

```bash
python phase4_crawler/crawler.py "Inception"
python phase4_crawler/test_crawler.py
```

Robustness: requests **retry up to 3× with growing backoff** (403/429 = rate-limit);
each source reports a **reason** on failure; TMDB and Trakt fall back from a movie
match to a **TV-series** match (so *Game of Thrones* works). Example — Inception:
146 reviews (TMDB 8 + Letterboxd 66 + Trakt 80). → [`Tokenizers/results/phase4/`](Tokenizers/results/phase4)

---

## Phase 5 — Web app (Streamlit)

```bash
streamlit run phase5_app/app.py                  # needs TMDB_API_KEY + TRAKT_CLIENT_ID
python phase5_app/test_pipeline_e2e.py           # headless end-to-end check (4/4 pass)
```

The dashboard shows the verdict **before** and **after** sarcasm correction (each
with confidence + a pie chart), the top **positive / negative / most-sarcastic**
reviews, a per-source **crawler report**, and a **History tab** with recent
analyses. Example — Inception → **GOOD, 66% confidence**, 20/146 reviews sarcastic.
→ [`Tokenizers/results/phase5/`](Tokenizers/results/phase5)

---

## Order of execution
Phases run in order 1 → 2 → 3 → 4 → 5. Phase 3 uses the IMDB sentiment model
(Phase 2) + our movie sarcasm model; Phase 5 imports Phases 3 + 4.

## Key models & datasets
- Sentiment: `fabriceyhc/bert-base-uncased-imdb` (used as-is)
- Sarcasm: **our** `Tokenizers/phase3_sarcasm/checkpoints/movie_reddit_model` (BERT fine-tuned on r/movies)
- Phase-2 data: `cornell-movie-review-data/rotten_tomatoes`
- Phase-3 data: `marcbishara/sarcasm-on-reddit` (SARC), filtered to movie subreddits

> Models and datasets are not committed (regenerate them with the scripts above); see
> [`Tokenizers/README.md`](Tokenizers/README.md) for the in-folder version of this guide.
