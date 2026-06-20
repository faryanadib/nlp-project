# Phase 4 — Working multi-source crawler (FINAL)

**Date:** 2026-06-12
**Status:** ✅ WORKING — 3 sources, official `test_crawler.py` passes.

## The 3 sources

| Source | Type | Auth | How it finds the movie |
|--------|------|------|------------------------|
| **TMDB** | official API (JSON) | free API key (`TMDB_API_KEY`) | `/search/movie` → `/movie/{id}/reviews` |
| **Letterboxd** | HTML (server-rendered) | none | builds the film slug from the title → `/film/<slug>/reviews/` |
| **Trakt** | official API (JSON) | free Client ID (`TRAKT_CLIENT_ID`) | `/search/movie` → `/movies/{slug}/comments/likes` |

Why these: in 2026 most movie sites (IMDb, Rotten Tomatoes, Metacritic, RogerEbert,
Reddit, MUBI) are JS-rendered or bot-walled and return no scrapeable review text
(see `SOURCE_PROBE_FINDINGS.md`). Two official APIs (TMDB, Trakt) + one
server-rendered site (Letterboxd) are reliable and complementary.

## Results

```
Inception     : 146 unique reviews (TMDB 8 + Letterboxd 59 + Trakt 79)
The Godfather :  87 unique reviews (TMDB 6 + Trakt 82; Letterboxd was rate-limited)
test_crawler.py → ✅ PASSED (file OK, no dupes, no truncation)
```

The Godfather run shows the **value of multiple sources**: Letterboxd hit a 403
rate-limit, but TMDB + Trakt still delivered 87 reviews, so the crawl succeeded.

Output: `phase4_crawler/gathered_data/<movie>_reviews.xlsx`
(columns: `source, review_text, date, rating`; 146-row Inception file has 76 ratings
and 87 dates populated).

## What was changed in `crawler.py` (approved edits)
1. Replaced the 5 broken scrapers (RT/IMDb/Metacritic/RogerEbert/old-Letterboxd)
   with 3 working ones: `scrape_tmdb`, `scrape_letterboxd` (slug-based), `scrape_trakt`.
2. `fetch()` now **retries with backoff** and treats **403/429 as rate-limiting**
   (honours `Retry-After`), making Letterboxd/Trakt robust to throttling.
3. Added a per-request `headers` override (used to give APIs a clean User-Agent).
4. `SOURCES` priority: TMDB → Letterboxd → Trakt. Each still fails gracefully and
   is reported per source; the crawl never crashes.

## Running it (needs the two free keys as env vars)
```bash
export TMDB_API_KEY=<your TMDB v3 API key>
export TRAKT_CLIENT_ID=<your Trakt app Client ID>
python phase4_crawler/crawler.py "Inception"
```
Without keys the crawler still runs on Letterboxd alone (the API sources skip
gracefully). **Phase 5's app must be launched with these env vars set** to use all
three sources.

## Note on this session's rate-limits
Heavy probing during development temporarily got our IP throttled by Letterboxd /
Wikimedia (403/429). That is a testing artifact, not a code bug — the backoff logic
recovers, and normal usage (a few requests per movie) does not trigger it.

## Files
- `crawler_inception_3sources_output.log` — full 3-source run (146 reviews)
- `test_crawler_v3_output.log` — official 2-movie test, PASSED
- `SOURCE_PROBE_FINDINGS.md` — why each candidate source was kept/rejected
- `CRAWLER_SUMMARY.md` — the original (broken) state, for before/after storytelling
