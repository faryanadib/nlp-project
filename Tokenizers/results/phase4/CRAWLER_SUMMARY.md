# Phase 4 — Multi-source review crawler

**Date:** 2026-06-12
**Result:** ❌ 0 reviews collected — crawler runs without crashing, but every source
returns nothing because the live sites have changed since the scrapers were written.

## What we ran
```
python phase4_crawler/crawler.py "Inception"      # → 0 reviews, file written, FAILED
python phase4_crawler/test_crawler.py             # Inception + The Godfather → FAILED
```

## Root-cause diagnosis (probe of each source)

The failure is **NOT a bug in the crawler logic** and **NOT (mostly) network
blocking** — the sites are reachable (HTTP 200), but their HTML/endpoints changed:

| Source | HTTP | Diagnosis |
|--------|------|-----------|
| Letterboxd | 200 (later 4xx) | film-link CSS selector matches nothing — HTML changed; rate-limits on repeat |
| IMDb (users) | 200 (suggest API works) | reviews-page selectors outdated |
| Metacritic | 200 | review-block selectors outdated |
| RogerEbert.com | 200 (later HTTPError) | review selectors outdated; rate-limits on repeat |
| Rotten Tomatoes | **404** | the `napi/search/all` endpoint no longer exists |

The crawler's error handling works exactly as designed: broken sources are skipped
and reported per-source, nothing crashes, an (empty) Excel file is still written.

## Why this happens

Web scraping is inherently brittle (the README warns about this explicitly). The
scrapers were written against a snapshot of each site's HTML; sites have since
restructured their markup, removed/renamed API endpoints (Rotten Tomatoes), and
added stricter bot rate-limiting (Letterboxd, RogerEbert). Reviving them would
require rewriting the CSS selectors for each source — brittle work with no
guarantee, and some sites (IMDb, RT) actively block automated clients.

## Impact on Phase 5
The Streamlit app imports this crawler to fetch live reviews. With 0 reviews it
has nothing to analyze. To demo the end-to-end pipeline (Phase 5) we will need a
small **sample reviews file** to stand in for live crawl output, OR to repair at
least one scraper. Decision pending with the user.

## Files
- `crawler_inception_output.log` — single-movie crawl
- `test_crawler_output.log` — 2-movie sanity test
- (empty) `phase4_crawler/gathered_data/*.xlsx` were written
