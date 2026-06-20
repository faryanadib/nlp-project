# Phase 4 — Source probe findings (why we picked these 3 sources)

**Date:** 2026-06-12
We probed many movie sites live. Most are now JS-rendered (React/Next.js) or
anti-bot, so their review text can't be scraped from the raw HTML. Working sources:

| Source | Result | Key? | Used? |
|--------|--------|------|-------|
| **Letterboxd** | `/film/<slug>/reviews/` returns real user reviews in server-rendered HTML | no | ✅ |
| **Wikipedia** | "Critical reception" section via the MediaWiki API (21 paras for Inception) | no | ✅ |
| **TMDB** | `/movie/{id}/reviews` clean JSON user reviews (401 without key → works with one) | free key | ✅ |
| Rotten Tomatoes | old `napi/search` endpoint 404; new pages JS-rendered (0 review markers) | — | ❌ |
| IMDb | reviews page returns HTTP 202 empty (bot wall / JS) | — | ❌ |
| Metacritic | HTTP 200 but review text is JS-loaded; no `__NEXT_DATA__`, no text selectors | — | ❌ |
| RogerEbert | review pages server-rendered but search is JS + aggressive rate-limiting (403) | — | ❌ |
| Common Sense Media | HTTP 200 but review text JS-loaded | — | ❌ |
| Reddit (r/movies) | 403 even with a browser UA — now requires OAuth | — | ❌ |
| MUBI | HTTP 200 but reviews JS-loaded | — | ❌ |

## Design decisions
- **Letterboxd**: search is JS-rendered, but film URLs are predictable
  (`/film/<lowercased-hyphenated-title>/`), so we build the slug directly instead
  of scraping search. Paginated, polite delay to avoid 403.
- **Wikipedia**: use the API (search → sections → parse the "Critical reception"
  section) — robust and key-free; gives critic-review prose.
- **TMDB**: official API, free key via env var `TMDB_API_KEY`; best-quality user
  reviews. Gracefully skipped if the key is absent.

All three fail gracefully and report per-source, exactly like the original design.
