"""
Phase 4 — Multi-Source Movie Review Crawler.

Given a movie name, scrapes individual review texts (not aggregate scores)
from a prioritized list of sources, deduplicates them, and saves an Excel
file to gathered_data/{movie_name}_reviews.xlsx with columns:
    source, review_text, date, rating

Strategy:
  * Sources are tried in priority order (most reviews per page first).
  * SPEED: every HTTP request has a REQUEST_TIMEOUT-second budget; a source
    that times out or errors is skipped and reported as failed.
  * COVERAGE: crawling stops once TARGET_REVIEWS unique reviews are collected
    or all sources are exhausted. Pagination is followed up to MAX_PAGES.

NOTE: web scraping is inherently brittle — sites change their HTML and may
rate-limit or block automated clients. Every scraper here fails gracefully
(returns what it got) and the summary reports failures per source.

Usage:
    python crawler.py "Inception"
"""

import argparse
import os
import re
import sys
import time
from urllib.parse import quote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Configuration ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "gathered_data")

REQUEST_TIMEOUT = 15        # seconds — skip a source if it is slower than this
TARGET_REVIEWS = 100        # stop crawling once we have this many unique reviews
MAX_PAGES = 5               # pagination depth per source
MIN_REVIEW_CHARS = 20       # discard shorter fragments
NEAR_DUP_PREFIX = 120       # chars of normalized text used as near-dup signature
POLITE_DELAY = 1.0          # seconds between requests to the same site

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def fetch(url, as_json=False, retries=3, headers=None):
    """GET a URL with the global timeout, retrying several times with an
    increasing backoff (rate-limits get a longer wait). `headers` overrides session
    headers for this request. Returns soup/json or None on failure."""
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            if r.status_code in (403, 429) and attempt < retries:
                # rate-limited (Letterboxd uses 403): honour Retry-After else back off
                wait = int(r.headers.get("Retry-After", 0)) or 5 * (attempt + 1)
                time.sleep(min(wait, 20))
                continue
            r.raise_for_status()
            return r.json() if as_json else BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))   # 2s, 4s, 6s …
                continue
            print(f"    ! request failed: {url[:80]} ({type(e).__name__})")
            return None


def clean(text):
    """Collapse whitespace in a review snippet."""
    return re.sub(r"\s+", " ", text or "").strip()


# ── Source scrapers ────────────────────────────────────────────────────────────
# Each scraper takes (movie_name, max_reviews) and returns a list of dicts:
#   {"source", "review_text", "date", "rating"}
#
# As of 2026 most movie sites are JS-rendered or bot-walled (see
# results/phase4/SOURCE_PROBE_FINDINGS.md). The three sources below were verified
# to return real review text: Letterboxd (HTML), TMDB (API), Wikipedia (API).

def letterboxd_slug(name):
    """Letterboxd film slugs are the lowercased title joined with hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def scrape_letterboxd(movie, max_reviews):
    """Letterboxd user reviews via the predictable /film/<slug>/reviews/ URL.

    Letterboxd's search page is JS-rendered, so instead of scraping search we
    build the slug from the movie name and hit the (server-rendered) reviews page.
    Returns (reviews, reason).
    """
    out = []
    slug = letterboxd_slug(movie)
    reason = "no reviews found (title/slug may not match)"
    for page in range(1, MAX_PAGES + 1):
        soup = fetch(f"https://letterboxd.com/film/{slug}/reviews/by/activity/page/{page}/")
        if not soup:
            if page == 1:
                reason = "rate-limited / blocked (HTTP error)"
            break
        blocks = soup.select("div.js-review-body, .body-text")
        if not blocks:
            break
        for b in blocks:
            text = clean(b.get_text(" "))
            container = b.find_parent(class_=re.compile("film-detail")) or b.parent
            rating = container.select_one("span.rating") if container else None
            date = container.select_one("span._nobr, time") if container else None
            if text:
                out.append({"source": "Letterboxd", "review_text": text,
                            "date": clean(date.get_text()) if date else None,
                            "rating": clean(rating.get_text()) if rating else None})
            if len(out) >= max_reviews:
                return out, "ok"
        time.sleep(POLITE_DELAY)
    return (out, "ok") if out else (out, reason)


def scrape_tmdb(movie, max_reviews):
    """TMDB user reviews via the official API — tries a movie match first, then a
    TV-series match (so e.g. 'Game of Thrones' still works). Requires a free
    TMDB_API_KEY env var. Returns (reviews, reason)."""
    out = []
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        return out, "skipped — no TMDB_API_KEY set"
    for kind in ("movie", "tv"):
        data = fetch(f"https://api.themoviedb.org/3/search/{kind}?api_key={key}"
                     f"&query={quote(movie)}", as_json=True)
        if data is None:
            return out, "request failed (rate-limited / HTTP error)"
        results = data.get("results", [])
        if not results:
            continue
        media_id = results[0]["id"]
        for page in range(1, MAX_PAGES + 1):
            rev = fetch(f"https://api.themoviedb.org/3/{kind}/{media_id}/reviews"
                        f"?api_key={key}&page={page}", as_json=True)
            items = (rev or {}).get("results", [])
            if not items:
                break
            for r in items:
                text = clean(r.get("content", ""))
                ar = (r.get("author_details") or {}).get("rating")
                if text:
                    out.append({"source": "TMDB", "review_text": text,
                                "date": (r.get("created_at") or "")[:10] or None,
                                "rating": str(ar) if ar is not None else None})
                if len(out) >= max_reviews:
                    return out, "ok"
            if page >= (rev or {}).get("total_pages", 1):
                break
            time.sleep(POLITE_DELAY)
        if out:
            return out, f"ok ({'TV series' if kind == 'tv' else 'movie'})"
    return out, "no match or no reviews on TMDB"


def scrape_trakt(movie, max_reviews):
    """Trakt user comments via the official API — tries a movie match first, then a
    TV-show match. Requires a free TRAKT_CLIENT_ID env var (register an app at
    trakt.tv/oauth/applications). Returns (reviews, reason)."""
    out = []
    cid = os.environ.get("TRAKT_CLIENT_ID")
    if not cid:
        return out, "skipped — no TRAKT_CLIENT_ID set"
    hdr = {"Content-Type": "application/json",
           "trakt-api-version": "2", "trakt-api-key": cid}
    for kind in ("movie", "show"):
        data = fetch(f"https://api.trakt.tv/search/{kind}?query={quote(movie)}&limit=1",
                     as_json=True, headers=hdr)
        if data is None:
            return out, "request failed (rate-limited / HTTP error)"
        if not data:
            continue
        slug = ((data[0].get(kind) or {}).get("ids") or {}).get("slug")
        if not slug:
            continue
        for page in range(1, MAX_PAGES + 1):
            comments = fetch(f"https://api.trakt.tv/{kind}s/{slug}/comments/likes"
                             f"?page={page}&limit=20", as_json=True, headers=hdr)
            if not comments:
                break
            for c in comments:
                text = clean(c.get("comment", ""))
                if text:
                    out.append({"source": "Trakt", "review_text": text,
                                "date": (c.get("created_at") or "")[:10] or None,
                                "rating": str(c["user_rating"]) if c.get("user_rating") else None})
                if len(out) >= max_reviews:
                    return out, "ok"
            time.sleep(POLITE_DELAY)
        if out:
            return out, f"ok ({'TV show' if kind == 'show' else 'movie'})"
    return out, "no match or no comments on Trakt"


# Priority order: most reviews per movie first.
SOURCES = [
    ("TMDB",       scrape_tmdb),
    ("Letterboxd", scrape_letterboxd),
    ("Trakt",      scrape_trakt),
]


# ── Dedup & pipeline ───────────────────────────────────────────────────────────
def normalize(text):
    """Lowercase, strip punctuation/whitespace — used for near-dup detection."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower())[:NEAR_DUP_PREFIX]


def deduplicate(reviews):
    """Remove exact and near-exact duplicates (same normalized prefix)."""
    seen, unique = set(), []
    for r in reviews:
        sig = normalize(r["review_text"])
        if sig and sig not in seen:
            seen.add(sig)
            unique.append(r)
    return unique


def crawl(movie_name):
    """Run all sources in priority order; return (DataFrame, per-source stats)."""
    all_reviews, stats = [], {}
    for name, scraper in SOURCES:
        if len(deduplicate(all_reviews)) >= TARGET_REVIEWS:
            stats[name] = "skipped (target reached)"
            continue
        print(f"\n→ {name} …")
        t0 = time.time()
        try:
            got, reason = scraper(movie_name, TARGET_REVIEWS)
        except Exception as e:  # a broken scraper must not kill the crawl
            got, reason = [], f"crashed: {type(e).__name__}: {e}"
            print(f"    ! scraper crashed: {type(e).__name__}: {e}")
        got = [r for r in got if len(r["review_text"]) >= MIN_REVIEW_CHARS]
        all_reviews.extend(got)
        dt = time.time() - t0
        stats[name] = (f"{len(got)} reviews in {dt:.1f}s" if got
                       else f"0 reviews — {reason} ({dt:.1f}s)")
        print(f"    {stats[name]}")

    unique = deduplicate(all_reviews)
    df = pd.DataFrame(unique, columns=["source", "review_text", "date", "rating"])
    return df, stats


def save(df, movie_name):
    """Write the Excel output file; returns its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", movie_name).strip("_")
    path = os.path.join(OUTPUT_DIR, f"{safe}_reviews.xlsx")
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", help="movie name to crawl reviews for")
    args = parser.parse_args()

    print(f"Crawling reviews for: {args.movie}")
    df, stats = crawl(args.movie)
    path = save(df, args.movie)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Per-source results:")
    for name, s in stats.items():
        print(f"  {name:<18} {s}")
    print(f"\nTotal unique reviews: {len(df)}")
    print(f"Saved → {path}")
    if len(df) == 0:
        print("❌ FAILED — no reviews collected (all sources failed/blocked)")
        sys.exit(1)
    elif len(df) < TARGET_REVIEWS:
        print(f"⚠️  collected fewer than the {TARGET_REVIEWS}-review target")
    print("✅ PASSED — crawl finished and Excel file written")


if __name__ == "__main__":
    main()
