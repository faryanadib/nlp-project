"""
Phase 4 — Sanity checks for the crawler.

Crawls TEST_MOVIES and asserts, per movie:
  * the Excel file is created and non-empty
  * no review is shorter than MIN_REVIEW_CHARS (not truncated)
  * no duplicate rows

Usage:
    python test_crawler.py
"""

import os
import sys

import pandas as pd

import crawler

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_MOVIES = ["Inception", "The Godfather"]   # 2 well-known movies


def check_movie(movie):
    """Crawl one movie and run all assertions; returns list of failure strings."""
    failures = []
    df, stats = crawler.crawl(movie)
    path = crawler.save(df, movie)

    # file created and non-empty
    if not os.path.isfile(path):
        failures.append("Excel file was not created")
    elif os.path.getsize(path) == 0:
        failures.append("Excel file is empty (0 bytes)")
    saved = pd.read_excel(path) if os.path.isfile(path) else pd.DataFrame()
    if len(saved) == 0:
        failures.append("Excel file contains no review rows")

    if len(saved) > 0:
        # reviews not truncated
        min_len = saved["review_text"].astype(str).str.len().min()
        if min_len < crawler.MIN_REVIEW_CHARS:
            failures.append(f"shortest review is {min_len} chars "
                            f"(< {crawler.MIN_REVIEW_CHARS})")
        # no duplicate rows
        n_dupes = int(saved.duplicated(subset="review_text").sum())
        if n_dupes:
            failures.append(f"{n_dupes} duplicate review rows found")

    return failures, len(saved), stats


def main():
    overall_ok = True
    for movie in TEST_MOVIES:
        print("\n" + "═" * 60)
        print(f"TESTING: {movie}")
        print("═" * 60)
        failures, n, _ = check_movie(movie)
        if failures:
            overall_ok = False
            for f in failures:
                print(f"[FAIL] {movie}: {f}")
        else:
            print(f"[PASS] {movie}: {n} unique reviews, file OK, no dupes, no truncation")

    print("\n" + "─" * 60)
    print("✅ PASSED — all crawler sanity checks OK" if overall_ok
          else "❌ FAILED — see failures above")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
