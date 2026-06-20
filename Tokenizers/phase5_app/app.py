"""
Phase 5 — Practical Application Panel (Streamlit web UI).

Type a movie name → the Phase 4 crawler collects reviews (TMDB · Letterboxd ·
Trakt) → every review is scored by the combined sentiment+sarcasm pipeline
(Phase 3). The dashboard then shows, in order:

  1. the RAW sentiment verdict + confidence (first model only, before sarcasm),
  2. the verdict AFTER the sarcasm model corrects sarcastic reviews,
  3. top positive / negative / most-sarcastic reviews,
  4. a per-source crawler report (with the reason each source returned), and
  5. a History tab with the last few analyses of the session.

Run with:
    streamlit run phase5_app/app.py

The run_pipeline() function is import-safe (no Streamlit calls) and is reused by
test_pipeline_e2e.py.
"""

import os
import sys
from collections import Counter
from datetime import datetime

# make the phase3 / phase4 modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "phase3_sarcasm"))
sys.path.insert(0, os.path.join(ROOT, "phase4_crawler"))

# ── Configuration ──────────────────────────────────────────────────────────────
GOOD_THRESHOLD = 0.60     # ≥ 60% effectively-positive reviews → GOOD
BAD_THRESHOLD = 0.40      # ≤ 40% → BAD; in between → MIXED
TOP_K = 5                 # how many top positive/negative/sarcastic reviews to show
MAX_REVIEWS_TO_SCORE = 150  # safety cap so the UI stays responsive
HISTORY_SIZE = 3          # how many past analyses to keep in the History tab
VERDICT_EMOJI = {"GOOD": "👍", "BAD": "👎", "MIXED": "🤔"}


# ── Pipeline (UI-independent, reused by the e2e test) ─────────────────────────
def compute_verdict(pos_share):
    """Map the effectively-positive share to (verdict, confidence%)."""
    if pos_share >= GOOD_THRESHOLD:
        return "GOOD", round(pos_share * 100)
    if pos_share <= BAD_THRESHOLD:
        return "BAD", round((1 - pos_share) * 100)
    return "MIXED", round((1 - abs(pos_share - 0.5) * 2) * 100)


def run_pipeline(movie_name, analyzer=None, progress_callback=None):
    """Crawl reviews for `movie_name`, score each, aggregate. Returns a dict with
    both the raw-sentiment (pre-sarcasm) and sarcasm-corrected results."""
    import crawler
    from combined_inference import CombinedAnalyzer

    if analyzer is None:
        analyzer = CombinedAnalyzer()

    df, stats = crawler.crawl(movie_name)
    rows = df.to_dict("records")[:MAX_REVIEWS_TO_SCORE]
    if not rows:
        return {"movie": movie_name, "n_reviews": 0, "crawl_stats": stats,
                "verdict": None, "confidence": None, "results": []}

    results = []
    for i, row in enumerate(rows):
        r = analyzer.analyze(str(row["review_text"]))
        r["source"] = row.get("source", "?")
        results.append(r)
        if progress_callback:
            progress_callback((i + 1) / len(rows))

    # ── Stage 1: raw sentiment (the first model only, before sarcasm) ──────────
    raw_pos = [r for r in results if r["sentiment"] == "Positive"]
    raw_pos_share = len(raw_pos) / len(results)
    raw_verdict, raw_confidence = compute_verdict(raw_pos_share)

    # ── Stage 2: after the sarcasm model corrects sarcastic positives ──────────
    pos = [r for r in results if r["effective_positive"]]
    neg = [r for r in results if not r["effective_positive"]]
    sarcastic = [r for r in results if r["sarcasm"] == "Sarcastic"]
    pos_share = len(pos) / len(results)
    verdict, confidence = compute_verdict(pos_share)

    return {
        "movie": movie_name,
        "n_reviews": len(results),
        "crawl_stats": stats,
        "source_counts": dict(Counter(r["source"] for r in results)),
        # before sarcasm
        "raw_pos_share": raw_pos_share,
        "raw_verdict": raw_verdict,
        "raw_confidence": raw_confidence,
        # after sarcasm
        "pos_share": pos_share,
        "verdict": verdict,
        "confidence": confidence,
        "n_sarcastic": len(sarcastic),
        "top_positive": sorted(pos, key=lambda r: r["sentiment_confidence"], reverse=True)[:TOP_K],
        "top_negative": sorted(neg, key=lambda r: r["sentiment_confidence"], reverse=True)[:TOP_K],
        "top_sarcastic": sorted(sarcastic, key=lambda r: r["sarcasm_confidence"], reverse=True)[:TOP_K],
        "results": results,
    }


# ── Streamlit render helpers (st/plt passed in; never run at import) ──────────
def _pie(st, plt, pos_share, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    p = pos_share * 100
    ax.pie([p, 100 - p], labels=["Positive", "Negative"], autopct="%1.0f%%",
           colors=["#4CAF50", "#E53935"], startangle=90)
    ax.set_title(title)
    st.pyplot(fig)


def _review_list(st, items, kind, conf_key="sentiment_confidence", show_sarc=False):
    if not items:
        st.info("None found.")
        return
    for r in items:
        tag = " · 🙄 sarcastic" if (show_sarc and r["sarcasm"] == "Sarcastic") else ""
        getattr(st, kind)(f"({r[conf_key]:.0%} · {r['source']}{tag}) {r['text'][:400]}")


def _crawl_report(st, stats):
    for src, msg in stats.items():
        icon = "✅" if "reviews in" in msg else ("⏭️" if "skip" in msg.lower() else "⚠️")
        st.write(f"{icon} **{src}** — {msg}")


def _render_results(st, plt, res):
    # ── Two-stage verdict: before vs after sarcasm ────────────────────────────
    st.subheader("Verdict — before vs after sarcasm")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 1️⃣ Raw sentiment (first model only)")
        st.markdown(f"## {VERDICT_EMOJI[res['raw_verdict']]} {res['raw_verdict']}")
        st.caption(f"**{res['raw_confidence']}%** confidence · "
                   f"{res['raw_pos_share']*100:.0f}% positive")
        _pie(st, plt, res["raw_pos_share"], "Before sarcasm")
    with c2:
        st.markdown("##### 2️⃣ After sarcasm correction")
        st.markdown(f"## {VERDICT_EMOJI[res['verdict']]} {res['verdict']}")
        st.caption(f"**{res['confidence']}%** confidence · "
                   f"{res['pos_share']*100:.0f}% positive")
        _pie(st, plt, res["pos_share"], "After sarcasm")

    if res["raw_verdict"] != res["verdict"]:
        st.warning(f"🙄 Sarcasm correction **changed the verdict**: "
                   f"{res['raw_verdict']} → {res['verdict']} "
                   f"({res['n_sarcastic']} sarcastic reviews re-counted as negative).")
    else:
        st.info(f"🙄 Sarcasm flagged {res['n_sarcastic']} review(s); verdict unchanged "
                f"({res['raw_pos_share']*100:.0f}% → {res['pos_share']*100:.0f}% positive).")

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Reviews analyzed", res["n_reviews"])
    m2.metric("Detected as sarcastic", res["n_sarcastic"],
              f"{res['n_sarcastic']/res['n_reviews']:.0%} of reviews")
    m3.metric("Sources used", len(res["source_counts"]))
    st.caption("Per source — " + " · ".join(f"**{k}**: {v}"
                                             for k, v in res["source_counts"].items()))

    # ── Top reviews ───────────────────────────────────────────────────────────
    st.subheader("👍 Top genuinely positive")
    _review_list(st, res["top_positive"], "success")
    st.subheader("👎 Top negative")
    _review_list(st, res["top_negative"], "error", show_sarc=True)
    st.subheader("🙄 Most sarcastic")
    _review_list(st, res["top_sarcastic"], "warning", conf_key="sarcasm_confidence")

    with st.expander("🛰️ Crawler report — per source"):
        _crawl_report(st, res["crawl_stats"])


def _render_history(st):
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No analyses yet — run one in the Analyze tab.")
        return
    st.caption(f"Last {len(hist)} analyses this session")
    for h in hist:
        with st.container(border=True):
            st.markdown(f"**🎬 {h['movie']}**  ·  {h['time']}  ·  {h['n_reviews']} reviews")
            h1, h2, h3 = st.columns(3)
            h1.metric("Before sarcasm", h["raw_verdict"], f"{h['raw_confidence']}%")
            h2.metric("After sarcasm", h["verdict"], f"{h['confidence']}%")
            h3.metric("Sarcastic", h["n_sarcastic"])


# ── Streamlit UI ───────────────────────────────────────────────────────────────
def main():
    import matplotlib.pyplot as plt
    import streamlit as st
    from combined_inference import CombinedAnalyzer

    st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬",
                       layout="wide")
    st.title("🎬 Movie Review Analyzer")
    st.caption("Sentiment + sarcasm analysis over freshly crawled reviews "
               "(TMDB · Letterboxd · Trakt)")

    @st.cache_resource(show_spinner="Loading models…")
    def get_analyzer():
        return CombinedAnalyzer()

    tab_analyze, tab_history = st.tabs(["🔍 Analyze", "🕑 History"])

    with tab_analyze:
        movie = st.text_input("Movie name", placeholder="e.g. Inception")
        if st.button("Analyze", type="primary") and movie.strip():
            progress = st.progress(0.0, text="Crawling reviews…")
            analyzer = get_analyzer()

            def on_progress(frac):
                progress.progress(frac, text=f"Scoring reviews… {frac:.0%}")

            with st.spinner(f"Collecting reviews for “{movie}”…"):
                res = run_pipeline(movie.strip(), analyzer, on_progress)
            progress.empty()

            if res["n_reviews"] == 0:
                st.error("No reviews could be collected for this title.")
                with st.expander("🛰️ Crawler report — per source", expanded=True):
                    _crawl_report(st, res["crawl_stats"])
            else:
                hist = st.session_state.setdefault("history", [])
                hist.insert(0, {
                    "movie": res["movie"], "time": datetime.now().strftime("%H:%M:%S"),
                    "n_reviews": res["n_reviews"], "n_sarcastic": res["n_sarcastic"],
                    "raw_verdict": res["raw_verdict"], "raw_confidence": res["raw_confidence"],
                    "verdict": res["verdict"], "confidence": res["confidence"],
                })
                st.session_state["history"] = hist[:HISTORY_SIZE]
                _render_results(st, plt, res)

    with tab_history:
        _render_history(st)

    # ── Info section ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "**How it works** — Reviews are crawled live from TMDB, Letterboxd and "
        "Trakt, then each one is scored by two fine-tuned BERT models: a sentiment "
        "classifier (positive/negative) and a movie-domain sarcasm detector. A "
        "review that sounds positive but is sarcastic is re-counted as negative; "
        "the dashboard shows the verdict both **before** and **after** that "
        "correction so you can see sarcasm's impact.")


if __name__ == "__main__":
    main()
