"""
method1.py — Lexical-Tabular Hybrid Pipeline
=============================================

METHOD SUMMARY:
  Two fast, deterministic scoring vectors, no embeddings, no ML models.

  Layer 0 — Hard Filter (rules.py):
    Instantly drops honeypots, pure-consulting careers, misaligned titles,
    and academic-only profiles before any scoring begins.

  Layer 1 — Targeted Lexical Score (0–100 pts):
    Keyword presence search against three "MUST HAVE" JD clusters:
      • Vector DB tools   (Pinecone, Weaviate, Qdrant, Milvus, FAISS …)
      • Eval metrics      (NDCG, MRR, MAP, A/B test, correlation …)
      • IR / Ranking core (ranking, retrieval, search, recommendation …)
    Matches in *career history descriptions* score higher than skill-list
    matches because descriptions prove production use, not just keyword listing.

  Layer 2 — Tabular Signal Score (0–120 pts):
    Structured fields evaluated against JD targets:
      • Years of experience sweet-spot (5–9 yrs)
      • Location affinity (Pune / Noida / Delhi NCR)
      • Notice period (≤30 days = max availability bonus)
      • Platform engagement (recruiter response rate)
      • GitHub activity (open-source signal)
      • Open-to-work flag

  Layer 3 — Soft Penalties (scoring.py):
    Multiplicative down-weights for: LangChain-novice, arch-only roles,
    job-hopping, framework-enthusiasm, CV/robotics-only, closed-source silo.

  Final sort → top 100 exact rows → submission CSV.

RUNTIME: ~30–60 s on CPU for 100 K candidates (pure stdlib + pandas).
NO NETWORK CALLS. NO GPU. NO EMBEDDINGS.
"""

from __future__ import annotations

import json
import sys
import time
import csv
from pathlib import Path

# ── Optional pandas (faster sort) — fall back to stdlib if not installed ──────
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ── Project root so relative imports work wherever this file is called from ───
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rules import evaluate_hard_filters
from src.scoring import calculate_advanced_penalties, compute_base_score
from src.config import AI_CORE_SKILLS


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — JD-targeted keyword clusters
# ══════════════════════════════════════════════════════════════════════════════

TECH_KEYWORDS: dict[str, list[str]] = {
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "chroma", "pgvector", "vespa",
        "typesense", "redis vector", "annoy",
    ],
    "eval_metrics": [
        "ndcg", "mrr", "map@", "mean average precision",
        "a/b test", "ab test", "online experiment",
        "correlation", "benchmark", "precision@", "recall@",
        "click-through", "ctr", "dwell time", "interleaving",
    ],
    "ranking_systems": [
        "ranking", "retrieval", "information retrieval",
        "search", "recommendation", "recommender",
        "embeddings", "hybrid search", "dense retrieval",
        "sparse retrieval", "reranking", "re-ranking",
        "learning to rank", "ltr", "query understanding",
    ],
}

# Points awarded per cluster
LEXICAL_WEIGHTS: dict[str, dict[str, float]] = {
    "vector_db":      {"history": 30.0, "skills": 10.0},
    "eval_metrics":   {"history": 30.0, "skills":  8.0},
    "ranking_systems": {"history": 40.0, "skills": 12.0},
}

# Tabular scoring config
TABULAR_BASE        = 50.0
EXP_IDEAL_MIN       = 5.0
EXP_IDEAL_MAX       = 9.0
EXP_BONUS           = 20.0
EXP_LOW_PENALTY     = -30.0
EXP_LOW_THRESHOLD   = 4.0
LOCATION_BONUS      = 20.0
LOCATION_TARGETS    = {"pune", "noida", "delhi", "gurgaon", "gurugram", "ncr"}
NOTICE_IDEAL_DAYS   = 30
NOTICE_IDEAL_BONUS  = 15.0
NOTICE_HIGH_DAYS    = 90
NOTICE_HIGH_PENALTY = -20.0
RESPONSE_MAX_BONUS  = 15.0
GITHUB_MAX_BONUS    = 10.0
OPEN_TO_WORK_BONUS  = 5.0

TOP_N      = 100
OUT_FILE   = ROOT / "team_redrob_challenge.csv"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER — auto-finds candidates file regardless of bracket-path weirdness
# ══════════════════════════════════════════════════════════════════════════════

def _find_candidates_file() -> Path:
    """
    Walk the project tree to find candidates.jsonl.
    Handles the '[PUB] ...' bracket-folder name that trips up raw Path joins
    on Windows.
    """
    for root, _dirs, files in (ROOT).walk() if hasattr(ROOT, "walk") else _os_walk(ROOT):
        root = Path(root)
        if "candidates.jsonl" in files:
            candidate = root / "candidates.jsonl"
            print(f"[DATA] Found candidates file: {candidate}")
            return candidate
    raise FileNotFoundError(
        "Could not locate candidates.jsonl anywhere under "
        f"{ROOT}. Run this script from the project root."
    )


def _os_walk(path: Path):
    """Compatibility shim for Python < 3.12 where Path.walk() doesn't exist."""
    import os
    yield from os.walk(str(path))


def _stream_candidates(path: Path):
    """Yield one candidate dict per line — constant memory, handles 100K+."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSON at line {lineno}: {exc}",
                      file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Targeted Lexical Score
# ══════════════════════════════════════════════════════════════════════════════

def calculate_lexical_score(candidate: dict) -> float:
    """
    Awards points for *production-depth* keyword presence.

    History descriptions score higher than the skills array because a skill
    tag proves nothing — a description proves you shipped it.
    """
    history  = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])

    history_text = " ".join(
        (j.get("description", "") + " " + j.get("title", "")).lower()
        for j in history
    )
    skills_text = " ".join(s.get("name", "").lower() for s in skills)

    score = 0.0

    for cluster, keywords in TECH_KEYWORDS.items():
        w = LEXICAL_WEIGHTS[cluster]
        in_history = any(kw in history_text for kw in keywords)
        in_skills  = any(kw in skills_text  for kw in keywords)

        if in_history:
            score += w["history"]
        elif in_skills:
            score += w["skills"]   # partial credit — skills list only

    return score


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Tabular Signal Score
# ══════════════════════════════════════════════════════════════════════════════

def calculate_tabular_score(candidate: dict) -> float:
    """
    Evaluates structured profile statistics against JD hiring targets.
    """
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    score = TABULAR_BASE

    # ── Experience alignment ────────────────────────────────────────────────
    yoe: float = profile.get("years_of_experience", 0)
    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        score += EXP_BONUS
    elif yoe < EXP_LOW_THRESHOLD:
        score += EXP_LOW_PENALTY  # heavy penalty for junior candidates

    # ── Location affinity ───────────────────────────────────────────────────
    location = profile.get("location", "").lower()
    if any(city in location for city in LOCATION_TARGETS):
        score += LOCATION_BONUS

    # ── Notice period availability ──────────────────────────────────────────
    notice: int = signals.get("notice_period_days", 90)
    if notice <= NOTICE_IDEAL_DAYS:
        score += NOTICE_IDEAL_BONUS
    elif notice > NOTICE_HIGH_DAYS:
        score += NOTICE_HIGH_PENALTY

    # ── Recruiter response rate ─────────────────────────────────────────────
    response_rate: float = signals.get("recruiter_response_rate", 0.0)
    score += response_rate * RESPONSE_MAX_BONUS

    # ── GitHub / open-source signal ─────────────────────────────────────────
    github_score: float = signals.get("github_activity_score", -1)
    if github_score > 0:
        score += (github_score / 100.0) * GITHUB_MAX_BONUS

    # ── Open to work ────────────────────────────────────────────────────────
    if signals.get("open_to_work_flag", False):
        score += OPEN_TO_WORK_BONUS

    return max(0.0, score)


# ══════════════════════════════════════════════════════════════════════════════
# REASONING GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _build_reasoning(candidate: dict, lexical: float, tabular: float, final: float) -> str:
    """
    Produces a concise, human-readable reasoning string for the submission CSV.
    Matches the format observed in sample_submission.csv.
    """
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    title    = profile.get("current_title", "N/A")
    yoe      = profile.get("years_of_experience", 0)
    rr       = signals.get("recruiter_response_rate", 0)

    ai_count = sum(
        1 for s in candidate.get("skills", [])
        if any(core in s.get("name", "").lower() for core in AI_CORE_SKILLS)
    )

    return (
        f"{title} with {yoe:.1f} yrs; {ai_count} AI core skills; "
        f"response rate {rr:.2f}. "
        f"[lex={lexical:.0f}, tab={tabular:.0f}, final={final:.4f}]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CSV WRITER — submission-spec compliant
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(rows: list[dict], out_path: Path) -> None:
    """
    Writes exactly TOP_N rows sorted by score descending, rank 1..100.
    Tie-breaking: candidate_id ascending (per submission spec).
    """
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = rows[:TOP_N]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rank, row in enumerate(top, 1):
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank":         rank,
                "score":        f"{row['score']:.4f}",
                "reasoning":    row["reasoning"],
            })

    print(f"\n[OUTPUT] Written {len(top)} rows → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("Method 1 — Lexical-Tabular Hybrid Pipeline")
    print("=" * 64)

    candidates_path = _find_candidates_file()

    shortlist: list[dict] = []
    stats = {"read": 0, "dropped": 0, "scored": 0}
    t0 = time.perf_counter()

    for candidate in _stream_candidates(candidates_path):
        stats["read"] += 1

        if stats["read"] % 5_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  … {stats['read']:,} read | {stats['dropped']:,} dropped "
                  f"| {stats['scored']:,} scored | {elapsed:.1f}s", end="\r")

        # ── Layer 0: Hard filters ────────────────────────────────────────────
        filter_result = evaluate_hard_filters(candidate)
        if not filter_result.passed:
            stats["dropped"] += 1
            continue

        # ── Layer 1: Lexical score ───────────────────────────────────────────
        lexical = calculate_lexical_score(candidate)

        # ── Layer 2: Tabular score ───────────────────────────────────────────
        tabular = calculate_tabular_score(candidate)

        combined_base = lexical + tabular

        # ── Layer 3: Soft penalties ─────────────────────────────────────────
        # Normalise combined_base (max ~220) to [0,1] before penalties,
        # then scale back to a [0, 220] range so score ordering is preserved.
        normalised = min(combined_base / 220.0, 1.0)
        penalised, _ = calculate_advanced_penalties(candidate, normalised)
        final_score = round(penalised * 220.0, 4)

        shortlist.append({
            "candidate_id": candidate["candidate_id"],
            "score":        final_score,
            "reasoning":    _build_reasoning(candidate, lexical, tabular, final_score),
        })
        stats["scored"] += 1

    # Clear the progress line
    print(" " * 80, end="\r")

    elapsed = time.perf_counter() - t0
    print(f"\n[STATS]")
    print(f"  Total read   : {stats['read']:,}")
    print(f"  Hard-dropped : {stats['dropped']:,}")
    print(f"  Scored       : {stats['scored']:,}")
    print(f"  Elapsed      : {elapsed:.2f}s")

    if not shortlist:
        print("[ERROR] No candidates survived filters. Cannot produce output.")
        sys.exit(1)

    emitted = min(len(shortlist), TOP_N)
    print(f"  Emitting     : {emitted} (of {len(shortlist):,} scored)")

    _write_csv(shortlist, OUT_FILE)
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
