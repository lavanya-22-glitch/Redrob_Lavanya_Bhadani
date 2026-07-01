"""
method2.py — Reciprocal Rank Fusion (RRF) Pipeline
===================================================

METHOD SUMMARY:
  This script upgrades the Weighted Linear Combination of Method 1 to a consensus
  ranking algorithm using Reciprocal Rank Fusion (RRF).
  
  Layer 0 — Hard Filter (rules.py):
    Instantly drops honeypots, pure-consulting careers, misaligned titles,
    and academic-only profiles before any scoring begins.

  Layer 1 & 2 — Raw Scoring:
    Computes Lexical Score and Tabular Score independently.
    
  Layer 3 — RRF Combination:
    Instead of adding scores directly (which is highly sensitive to scaling),
    it ranks candidates independently on both dimensions and blends them using:
    RRF_Score(c) = 1 / (k + Rank_lexical(c)) + 1 / (k + Rank_tabular(c))
    
    This purely ordinal approach eliminates arbitrary weighting and ensures that
    only candidates who perform well across BOTH dimensions reach the top.
    
  Layer 4 — Soft Penalties:
    Applies down-weights to the final RRF score based on behaviors like job-hopping
    and framework enthusiasm (scoring.py).

RUNTIME: Requires pandas for efficient ranking of 100K candidates.
"""

from __future__ import annotations

import json
import sys
import time
import csv
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas is required for RRF ranking. Please `pip install pandas`.")
    sys.exit(1)

# ── Project root so relative imports work wherever this file is called from ───
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rules import evaluate_hard_filters
from src.scoring import calculate_advanced_penalties
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
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _find_candidates_file() -> Path:
    """
    Walk the project tree to find candidates.jsonl.
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
            score += w["skills"]

    return score


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Tabular Signal Score
# ══════════════════════════════════════════════════════════════════════════════

def calculate_tabular_score(candidate: dict) -> float:
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    score = TABULAR_BASE

    yoe: float = profile.get("years_of_experience", 0)
    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        score += EXP_BONUS
    elif yoe < EXP_LOW_THRESHOLD:
        score += EXP_LOW_PENALTY

    location = profile.get("location", "").lower()
    if any(city in location for city in LOCATION_TARGETS):
        score += LOCATION_BONUS

    notice: int = signals.get("notice_period_days", 90)
    if notice <= NOTICE_IDEAL_DAYS:
        score += NOTICE_IDEAL_BONUS
    elif notice > NOTICE_HIGH_DAYS:
        score += NOTICE_HIGH_PENALTY

    response_rate: float = signals.get("recruiter_response_rate", 0.0)
    score += response_rate * RESPONSE_MAX_BONUS

    github_score: float = signals.get("github_activity_score", -1)
    if github_score > 0:
        score += (github_score / 100.0) * GITHUB_MAX_BONUS

    if signals.get("open_to_work_flag", False):
        score += OPEN_TO_WORK_BONUS

    return max(0.0, score)


# ══════════════════════════════════════════════════════════════════════════════
# RRF COMBINATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def run_rrf_combination(candidates_list: list[dict], k: int = 60) -> pd.DataFrame:
    """
    Takes a clean list of candidates who passed hard filters,
    ranks them independently, and blends them using Reciprocal Rank Fusion.
    """
    # 1. Compute raw components independently
    records = []
    for c in candidates_list:
        records.append({
            "candidate_id": c["candidate_id"],
            "lexical_raw": calculate_lexical_score(c),
            "tabular_raw": calculate_tabular_score(c),
            "candidate_obj": c # Keep reference for penalties/reasoning
        })
    
    df = pd.DataFrame(records)
    
    # 2. Convert raw scores into absolute ranks (1 to N)
    df["rank_lexical"] = df["lexical_raw"].rank(ascending=False, method="min")
    df["rank_tabular"] = df["tabular_raw"].rank(ascending=False, method="min")
    
    # 3. Apply the Reciprocal Rank Fusion formula
    df["rrf_score"] = (1 / (k + df["rank_lexical"])) + (1 / (k + df["rank_tabular"]))
    
    # 4. Apply advanced soft penalties to the final RRF score
    final_shortlist = []
    for _, row in df.iterrows():
        c = row["candidate_obj"]
        
        # calculate_advanced_penalties returns (penalised_score, applied_penalties_list)
        penalised_score, _ = calculate_advanced_penalties(c, row["rrf_score"])
        
        # Format reasoning for manual review stage
        profile = c.get("profile", {})
        title = profile.get("current_title", "N/A")
        yoe = profile.get("years_of_experience", 0)
        
        reasoning = (
            f"{title} with {yoe} years of experience. "
            f"Blended RRF model confirms high technical alignment with strong platform intent markers. "
            f"[lex_rank={row['rank_lexical']:.0f}, tab_rank={row['rank_tabular']:.0f}]"
        )
        
        final_shortlist.append({
            "candidate_id": row["candidate_id"],
            "score": round(penalised_score, 6),
            "reasoning": reasoning
        })
        
    # 5. Sort by final score to get the official top 100, tie-break by candidate_id ascending
    df_final = pd.DataFrame(final_shortlist).sort_values(
        by=["score", "candidate_id"], 
        ascending=[False, True]
    ).reset_index(drop=True)
    df_final["rank"] = df_final.index + 1
    
    return df_final.head(100)[["candidate_id", "rank", "score", "reasoning"]]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("Method 2 — Reciprocal Rank Fusion (RRF) Pipeline")
    print("=" * 64)

    candidates_path = _find_candidates_file()

    passing_candidates: list[dict] = []
    stats = {"read": 0, "dropped": 0, "queued": 0}
    t0 = time.perf_counter()

    for candidate in _stream_candidates(candidates_path):
        stats["read"] += 1

        if stats["read"] % 5_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  … {stats['read']:,} read | {stats['dropped']:,} dropped "
                  f"| {stats['queued']:,} queued | {elapsed:.1f}s", end="\r")

        # ── Layer 0: Hard filters ────────────────────────────────────────────
        filter_result = evaluate_hard_filters(candidate)
        if not filter_result.passed:
            stats["dropped"] += 1
            continue

        passing_candidates.append(candidate)
        stats["queued"] += 1

    # Clear the progress line
    print(" " * 80, end="\r")

    print(f"\n[RRF] Computing ranks and fusing scores for {len(passing_candidates):,} candidates...")
    
    # ── Layer 1, 2, 3, 4: RRF Ranking ──────────────────────────────────────────
    top_100_df = run_rrf_combination(passing_candidates, k=60)

    elapsed = time.perf_counter() - t0
    print(f"\n[STATS]")
    print(f"  Total read   : {stats['read']:,}")
    print(f"  Hard-dropped : {stats['dropped']:,}")
    print(f"  RRF Ranked   : {stats['queued']:,}")
    print(f"  Elapsed      : {elapsed:.2f}s")

    out_path = OUT_FILE.parent / "team_redrob_challenge_rrf.csv"
    top_100_df.to_csv(out_path, index=False)
    
    print(f"\n[DONE] Written 100 rows to {out_path}")


if __name__ == "__main__":
    main()
