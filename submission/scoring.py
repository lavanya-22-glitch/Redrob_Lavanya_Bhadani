"""
scoring.py — Layer 2: Base scoring + soft-penalty engine.

Architecture
============
  1. `compute_base_score(candidate)`  → float in [0, 1]
        Weighted sum of five positive signals: AI skills, experience alignment,
        title alignment, platform engagement, and education.

  2. `calculate_advanced_penalties(candidate, base_score)` → float in [0, 1]
        Applies multiplicative down-weights for the six nuanced negative signals
        (Problems 5–10) that rule-based hard filters cannot reliably catch.

  3. `score_candidate(candidate)` → ScoringResult
        Convenience wrapper that calls both stages and returns a structured
        result with a detailed audit trail of every penalty applied.

All thresholds and keyword sets are imported from config.py so that nothing
in this file contains magic numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import (
    # Positive scoring
    AI_CORE_SKILLS,
    EDUCATION_TIER_SCORES,
    IDEAL_EXP_MAX,
    IDEAL_EXP_MIN,
    POSITIVE_TITLE_KEYWORDS,
    WEIGHT_AI_SKILLS,
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_OPEN_SOURCE,
    WEIGHT_PLATFORM_SIGNALS,
    WEIGHT_TITLE_ALIGNMENT,
    # Penalty: Problem 5
    LANGCHAIN_SHALLOW_THRESHOLD_MONTHS,
    LANGCHAIN_WRAPPER_PENALTY,
    LEGACY_IR_KEYWORDS,
    # Penalty: Problem 6
    ARCH_ONLY_PENALTY,
    ARCH_TITLE_KEYWORDS,
    HANDS_ON_VERBS,
    # Penalty: Problem 7
    JOBHOP_AVG_TENURE_MONTHS,
    JOBHOP_MIN_JOBS_REQUIRED,
    JOBHOP_PENALTY,
    # Penalty: Problem 8
    FRAMEWORK_COUNT_THRESHOLD,
    FRAMEWORK_PENALTY,
    HIGH_LEVEL_FRAMEWORKS,
    SYSTEMS_KEYWORDS,
    # Penalty: Problem 9
    CV_ROBOTICS_KEYWORDS,
    CV_ROBOTICS_PENALTY,
    NLP_IR_KEYWORDS,
    # Penalty: Problem 10
    CLOSED_SOURCE_PENALTY,
    PUBLIC_ARTIFACT_KEYWORDS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PenaltyRecord:
    """Documents a single soft penalty that was applied."""
    problem_id: int
    name: str
    multiplier: float
    detail: str


@dataclass
class ScoringResult:
    candidate_id: str
    base_score: float
    final_score: float
    penalties_applied: list[PenaltyRecord] = field(default_factory=list)
    reasoning: str = ""

    def penalty_summary(self) -> str:
        if not self.penalties_applied:
            return "No penalties applied."
        parts = [
            f"P{p.problem_id}({p.name}×{p.multiplier})"
            for p in self.penalties_applied
        ]
        return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_corpus(candidate: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (history_text, skills_text) — lowercased blobs used for keyword
    searches throughout the penalty functions.
    """
    history: list[dict] = candidate.get("career_history", [])
    history_text = " ".join(
        (job.get("description", "") + " " + job.get("title", "")).lower()
        for job in history
    )
    skills_text = " ".join(
        s.get("name", "").lower() for s in candidate.get("skills", [])
    )
    return history_text, skills_text


def _contains_any(text: str, keywords: frozenset[str]) -> bool:
    return any(kw in text for kw in keywords)


def _count_matches(text: str, keywords: frozenset[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Base score (positive signals)
# ──────────────────────────────────────────────────────────────────────────────

def _score_ai_skills(candidate: dict[str, Any]) -> float:
    """
    Fraction of AI_CORE_SKILLS found in the candidate's skill list, weighted
    by proficiency and endorsement credibility.
    """
    skills: list[dict] = candidate.get("skills", [])
    if not skills:
        return 0.0

    PROFICIENCY_WEIGHT = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}
    total_weight = 0.0
    matched = 0.0

    for s in skills:
        name = s.get("name", "").lower()
        if any(core in name for core in AI_CORE_SKILLS):
            prof = PROFICIENCY_WEIGHT.get(s.get("proficiency", "beginner"), 0.4)
            # Endorsement bonus: saturates at 50 endorsements
            endorse_bonus = min(s.get("endorsements", 0) / 50.0, 1.0) * 0.2
            matched += min(prof + endorse_bonus, 1.0)

    # Normalise: assume 10 AI-core skills is a "full" set
    return min(matched / 10.0, 1.0)


def _score_experience(candidate: dict[str, Any]) -> float:
    """
    Gaussian-shaped score that peaks at the ideal experience band.
    """
    yoe: float = candidate.get("profile", {}).get("years_of_experience", 0)
    if IDEAL_EXP_MIN <= yoe <= IDEAL_EXP_MAX:
        return 1.0
    elif yoe < IDEAL_EXP_MIN:
        return yoe / IDEAL_EXP_MIN
    else:
        # Graceful decay for over-experienced candidates
        return max(0.5, 1.0 - (yoe - IDEAL_EXP_MAX) / 10.0)


def _score_title_alignment(candidate: dict[str, Any]) -> float:
    """
    Binary-ish signal: strong positive if current title matches known
    engineering-track titles relevant to the JD.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()
    if _contains_any(title, POSITIVE_TITLE_KEYWORDS):
        return 1.0
    # Partial credit for generic engineering titles
    if any(kw in title for kw in ["engineer", "scientist", "developer", "architect"]):
        return 0.60
    return 0.10


def _score_platform_signals(candidate: dict[str, Any]) -> float:
    """
    Composite of Redrob engagement signals: responsiveness, recency,
    profile completeness, and recruiter interest.
    """
    signals: dict = candidate.get("redrob_signals", {})

    completeness = signals.get("profile_completeness_score", 0) / 100.0
    response_rate = signals.get("recruiter_response_rate", 0)
    interview_rate = signals.get("interview_completion_rate", 0)

    # Recency: penalise if last active > 6 months ago
    from datetime import datetime
    last_active_str = signals.get("last_active_date", "")
    recency_score = 0.5
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
        months_since = (datetime.now() - last_active).days / 30
        recency_score = max(0.0, 1.0 - months_since / 12.0)
    except (ValueError, TypeError):
        pass

    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.6

    return (
        0.25 * completeness
        + 0.25 * response_rate
        + 0.20 * interview_rate
        + 0.20 * recency_score
        + 0.10 * open_to_work
    )


def _score_education(candidate: dict[str, Any]) -> float:
    """
    Best education-tier score across all degrees, with a small bonus for
    AI/ML/CS field of study.
    """
    education: list[dict] = candidate.get("education", [])
    if not education:
        return 0.40

    RELEVANT_FIELDS = {"computer science", "machine learning", "ai",
                       "data science", "statistics", "mathematics",
                       "information technology", "electronics"}

    best = 0.0
    for edu in education:
        tier_score = EDUCATION_TIER_SCORES.get(edu.get("tier", "unknown"), 0.60)
        field = edu.get("field_of_study", "").lower()
        field_bonus = 0.10 if any(f in field for f in RELEVANT_FIELDS) else 0.0
        best = max(best, min(tier_score + field_bonus, 1.0))

    return best


def _score_open_source(candidate: dict[str, Any]) -> float:
    """
    Open-source / public-artifact presence score.
    Uses GitHub activity score from Redrob signals.
    """
    signals: dict = candidate.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", -1)

    if github_score == -1:
        return 0.0   # No GitHub linked
    return min(github_score / 100.0, 1.0)


def compute_base_score(candidate: dict[str, Any]) -> float:
    """
    Compute the weighted base score for a candidate from positive signals only.
    Returns a float in [0, 1].
    """
    score = (
        WEIGHT_AI_SKILLS       * _score_ai_skills(candidate)
        + WEIGHT_EXPERIENCE    * _score_experience(candidate)
        + WEIGHT_TITLE_ALIGNMENT * _score_title_alignment(candidate)
        + WEIGHT_PLATFORM_SIGNALS * _score_platform_signals(candidate)
        + WEIGHT_EDUCATION     * _score_education(candidate)
        + WEIGHT_OPEN_SOURCE   * _score_open_source(candidate)
    )
    return min(max(score, 0.0), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Soft penalties (Problems 5–10)
# ──────────────────────────────────────────────────────────────────────────────

def _penalty_langchain_novice(
    candidate: dict[str, Any],
    history_text: str,
    skills_text: str,
) -> PenaltyRecord | None:
    """
    PROBLEM 5 — Shallow / Recent LLM-only work (< 12 months, LangChain-dependent).

    Fires when:
      • LangChain is present in skill/history text, AND
      • No legacy IR foundations (search, ranking, BM25…) are found, AND
      • The candidate's LangChain skill duration is ≤ 12 months.
    """
    has_langchain = "langchain" in history_text or "langchain" in skills_text
    if not has_langchain:
        return None

    has_legacy_ir = _contains_any(history_text, LEGACY_IR_KEYWORDS)
    if has_legacy_ir:
        return None  # Solid IR foundations redeem the LangChain usage

    # Look up the actual LangChain skill duration
    langchain_months = 0
    for s in candidate.get("skills", []):
        if "langchain" in s.get("name", "").lower():
            langchain_months = max(langchain_months, s.get("duration_months", 0))

    if langchain_months <= LANGCHAIN_SHALLOW_THRESHOLD_MONTHS:
        return PenaltyRecord(
            problem_id=5,
            name="LangChainNovice",
            multiplier=LANGCHAIN_WRAPPER_PENALTY,
            detail=(
                f"LangChain present ({langchain_months} months), "
                f"no legacy IR/search foundations, shallow wrapper experience."
            ),
        )
    return None


def _penalty_architecture_only(
    candidate: dict[str, Any],
    history_text: str,
    skills_text: str,  # noqa: ARG001 — kept for interface consistency
) -> PenaltyRecord | None:
    """
    PROBLEM 6 — Architecture-only / hasn't coded in 18+ months.

    Fires when:
      • Current title is purely advisory (Architect / Director / Head of…), AND
      • The most recent role description lacks hands-on action verbs.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()

    # Must be an arch/leadership title WITHOUT 'engineer' in it
    is_arch_title = (
        _contains_any(title, ARCH_TITLE_KEYWORDS)
        and "engineer" not in title
    )
    if not is_arch_title:
        return None

    # Check if the most recent role description has hands-on verbs
    history: list[dict] = candidate.get("career_history", [])
    if not history:
        return None

    recent_desc = history[0].get("description", "").lower()
    has_hands_on = _contains_any(recent_desc, HANDS_ON_VERBS)

    if not has_hands_on:
        return PenaltyRecord(
            problem_id=6,
            name="ArchitectureOnly",
            multiplier=ARCH_ONLY_PENALTY,
            detail=(
                f"Current title '{title}' is purely advisory and most recent "
                f"role description has no hands-on action verbs."
            ),
        )
    return None


def _penalty_job_hopper(
    candidate: dict[str, Any],
    history_text: str,  # noqa: ARG001
    skills_text: str,  # noqa: ARG001
) -> PenaltyRecord | None:
    """
    PROBLEM 7 — Title Chasers / Job-hopping every ~1.5 years.

    Fires when average tenure across all jobs is ≤ 18 months
    (requires at least 3 recorded jobs to be statistically meaningful).
    """
    history: list[dict] = candidate.get("career_history", [])
    if len(history) < JOBHOP_MIN_JOBS_REQUIRED:
        return None

    avg_tenure = sum(j.get("duration_months", 0) for j in history) / len(history)
    if avg_tenure <= JOBHOP_AVG_TENURE_MONTHS:
        return PenaltyRecord(
            problem_id=7,
            name="JobHopper",
            multiplier=JOBHOP_PENALTY,
            detail=(
                f"Average tenure {avg_tenure:.1f} months across {len(history)} jobs "
                f"(threshold: ≤{JOBHOP_AVG_TENURE_MONTHS} months)."
            ),
        )
    return None


def _penalty_framework_enthusiast(
    candidate: dict[str, Any],
    history_text: str,
    skills_text: str,
) -> PenaltyRecord | None:
    """
    PROBLEM 8 — Framework Enthusiasts vs. Systems Thinkers.

    Fires when:
      • More than FRAMEWORK_COUNT_THRESHOLD high-level framework tools are found, AND
      • Zero systems-level keywords appear in the history corpus.
    """
    combined = history_text + " " + skills_text
    framework_count = _count_matches(combined, HIGH_LEVEL_FRAMEWORKS)
    systems_count = _count_matches(history_text, SYSTEMS_KEYWORDS)

    if framework_count > FRAMEWORK_COUNT_THRESHOLD and systems_count == 0:
        return PenaltyRecord(
            problem_id=8,
            name="FrameworkEnthusiast",
            multiplier=FRAMEWORK_PENALTY,
            detail=(
                f"Found {framework_count} high-level framework tools, "
                f"0 systems-level keywords — likely demo/tutorial-driven."
            ),
        )
    return None


def _penalty_cv_robotics_only(
    candidate: dict[str, Any],
    history_text: str,
    skills_text: str,
) -> PenaltyRecord | None:
    """
    PROBLEM 9 — Computer Vision / Robotics domain without NLP / IR footprint.

    Fires when CV/robotics keywords dominate the profile and NLP/IR keywords
    are completely absent — a red flag for re-learning fundamentals.
    """
    combined = history_text + " " + skills_text
    has_cv = _contains_any(combined, CV_ROBOTICS_KEYWORDS)
    has_nlp = _contains_any(combined, NLP_IR_KEYWORDS)

    if has_cv and not has_nlp:
        return PenaltyRecord(
            problem_id=9,
            name="CVRoboticsOnly",
            multiplier=CV_ROBOTICS_PENALTY,
            detail=(
                "Profile dominated by CV/robotics keywords with no NLP/IR evidence — "
                "significant domain gap for the target role."
            ),
        )
    return None


def _penalty_closed_source_silo(
    candidate: dict[str, Any],
    history_text: str,
    skills_text: str,  # noqa: ARG001
) -> PenaltyRecord | None:
    """
    PROBLEM 10 — Closed-Source Silos / No External Validation.

    Fires when:
      • GitHub activity score is -1 (no GitHub linked), AND
      • The profile text contains no references to public artifacts
        (papers, patents, conference talks, open-source contributions).
    """
    signals: dict = candidate.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", -1)

    if github_score != -1:
        return None  # Has a GitHub — no penalty

    has_public_artifacts = _contains_any(history_text, PUBLIC_ARTIFACT_KEYWORDS)

    # Also check the profile summary
    summary = candidate.get("profile", {}).get("summary", "").lower()
    has_public_artifacts = has_public_artifacts or _contains_any(summary, PUBLIC_ARTIFACT_KEYWORDS)

    if not has_public_artifacts:
        return PenaltyRecord(
            problem_id=10,
            name="ClosedSourceSilo",
            multiplier=CLOSED_SOURCE_PENALTY,
            detail=(
                "No GitHub linked (score=-1) and no references to papers, "
                "patents, conference talks, or open-source work found."
            ),
        )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Penalty registry (Problems 5–10, in order)
# ──────────────────────────────────────────────────────────────────────────────

_PENALTY_FUNCTIONS = [
    _penalty_langchain_novice,     # Problem 5
    _penalty_architecture_only,    # Problem 6
    _penalty_job_hopper,           # Problem 7
    _penalty_framework_enthusiast, # Problem 8
    _penalty_cv_robotics_only,     # Problem 9
    _penalty_closed_source_silo,   # Problem 10
]


def calculate_advanced_penalties(
    candidate: dict[str, Any],
    base_score: float,
) -> tuple[float, list[PenaltyRecord]]:
    """
    Apply all soft-penalty multipliers to the base score.

    Penalties are multiplicative and cumulative — a candidate can trigger
    multiple penalties simultaneously, each further reducing their score.

    Args:
        candidate:  Candidate dict conforming to candidate_schema.json.
        base_score: Score computed by `compute_base_score`.

    Returns:
        (final_score, list_of_applied_penalties)
    """
    history_text, skills_text = _extract_text_corpus(candidate)
    score = base_score
    applied: list[PenaltyRecord] = []

    for fn in _PENALTY_FUNCTIONS:
        penalty = fn(candidate, history_text, skills_text)
        if penalty is not None:
            score *= penalty.multiplier
            applied.append(penalty)
            logger.debug(
                "[PENALTY] %s | Problem %d (%s) | ×%.2f → score now %.4f",
                candidate.get("candidate_id", "?"),
                penalty.problem_id,
                penalty.name,
                penalty.multiplier,
                score,
            )

    return min(max(score, 0.0), 1.0), applied


# ──────────────────────────────────────────────────────────────────────────────
# Public API — combined scoring entry point
# ──────────────────────────────────────────────────────────────────────────────

def score_candidate(candidate: dict[str, Any]) -> ScoringResult:
    """
    Full scoring pipeline for a single candidate.

    1. Compute weighted base score from positive signals.
    2. Apply multiplicative soft penalties for nuanced negative signals.
    3. Return a ScoringResult with full audit trail.

    Args:
        candidate: Candidate dict (must have passed Layer 1 hard filters).

    Returns:
        ScoringResult with final_score, penalties_applied, and reasoning string.
    """
    cid = candidate.get("candidate_id", "<unknown>")
    profile = candidate.get("profile", {})

    base = compute_base_score(candidate)
    final, penalties = calculate_advanced_penalties(candidate, base)

    # Build human-readable reasoning for the submission CSV
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "N/A")
    ai_skills_count = sum(
        1 for s in candidate.get("skills", [])
        if any(core in s.get("name", "").lower() for core in AI_CORE_SKILLS)
    )
    rr = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0)
    penalty_note = f" Penalties: {'; '.join(p.name for p in penalties)}." if penalties else ""

    reasoning = (
        f"{title} with {yoe:.1f} yrs; {ai_skills_count} AI core skills; "
        f"response rate {rr:.2f}.{penalty_note}"
    )

    result = ScoringResult(
        candidate_id=cid,
        base_score=round(base, 4),
        final_score=round(final, 4),
        penalties_applied=penalties,
        reasoning=reasoning,
    )
    logger.debug("[SCORE] %s | base=%.4f | final=%.4f | penalties=%s",
                 cid, base, final, result.penalty_summary())
    return result
