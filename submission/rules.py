"""
rules.py — Layer 1: Hard-filter (binary drop) engine.

Each rule is a standalone function that returns (passed: bool, reason: str).
The orchestrator `evaluate_hard_filters` chains them in priority order and
short-circuits on the first failure, keeping the hot path O(k) rather than O(n).

Adding a new hard rule:
    1. Write a function `_check_<name>(candidate) -> tuple[bool, str]`.
    2. Append it to the _RULES registry at the bottom of this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import (
    BANNED_CONSULTING_FIRMS,
    HONEYPOT_TOLERANCE_MONTHS,
    INVALID_TRACK_KEYWORDS,
    RESEARCH_TITLE_KEYWORDS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FilterResult:
    passed: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> "FilterResult":
        return cls(passed=True)

    @classmethod
    def fail(cls, reason: str) -> "FilterResult":
        return cls(passed=False, reason=reason)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> datetime | None:
    """Safely parse a YYYY-MM-DD date string. Returns None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.debug("Could not parse date: %r", date_str)
        return None


def _calendar_months(start: datetime, end: datetime) -> int:
    """Number of whole months between two datetimes."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def _title_lower(candidate: dict[str, Any]) -> str:
    return candidate.get("profile", {}).get("current_title", "").lower().strip()


# ──────────────────────────────────────────────────────────────────────────────
# Individual hard-filter rules
# ──────────────────────────────────────────────────────────────────────────────

def _check_honeypot(candidate: dict[str, Any]) -> FilterResult:
    """
    PROBLEM 1 — Timeline Honeypots.

    Flags candidates whose stated `duration_months` exceeds the real calendar
    span by more than HONEYPOT_TOLERANCE_MONTHS.  This is a strong indicator of
    fabricated experience that cannot be fixed by scoring alone.
    """
    history: list[dict] = candidate.get("career_history", [])
    now = datetime.now()

    for job in history:
        stated_months: int = job.get("duration_months", 0)
        start = _parse_date(job.get("start_date"))
        if start is None:
            continue  # can't verify — skip rather than falsely flag

        end_raw = job.get("end_date")
        end = _parse_date(end_raw) if end_raw else now

        real_months = _calendar_months(start, end)
        if stated_months > (real_months + HONEYPOT_TOLERANCE_MONTHS):
            return FilterResult.fail(
                f"Honeypot: Stated duration ({stated_months} months) exceeds "
                f"real calendar time ({real_months} months) at '{job.get('company', '?')}'"
            )

    return FilterResult.ok()


def _check_pure_consulting(candidate: dict[str, Any]) -> FilterResult:
    """
    PROBLEM 2 — Pure-Play IT Consulting (entire career).

    Only disqualifies when EVERY recorded job is at a banned outsourcing firm.
    A single stint at a product company redeems the candidate.
    """
    history: list[dict] = candidate.get("career_history", [])
    if not history:
        return FilterResult.ok()

    companies = [job.get("company", "").lower().strip() for job in history]

    def _is_banned(company_name: str) -> bool:
        return any(firm in company_name for firm in BANNED_CONSULTING_FIRMS)

    if all(_is_banned(c) for c in companies):
        return FilterResult.fail(
            "Disqualified: Entire career spent at IT outsourcing / consulting firms — "
            "no product-company exposure detected."
        )

    return FilterResult.ok()


def _check_misaligned_title(candidate: dict[str, Any]) -> FilterResult:
    """
    PROBLEM 3 — Misaligned Title / Non-Engineering Domain.

    Drops candidates whose current title belongs to a non-technical domain,
    regardless of how many AI keywords are stuffed into their skill list.
    """
    title = _title_lower(candidate)
    if not title:
        return FilterResult.ok()

    matched = next(
        (kw for kw in INVALID_TRACK_KEYWORDS if kw in title),
        None,
    )
    if matched:
        return FilterResult.fail(
            f"Disqualified: Current title '{title}' is a non-engineering domain "
            f"(matched keyword: '{matched}')."
        )

    return FilterResult.ok()


def _check_pure_research(candidate: dict[str, Any]) -> FilterResult:
    """
    PROBLEM 4 — Pure Research / Academic Labs.

    Rejects candidates currently in a pure-research or academic role with no
    production engineering track record implied.
    """
    title = _title_lower(candidate)
    matched = next(
        (kw for kw in RESEARCH_TITLE_KEYWORDS if kw in title),
        None,
    )
    if matched:
        return FilterResult.fail(
            f"Disqualified: Current title '{title}' is a pure academic / research role "
            f"(matched keyword: '{matched}'). Needs production AI engineering track."
        )

    return FilterResult.ok()


# ──────────────────────────────────────────────────────────────────────────────
# Rule registry — order matters: cheapest / most decisive first
# ──────────────────────────────────────────────────────────────────────────────

_RULES = [
    _check_misaligned_title,   # O(1) — cheapest check first
    _check_pure_research,      # O(1)
    _check_pure_consulting,    # O(jobs)
    _check_honeypot,           # O(jobs) — most expensive, last
]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_hard_filters(candidate: dict[str, Any]) -> FilterResult:
    """
    Run all hard-filter rules against a candidate, short-circuiting on the
    first failure.

    Args:
        candidate: A single candidate dict conforming to candidate_schema.json.

    Returns:
        FilterResult(passed=True, reason="")           — candidate survives
        FilterResult(passed=False, reason="<detail>")  — candidate is dropped
    """
    cid = candidate.get("candidate_id", "<unknown>")
    for rule_fn in _RULES:
        result = rule_fn(candidate)
        if not result.passed:
            logger.debug("[HARD FILTER] DROPPED %s | %s | %s",
                         cid, rule_fn.__name__, result.reason)
            return result

    logger.debug("[HARD FILTER] PASSED %s", cid)
    return FilterResult.ok()
