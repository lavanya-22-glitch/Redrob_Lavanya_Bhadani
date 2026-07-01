"""
tests/test_rules.py — Unit tests for Layer 1 hard-filter rules.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from src.rules import evaluate_hard_filters


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — canonical candidate skeletons
# ──────────────────────────────────────────────────────────────────────────────

def _make_candidate(
    title: str = "ML Engineer",
    company: str = "Acme AI",
    companies: list[str] | None = None,
    start_date: str = "2021-01-01",
    end_date: str | None = "2024-01-01",
    duration_months: int = 36,
) -> dict:
    """
    Build a minimal, valid candidate dict for rule testing.
    Override individual fields to isolate each rule.
    """
    history = []
    if companies:
        for i, comp in enumerate(companies):
            history.append({
                "company": comp,
                "title": title,
                "start_date": f"202{i}-01-01",
                "end_date": f"202{i+1}-01-01",
                "duration_months": 12,
                "is_current": i == len(companies) - 1,
                "industry": "Software",
                "company_size": "501-1000",
                "description": "Built and shipped ML models.",
            })
    else:
        history.append({
            "company": company,
            "title": title,
            "start_date": start_date,
            "end_date": end_date,
            "duration_months": duration_months,
            "is_current": end_date is None,
            "industry": "Software",
            "company_size": "201-500",
            "description": "Built and deployed ML pipelines.",
        })

    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Senior ML Engineer",
            "summary": "AI/ML practitioner with production experience.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": title,
            "current_company": companies[-1] if companies else company,
            "current_company_size": "501-1000",
            "current_industry": "Software",
        },
        "career_history": history,
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 80,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-01-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 12,
            "skill_assessment_scores": {},
            "connection_count": 200,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Problem 1 — Honeypot tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHoneypot:
    def test_legitimate_duration_passes(self):
        c = _make_candidate(start_date="2021-01-01", end_date="2024-01-01", duration_months=36)
        result = evaluate_hard_filters(c)
        assert result.passed

    def test_inflated_duration_fails(self):
        # States 60 months but calendar span is only 36 months
        c = _make_candidate(start_date="2021-01-01", end_date="2024-01-01", duration_months=60)
        result = evaluate_hard_filters(c)
        assert not result.passed
        assert "Honeypot" in result.reason

    def test_within_tolerance_passes(self):
        # +2 months tolerance allowed
        c = _make_candidate(start_date="2021-01-01", end_date="2024-01-01", duration_months=38)
        result = evaluate_hard_filters(c)
        assert result.passed

    def test_current_job_no_end_date(self):
        # Current roles have no end_date — should use `now` and not falsely flag
        c = _make_candidate(start_date="2023-01-01", end_date=None, duration_months=20)
        result = evaluate_hard_filters(c)
        assert result.passed


# ──────────────────────────────────────────────────────────────────────────────
# Problem 2 — Pure-play IT consulting
# ──────────────────────────────────────────────────────────────────────────────

class TestPureConsulting:
    def test_all_consulting_fails(self):
        c = _make_candidate(companies=["TCS", "Infosys", "Wipro"])
        result = evaluate_hard_filters(c)
        assert not result.passed
        assert "consulting" in result.reason.lower()

    def test_mixed_career_passes(self):
        c = _make_candidate(companies=["TCS", "Infosys", "Acme AI Product Co"])
        result = evaluate_hard_filters(c)
        assert result.passed

    def test_single_product_company_passes(self):
        c = _make_candidate(companies=["Stripe"])
        result = evaluate_hard_filters(c)
        assert result.passed


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3 — Misaligned titles
# ──────────────────────────────────────────────────────────────────────────────

class TestMisalignedTitle:
    @pytest.mark.parametrize("bad_title", [
        "Marketing Manager",
        "HR Manager",
        "Sales Executive",
        "Recruiter",
        "Content Writer",
        "Operations Manager",
    ])
    def test_non_engineering_titles_fail(self, bad_title: str):
        c = _make_candidate(title=bad_title)
        result = evaluate_hard_filters(c)
        assert not result.passed
        assert "Non-engineering" in result.reason or "non-engineering" in result.reason.lower()

    @pytest.mark.parametrize("good_title", [
        "ML Engineer",
        "Senior Machine Learning Engineer",
        "AI Research Engineer",
        "Applied Scientist",
        "Data Scientist",
        "Deep Learning Engineer",
    ])
    def test_engineering_titles_pass(self, good_title: str):
        c = _make_candidate(title=good_title)
        result = evaluate_hard_filters(c)
        assert result.passed


# ──────────────────────────────────────────────────────────────────────────────
# Problem 4 — Pure research / academic drift
# ──────────────────────────────────────────────────────────────────────────────

class TestPureResearch:
    @pytest.mark.parametrize("research_title", [
        "Postdoc",
        "Research Fellow",
        "PhD Candidate",
        "Academic Researcher",
    ])
    def test_research_titles_fail(self, research_title: str):
        c = _make_candidate(title=research_title)
        result = evaluate_hard_filters(c)
        assert not result.passed
        assert "research" in result.reason.lower() or "academic" in result.reason.lower()

    def test_applied_research_engineer_passes(self):
        # "Research Engineer" should be fine (it's in POSITIVE_TITLE_KEYWORDS, not RESEARCH)
        c = _make_candidate(title="Research Engineer")
        result = evaluate_hard_filters(c)
        assert result.passed
