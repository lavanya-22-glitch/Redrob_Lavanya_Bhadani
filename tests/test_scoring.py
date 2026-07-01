"""
tests/test_scoring.py — Unit tests for Layer 2 soft-penalty functions.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from src.scoring import calculate_advanced_penalties, compute_base_score, score_candidate


# ──────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────────────

def _base_candidate(
    title: str = "ML Engineer",
    yoe: float = 6.0,
    skills: list[dict] | None = None,
    history: list[dict] | None = None,
    github_score: float = 55.0,
) -> dict:
    return {
        "candidate_id": "CAND_TEST",
        "profile": {
            "anonymized_name": "Test",
            "headline": "ML Engineer",
            "summary": "Production ML engineer with NLP and IR experience.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": "Acme AI",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": history or [
            {
                "company": "Acme AI",
                "title": title,
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": (
                    "Built and shipped NLP retrieval models. "
                    "Implemented BM25 and semantic search pipelines."
                ),
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2016,
                "end_year": 2020,
                "grade": "9.0 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": skills or [
            {"name": "NLP", "proficiency": "expert", "endorsements": 45, "duration_months": 40},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
            {"name": "Transformers", "proficiency": "advanced", "endorsements": 25, "duration_months": 30},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 30,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.75,
            "avg_response_time_hours": 10,
            "skill_assessment_scores": {},
            "connection_count": 400,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 60},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": github_score,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 10,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Base score tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBaseScore:
    def test_strong_candidate_scores_high(self):
        c = _base_candidate()
        score = compute_base_score(c)
        assert score >= 0.55, f"Expected ≥0.55, got {score}"

    def test_score_is_bounded(self):
        c = _base_candidate()
        score = compute_base_score(c)
        assert 0.0 <= score <= 1.0

    def test_more_ai_skills_increases_score(self):
        few_skills = _base_candidate(skills=[
            {"name": "NLP", "proficiency": "beginner", "endorsements": 1, "duration_months": 5},
        ])
        many_skills = _base_candidate(skills=[
            {"name": k, "proficiency": "expert", "endorsements": 40, "duration_months": 30}
            for k in ["NLP", "PyTorch", "Transformers", "LLM", "RAG", "MLflow",
                      "Fine-tuning LLMs", "RLHF", "Embeddings", "MLOps"]
        ])
        assert compute_base_score(many_skills) > compute_base_score(few_skills)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 5 — LangChain novice penalty
# ──────────────────────────────────────────────────────────────────────────────

class TestLangChainNovicePenalty:
    def test_langchain_no_ir_shallow_penalized(self):
        skills = [
            {"name": "LangChain", "proficiency": "intermediate",
             "endorsements": 5, "duration_months": 8},
        ]
        history = [{
            "company": "Startup",
            "title": "AI Developer",
            "start_date": "2024-01-01",
            "end_date": None,
            "duration_months": 8,
            "is_current": True,
            "industry": "Software",
            "company_size": "1-10",
            "description": "Used LangChain to build chatbot demos and prototypes.",
        }]
        c = _base_candidate(skills=skills, history=history)
        base = compute_base_score(c)
        final, penalties = calculate_advanced_penalties(c, base)
        penalty_names = [p.name for p in penalties]
        assert "LangChainNovice" in penalty_names
        assert final < base

    def test_langchain_with_ir_foundations_not_penalized(self):
        skills = [
            {"name": "LangChain", "proficiency": "advanced",
             "endorsements": 20, "duration_months": 6},
        ]
        history = [{
            "company": "Acme Search",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 36,
            "is_current": True,
            "industry": "Software",
            "company_size": "201-500",
            "description": (
                "Built production search ranking system using Elasticsearch and BM25. "
                "Now integrating LangChain for RAG over the search corpus."
            ),
        }]
        c = _base_candidate(skills=skills, history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "LangChainNovice" for p in penalties)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 6 — Architecture-only penalty
# ──────────────────────────────────────────────────────────────────────────────

class TestArchitectureOnlyPenalty:
    def test_pure_architect_no_verbs_penalized(self):
        history = [{
            "company": "Big Corp",
            "title": "Principal Architect",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 48,
            "is_current": True,
            "industry": "Software",
            "company_size": "10001+",
            "description": (
                "Led architecture reviews. Defined technical strategy. "
                "Communicated with stakeholders on platform direction."
            ),
        }]
        c = _base_candidate(title="Principal Architect", history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert any(p.name == "ArchitectureOnly" for p in penalties)

    def test_hands_on_architect_not_penalized(self):
        history = [{
            "company": "Startup",
            "title": "Senior Architect",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "Software",
            "company_size": "51-200",
            "description": (
                "Architected and personally built the core inference engine. "
                "Deployed the system to production on Kubernetes."
            ),
        }]
        c = _base_candidate(title="Senior Architect", history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "ArchitectureOnly" for p in penalties)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 7 — Job hopper penalty
# ──────────────────────────────────────────────────────────────────────────────

class TestJobHopperPenalty:
    def test_short_average_tenure_penalized(self):
        history = [
            {
                "company": f"Company{i}", "title": "ML Engineer",
                "start_date": f"202{i}-01-01", "end_date": f"202{i}-12-01",
                "duration_months": 11, "is_current": i == 3,
                "industry": "Software", "company_size": "51-200",
                "description": "Built ML stuff.",
            }
            for i in range(4)
        ]
        c = _base_candidate(history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert any(p.name == "JobHopper" for p in penalties)

    def test_stable_tenure_not_penalized(self):
        history = [
            {
                "company": f"Company{i}", "title": "ML Engineer",
                "start_date": f"201{i}-01-01", "end_date": f"202{i}-01-01",
                "duration_months": 24, "is_current": i == 2,
                "industry": "Software", "company_size": "201-500",
                "description": "Built and deployed NLP systems.",
            }
            for i in range(3)
        ]
        c = _base_candidate(history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "JobHopper" for p in penalties)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 8 — Framework enthusiast penalty
# ──────────────────────────────────────────────────────────────────────────────

class TestFrameworkEnthusiastPenalty:
    def test_framework_heavy_no_systems_penalized(self):
        skills = [
            {"name": k, "proficiency": "intermediate", "endorsements": 5, "duration_months": 8}
            for k in ["LangChain", "LlamaIndex", "Flowise", "CrewAI"]
        ]
        history = [{
            "company": "Demo Co",
            "title": "AI Developer",
            "start_date": "2024-01-01",
            "end_date": None,
            "duration_months": 12,
            "is_current": True,
            "industry": "Software",
            "company_size": "1-10",
            "description": "Used LangChain and LlamaIndex to build demos. Flowise pipelines.",
        }]
        c = _base_candidate(skills=skills, history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert any(p.name == "FrameworkEnthusiast" for p in penalties)

    def test_systems_keywords_redeem_framework_usage(self):
        skills = [
            {"name": k, "proficiency": "advanced", "endorsements": 20, "duration_months": 20}
            for k in ["LangChain", "LlamaIndex", "Flowise"]
        ]
        history = [{
            "company": "Infra AI",
            "title": "ML Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 40,
            "is_current": True,
            "industry": "Software",
            "company_size": "201-500",
            "description": (
                "Optimised model inference latency via quantization and ONNX export. "
                "Built distributed indexing pipeline with sharding. Used LangChain for orchestration."
            ),
        }]
        c = _base_candidate(skills=skills, history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "FrameworkEnthusiast" for p in penalties)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 9 — CV / Robotics only penalty
# ──────────────────────────────────────────────────────────────────────────────

class TestCVRoboticsOnlyPenalty:
    def test_cv_only_no_nlp_penalized(self):
        skills = [
            {"name": k, "proficiency": "advanced", "endorsements": 20, "duration_months": 30}
            for k in ["Computer Vision", "YOLO", "ROS"]
        ]
        history = [{
            "company": "Robot Co",
            "title": "Computer Vision Engineer",
            "start_date": "2019-01-01",
            "end_date": None,
            "duration_months": 60,
            "is_current": True,
            "industry": "Robotics",
            "company_size": "51-200",
            "description": "Built YOLO-based detection pipelines. SLAM and LIDAR integration.",
        }]
        c = _base_candidate(title="Computer Vision Engineer", skills=skills, history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert any(p.name == "CVRoboticsOnly" for p in penalties)

    def test_cv_with_nlp_not_penalized(self):
        skills = [
            {"name": k, "proficiency": "advanced", "endorsements": 20, "duration_months": 30}
            for k in ["Computer Vision", "BERT", "NLP", "Transformers"]
        ]
        history = [{
            "company": "Multimodal AI",
            "title": "ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 50,
            "is_current": True,
            "industry": "Software",
            "company_size": "201-500",
            "description": "Multimodal models combining vision and NLP. LLM + retrieval pipelines.",
        }]
        c = _base_candidate(skills=skills, history=history)
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "CVRoboticsOnly" for p in penalties)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 10 — Closed-source silo penalty
# ──────────────────────────────────────────────────────────────────────────────

class TestClosedSourceSiloPenalty:
    def test_no_github_no_artifacts_penalized(self):
        c = _base_candidate(github_score=-1)
        # Ensure no public keywords in description
        c["career_history"][0]["description"] = "Internal work at bank. All NDA'd."
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert any(p.name == "ClosedSourceSilo" for p in penalties)

    def test_no_github_but_has_paper_not_penalized(self):
        c = _base_candidate(github_score=-1)
        c["career_history"][0]["description"] = (
            "Published a paper at ACL on retrieval augmented generation. "
            "Conference talk at NeurIPS 2024."
        )
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "ClosedSourceSilo" for p in penalties)

    def test_has_github_not_penalized(self):
        c = _base_candidate(github_score=72.0)
        c["career_history"][0]["description"] = "Internal model work, NDA."
        base = compute_base_score(c)
        _, penalties = calculate_advanced_penalties(c, base)
        assert not any(p.name == "ClosedSourceSilo" for p in penalties)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: score_candidate full pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreCandidateIntegration:
    def test_returns_scoring_result(self):
        from src.scoring import ScoringResult
        c = _base_candidate()
        result = score_candidate(c)
        assert isinstance(result, ScoringResult)
        assert result.candidate_id == "CAND_TEST"
        assert 0.0 <= result.final_score <= 1.0
        assert result.reasoning

    def test_multiple_penalties_compound(self):
        # Build a candidate that should trigger multiple penalties
        skills = [
            {"name": k, "proficiency": "intermediate", "endorsements": 3, "duration_months": 6}
            for k in ["LangChain", "LlamaIndex", "Flowise"]
        ]
        history = [
            {
                "company": f"Co{i}", "title": "AI Dev",
                "start_date": f"202{i}-01-01", "end_date": f"202{i}-11-01",
                "duration_months": 10, "is_current": i == 3,
                "industry": "Software", "company_size": "1-10",
                "description": "LangChain demos. ChatGPT integrations.",
            }
            for i in range(4)
        ]
        c = _base_candidate(skills=skills, history=history, github_score=-1)
        c["career_history"][-1]["description"] += " Internal NDA work only."
        result = score_candidate(c)
        assert result.final_score < 0.5, (
            f"Multiple penalties should drastically reduce score, got {result.final_score}"
        )
