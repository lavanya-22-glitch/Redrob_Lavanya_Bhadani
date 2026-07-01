"""
config.py — Single source of truth for all tunable constants, keyword sets,
and penalty / bonus multipliers used by the two-layer ranking pipeline.

Adjust values here without touching business logic in rules.py or scoring.py.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1 ─ Hard-filter constants
# ──────────────────────────────────────────────────────────────────────────────

# Firms whose presence disqualifies a candidate ONLY if their ENTIRE career
# was spent at these consulting/outsourcing shops.
BANNED_CONSULTING_FIRMS: frozenset[str] = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree",
    "persistent systems", "mastech",
})

# Titles that flag a candidate as belonging to a pure-research / academic track.
RESEARCH_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "postdoc", "post-doc", "research fellow", "phd candidate",
    "academic researcher", "research scientist", "research-only",
    "university researcher", "visiting researcher",
})

# Current-title keywords that indicate a non-engineering domain.
# Mis-aligned keyword stuffers whose *actual* domain is not AI/ML engineering.
INVALID_TRACK_KEYWORDS: frozenset[str] = frozenset({
    "marketing manager", "hr manager", "human resources manager",
    "sales manager", "sales executive", "recruiter",
    "talent acquisition", "customer support", "operations manager",
    "content writer", "graphic designer", "accountant",
    "business analyst", "civil engineer", "mechanical engineer",
    "supply chain", "logistics manager",
})

# Grace period (months) for duration_months vs actual calendar span.
# Allows ±2 months of rounding before flagging as a honeypot.
HONEYPOT_TOLERANCE_MONTHS: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 ─ Soft-penalty / scoring constants
# ──────────────────────────────────────────────────────────────────────────────

# ── Problem 5: LangChain-wrapper novice ──────────────────────────────────────
LANGCHAIN_WRAPPER_PENALTY: float = 0.40
LANGCHAIN_SHALLOW_THRESHOLD_MONTHS: int = 12   # < 12 months → novice
LEGACY_IR_KEYWORDS: frozenset[str] = frozenset({
    "search", "ranking", "bm25", "elasticsearch", "solr",
    "recommend", "recommendation", "information retrieval",
    "lucene", "opensearch", "vespa", "typesense",
})

# ── Problem 6: Architecture-only / no recent hands-on code ───────────────────
ARCH_ONLY_PENALTY: float = 0.50
ARCH_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "architect", "director of engineering", "vp of engineering",
    "chief", "head of ai", "head of ml",
})
HANDS_ON_VERBS: frozenset[str] = frozenset({
    "coded", "built", "shipped", "deployed", "implemented",
    "developed", "wrote", "engineered", "authored", "contributed",
    "refactored", "optimized", "debugged", "released",
})

# ── Problem 7: Title-chaser / job-hopper ─────────────────────────────────────
JOBHOP_PENALTY: float = 0.60
JOBHOP_AVG_TENURE_MONTHS: int = 18            # avg tenure ≤ 18 months
JOBHOP_MIN_JOBS_REQUIRED: int = 3             # need at least 3 jobs to judge

# ── Problem 8: Framework enthusiast (no systems thinking) ────────────────────
FRAMEWORK_PENALTY: float = 0.50
HIGH_LEVEL_FRAMEWORKS: frozenset[str] = frozenset({
    "langchain", "llamaindex", "flowise", "openai api",
    "autogpt", "crewai", "dspy", "haystack",
})
SYSTEMS_KEYWORDS: frozenset[str] = frozenset({
    "indexing", "quantization", "latency", "distributed",
    "sharding", "optimization", "inference", "throughput",
    "batching", "cuda", "triton", "onnx", "trt", "tensorrt",
    "kernel", "parallelism", "memory bandwidth", "compression",
})
FRAMEWORK_COUNT_THRESHOLD: int = 2            # more than 2 high-level tools

# ── Problem 9: CV / Robotics without NLP / IR footprint ─────────────────────
CV_ROBOTICS_PENALTY: float = 0.30
CV_ROBOTICS_KEYWORDS: frozenset[str] = frozenset({
    "computer vision", "yolo", "ros", "lidar", "robotics",
    "point cloud", "slam", "object detection", "pose estimation",
    "depth estimation", "stereo vision",
})
NLP_IR_KEYWORDS: frozenset[str] = frozenset({
    "nlp", "retrieval", "ranking", "transformers", "bert",
    "llm", "large language model", "text classification",
    "named entity", "ner", "sentiment", "embeddings",
    "semantic search", "vector database",
})

# ── Problem 10: Closed-source silo / no external validation ─────────────────
CLOSED_SOURCE_PENALTY: float = 0.70
PUBLIC_ARTIFACT_KEYWORDS: frozenset[str] = frozenset({
    "paper", "published", "patent", "conference", "talk",
    "arxiv", "blog", "open source", "github", "kaggle",
    "hugging face", "huggingface", "open-source",
})

# ──────────────────────────────────────────────────────────────────────────────
# POSITIVE SCORING WEIGHTS  (used in scoring.py base score calculation)
# ──────────────────────────────────────────────────────────────────────────────

# Target AI / ML core skills that earn positive signal
AI_CORE_SKILLS: frozenset[str] = frozenset({
    "machine learning", "deep learning", "nlp", "llm", "large language model",
    "transformers", "bert", "gpt", "fine-tuning", "fine-tuning llms",
    "rag", "retrieval augmented generation", "vector database",
    "pytorch", "tensorflow", "keras", "jax", "mlflow",
    "hugging face", "huggingface", "langchain", "llamaindex",
    "reinforcement learning", "rlhf", "reward modeling",
    "feature engineering", "model deployment", "mlops",
    "embeddings", "semantic search", "information retrieval",
    "knowledge distillation", "quantization",
    "cuda", "triton", "onnx", "tensorrt",
})

# Score weights (sum should be ~1.0 before penalties)
WEIGHT_AI_SKILLS: float = 0.30
WEIGHT_EXPERIENCE: float = 0.20
WEIGHT_TITLE_ALIGNMENT: float = 0.20
WEIGHT_PLATFORM_SIGNALS: float = 0.15
WEIGHT_EDUCATION: float = 0.10
WEIGHT_OPEN_SOURCE: float = 0.05

# Ideal experience band for the JD (years)
IDEAL_EXP_MIN: float = 3.0
IDEAL_EXP_MAX: float = 10.0

# Education tier multipliers
EDUCATION_TIER_SCORES: dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.85,
    "tier_3": 0.70,
    "tier_4": 0.55,
    "unknown": 0.60,
}

# Titles that signal strong positive alignment with the JD
POSITIVE_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "ml engineer", "machine learning engineer", "ai engineer",
    "nlp engineer", "research engineer", "applied scientist",
    "data scientist", "deep learning engineer", "llm engineer",
    "senior ml", "staff ml", "principal ml",
    "ai researcher", "applied ml", "applied ai",
})

# ──────────────────────────────────────────────────────────────────────────────
# SUBMISSION CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TOP_N: int = 100          # number of candidates to emit in the submission CSV
SCORE_DECIMALS: int = 4   # rounding for the score column
