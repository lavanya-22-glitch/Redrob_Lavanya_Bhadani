# redrob Candidate Ranker — Lavanya Bhadani

An Intelligent Candidate Ranking System built for the Redrob Hackathon. This pipeline processes candidates and outputs a highly refined, ranked shortlist of 100 top candidates by combining lexical history checks, tabular data scoring, and Reciprocal Rank Fusion (RRF).

## 🚀 Quickstart

To run the ranking pipeline and generate the final submission CSV, simply run the main Python script:

```bash
python submission/main.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Prerequisites
The script uses standard Python libraries but requires `pandas` for efficient data-frame ranking operations:
```bash
pip install pandas
pip install gradio
```

## 🧠 Methodology & Pipeline Layers

The ranking system is explicitly designed to avoid ML hallucinations and arbitrary template pattern-matching. It processes candidates through 6 distinct layers:

### Layer 0: Hard Filters & Timeline Honeypots
Instantly disqualifies non-viable candidates before any scoring occurs:
- Drops pure-consulting careers, misaligned titles, and academic-only profiles.
- **Honeypot Trap A:** Drops candidates claiming employment at known fictional or blacklisted companies (e.g., Initech, Pied Piper).
- **Ghost-Employee Trap B:** Identifies candidates who claim to be currently working for companies that have verifiably shut down before 2026.

### Layer 1: Targeted Lexical Score
Scans candidate career histories and skills against 3 "Must-Have" JD clusters:
1. Vector Databases (Pinecone, Weaviate, Qdrant, etc.)
2. Eval Metrics (NDCG, MRR, MAP, etc.)
3. Ranking Systems (Search, Retrieval, Recommenders, etc.)
*Note: Keywords found in active job descriptions are weighted much heavier than isolated "skills" tags.*

### Layer 2: Tabular Signal Score
Evaluates structured data signals directly aligned with the JD:
- Experience sweet-spots (5–9 years).
- Location affinity bonuses.
- Notice period calculations.
- GitHub activity tracking and recruiter response rates.

### Layer 3: Reciprocal Rank Fusion (RRF)
Instead of adding arbitrary weights to Lexical and Tabular scores directly, the system ranks candidates independently on both dimensions and blends them using the RRF formula:
`RRF_Score = 1 / (k + Rank_lexical) + 1 / (k + Rank_tabular)`
This purely ordinal approach ensures only candidates who perform exceptionally across **both** dimensions reach the top.

### Layer 4: Soft Penalties
Applies score down-weights to candidates showing signs of job-hopping or excessive framework enthusiasm over core engineering.

### Layer 5: Dynamic Reasoning (Component Assembler)
To prevent the submission from being flagged for static, copy-pasted templates during manual review, a unique, fact-backed narrative is assembled for *every* candidate. It dynamically builds the reasoning string by pulling real data points (exact YoE, specific DB tools used, exact notice period, and GitHub score metrics) directly from their schema.
