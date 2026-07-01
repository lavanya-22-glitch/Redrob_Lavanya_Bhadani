"""
pipeline.py — Orchestrator that wires Layer 1 (hard filters) and Layer 2
(soft scoring + penalties) into a single end-to-end ranking pipeline.

Responsibilities:
  • Stream candidates from a JSONL file (constant memory, handles 50K records).
  • Apply hard filters — dropped candidates are logged but never scored.
  • Score surviving candidates.
  • Emit the top-N ranked list as a submission-compliant CSV.
  • Report a rich audit summary to stdout and the log file.

Usage (from rank.py CLI):
    from src.pipeline import run_pipeline
    run_pipeline(candidates_path, output_path)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from src.config import SCORE_DECIMALS, TOP_N
from src.rules import FilterResult, evaluate_hard_filters
from src.scoring import ScoringResult, score_candidate

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    total_read: int = 0
    hard_dropped: int = 0
    scored: int = 0
    emitted: int = 0

    @property
    def pass_rate(self) -> float:
        return self.scored / self.total_read if self.total_read else 0.0

    def __str__(self) -> str:
        return (
            f"  Total read     : {self.total_read:,}\n"
            f"  Hard-dropped   : {self.hard_dropped:,}\n"
            f"  Scored         : {self.scored:,}\n"
            f"  Pass rate      : {self.pass_rate:.1%}\n"
            f"  Emitted (Top-N): {self.emitted:,}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stream_jsonl(path: Path) -> Generator[dict[str, Any], None, None]:
    """
    Lazily yield one candidate dict per line from a .jsonl file.
    Skips blank lines and logs malformed lines without crashing.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", line_no, exc)


def _stream_json_array(path: Path) -> Generator[dict[str, Any], None, None]:
    """
    Load and yield candidates from a plain JSON array file.
    Used for sample_candidates.json during development.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    yield from data


def _load_candidates(path: Path) -> Generator[dict[str, Any], None, None]:
    """Auto-detect JSONL vs JSON array based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _stream_jsonl(path)
    elif suffix == ".json":
        yield from _stream_json_array(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Expected .jsonl or .json")


def _write_submission_csv(results: list[ScoringResult], output_path: Path) -> None:
    """
    Write the ranked list to a submission-compliant CSV.
    Ensures:
      • Exactly TOP_N rows.
      • Non-increasing score order.
      • Tie-breaking by candidate_id ascending (per spec).
      • Scores rounded to SCORE_DECIMALS decimal places.
    """
    # Sort: primary = score descending, secondary = candidate_id ascending (tie-break)
    results.sort(key=lambda r: (-r.final_score, r.candidate_id))
    top = results[:TOP_N]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, result in enumerate(top, start=1):
            writer.writerow([
                result.candidate_id,
                rank,
                f"{result.final_score:.{SCORE_DECIMALS}f}",
                result.reasoning,
            ])

    logger.info("Submission written → %s (%d rows)", output_path, len(top))


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    candidates_path: Path,
    output_path: Path,
    *,
    verbose: bool = False,
) -> PipelineStats:
    """
    Full end-to-end pipeline: load → filter → score → rank → write CSV.

    Args:
        candidates_path : Path to candidates.jsonl or sample_candidates.json.
        output_path     : Destination path for the submission CSV.
        verbose         : If True, prints progress dots every 1,000 candidates.

    Returns:
        PipelineStats summarising what happened.
    """
    stats = PipelineStats()
    scored_results: list[ScoringResult] = []
    drop_log: list[tuple[str, str]] = []   # (candidate_id, reason)

    logger.info("Pipeline started. Reading: %s", candidates_path)

    for candidate in _load_candidates(candidates_path):
        stats.total_read += 1

        if verbose and stats.total_read % 1_000 == 0:
            print(f"\r  Processed {stats.total_read:,} candidates …", end="", flush=True)

        cid = candidate.get("candidate_id", f"UNKNOWN_{stats.total_read}")

        # ── Layer 1: Hard filters ────────────────────────────────────────────
        filter_result: FilterResult = evaluate_hard_filters(candidate)
        if not filter_result.passed:
            stats.hard_dropped += 1
            drop_log.append((cid, filter_result.reason))
            continue

        # ── Layer 2: Score + soft penalties ─────────────────────────────────
        result: ScoringResult = score_candidate(candidate)
        scored_results.append(result)
        stats.scored += 1

    if verbose:
        print()  # newline after progress dots

    if not scored_results:
        logger.error("No candidates survived the hard filters. Cannot produce submission.")
        sys.exit(1)

    # ── Emit top-N ──────────────────────────────────────────────────────────
    _write_submission_csv(scored_results, output_path)
    stats.emitted = min(len(scored_results), TOP_N)

    logger.info("Pipeline complete.\n%s", stats)

    # Optionally dump drop log alongside the submission
    drop_log_path = output_path.with_name(output_path.stem + "_dropped.csv")
    _write_drop_log(drop_log, drop_log_path)

    return stats


def _write_drop_log(drop_log: list[tuple[str, str]], path: Path) -> None:
    """Write a secondary CSV listing every hard-dropped candidate and why."""
    if not drop_log:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "drop_reason"])
        writer.writerows(drop_log)
    logger.info("Drop log written → %s (%d entries)", path, len(drop_log))
