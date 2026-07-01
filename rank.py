#!/usr/bin/env python3
"""
rank.py — CLI entry point for the Redrob Intelligent Candidate Ranker.

Usage
─────
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  python rank.py --candidates ./sample_candidates.json --out ./dev_submission.csv --verbose
  python rank.py --help

The single command satisfies the hackathon's reproduce_command requirement:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup  (configures root logger before any import of src.*)
# ──────────────────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rank.py",
        description=(
            "Redrob Intelligent Candidate Ranker — two-layer pipeline.\n"
            "Layer 1: Hard filters (rules.py). "
            "Layer 2: Base score + soft penalties (scoring.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to candidates.jsonl (production) or sample_candidates.json (dev).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="PATH",
        help="Destination path for the submission CSV (e.g. ./submission.csv).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging and live progress output.",
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.verbose)
    logger = logging.getLogger("rank")

    candidates_path: Path = args.candidates.resolve()
    output_path: Path = args.out.resolve()

    if not candidates_path.exists():
        logger.error("Candidates file not found: %s", candidates_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Redrob Intelligent Candidate Ranker")
    logger.info("  Input  : %s", candidates_path)
    logger.info("  Output : %s", output_path)
    logger.info("=" * 60)

    # Deferred import so logging is configured before src modules emit messages
    from src.pipeline import run_pipeline  # noqa: PLC0415

    t0 = time.perf_counter()
    stats = run_pipeline(candidates_path, output_path, verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("DONE in %.2fs\n%s", elapsed, stats)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
