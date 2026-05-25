"""DermArbiter Experiment — Fairness Analysis.

Analyzes the fairness of DermArbiter predictions across subgroups using Fitzpatrick skin types.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dermarbiter.evaluation.fairness_analyzer import FairnessAnalyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Fairness Analysis on Experiment Results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="Path to JSONL results file.")
    parser.add_argument(
        "--group-key", default="fitzpatrick_type",
        help="Record key for demographic group.",
    )
    parser.add_argument("--output-json", default=None, help="Export fairness report to JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    analyzer = FairnessAnalyzer.from_jsonl(args.results, group_key=args.group_key)
    analyzer.print_report()

    if args.output_json:
        import json
        report = analyzer.compute_all()
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        print(f"Fairness report exported to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
