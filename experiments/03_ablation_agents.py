"""DermArbiter Experiment — Agent Ablation.

Systematically removes one agent at a time to measure their individual contributions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dermarbiter.experiments.ablation import AblationRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Agent Ablation Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--output", default="results/ablation_agent.jsonl", help="Output path.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode.")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap total cases.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    runner = AblationRunner(
        config_path=args.config,
        data_path=args.data,
        ablation_type="agent",
        output_path=args.output,
        mock=args.mock,
        max_cases=args.max_cases,
    )

    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
