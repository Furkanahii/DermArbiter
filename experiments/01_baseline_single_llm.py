"""DermArbiter Experiment — Single LLM Baseline.

Evaluates a single LLM (Specialist agent) as a baseline without the debate protocol.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dermarbiter.experiments.runner import ExperimentRunner

logger = logging.getLogger("baseline_single_llm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Single LLM Baseline Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--output", default="results/baseline_single_llm.jsonl", help="Output path.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode.")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap total cases.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Instantiate the base runner
    runner = ExperimentRunner(
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        mock=args.mock,
        max_cases=args.max_cases,
    )

    # Initialize agents and tools
    if args.mock:
        runner._setup_mock()
    else:
        raise NotImplementedError("Non-mock real backend evaluation requires live GPU environment.")

    # Filter to only the Specialist agent (simulates single LLM baseline)
    if runner._agents:
        logger.info("Ablating all agents except 'specialist' to simulate Single-LLM baseline.")
        # We keep specialist as the sole agent. Moderator is normally the coordinator,
        # but in a single agent setup we bypass debate and Moderator.
        # Let's keep specialist and moderator but disable generalist/skeptic.
        filtered_agents = {}
        for role in ["specialist", "moderator"]:
            if role in runner._agents:
                filtered_agents[role] = runner._agents[role]
        runner._agents = filtered_agents

    runner._orchestrator = runner._build_orchestrator()

    # Run the experiment
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
