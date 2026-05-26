"""DermArbiter Full-Panel Runner — Complete debate system evaluation.

Runs the full DermArbiter debate panel (all agents + all tools) on every
case in the dataset.  This is the primary evaluation entry-point that
exercises the complete system end-to-end.

Usage::

    python -m dermarbiter.experiments.full_panel \
        --config configs/default.yaml \
        --data   data/sample_cases.jsonl \
        --output results/dermarbiter_full.jsonl \
        --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dermarbiter.core.blackboard import BlackboardState
from dermarbiter.core.orchestrator import DermArbiterOrchestrator
from dermarbiter.experiments.runner import ExperimentRunner, _load_cases
from dermarbiter.tools.base_tool import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FullPanelRunner
# ---------------------------------------------------------------------------

class FullPanelRunner:
    """Run the full DermArbiter debate panel for batch evaluation.

    Instantiates *all* agents and *all* tools, then runs the complete
    multi-agent debate protocol on each case.  Results are written in the
    same JSONL schema used by :class:`BaselineRunner` and
    :class:`AblationRunner` so that downstream analysis tools work
    uniformly.

    Args:
        config_path:  Path to the YAML config file or directory.
        data_path:    Path to the JSONL dataset.
        output_path:  Path for the full-panel results JSONL.
        mock:         Use mock agents/tools (CPU-only testing).
        max_cases:    Limit the number of cases to process.
    """

    def __init__(
        self,
        config_path: str,
        data_path: str,
        output_path: str = "results/dermarbiter_full.jsonl",
        mock: bool = False,
        max_cases: Optional[int] = None,
    ) -> None:
        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.mock = mock
        self.max_cases = max_cases

    # ----- Setup -----------------------------------------------------------

    def _build_full_orchestrator(self) -> DermArbiterOrchestrator:
        """Create an orchestrator with the complete agent panel and tools.

        Returns:
            A ``DermArbiterOrchestrator`` with all agents and all tools
            registered (i.e., the default production configuration).
        """
        from dermarbiter.core.mock_factory import (
            create_mock_agents,
            create_mock_registry,
        )

        agents = create_mock_agents()
        registry = create_mock_registry()

        logger.info(
            "Full-panel setup: %d agents (%s), %d tools (%s)",
            len(agents),
            sorted(agents),
            len(registry.tool_names),
            sorted(registry.tool_names),
        )

        return DermArbiterOrchestrator(
            agents=agents,
            tool_registry=registry,
        )

    # ----- Single-case execution ------------------------------------------

    def _run_single_case(
        self,
        orchestrator: DermArbiterOrchestrator,
        case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one case through the full-panel orchestrator.

        Args:
            orchestrator: The full-panel orchestrator.
            case:         A case dict loaded from JSONL.

        Returns:
            A result dict matching the standard experiment JSONL schema.
        """
        initial_state = BlackboardState(
            case_id=case.get("case_id", "UNKNOWN"),
            query=case.get("query", ""),
            image_path=case.get("image_path"),
            patient_context=case.get("patient_context", {}),
        )

        t0 = time.monotonic()
        try:
            final_state = orchestrator.run(initial_state)
            latency_ms = (time.monotonic() - t0) * 1000.0

            predicted = (
                final_state.final_diagnosis[0]
                if final_state.final_diagnosis
                else ""
            )
            debate_rounds = len(set(t.round_num for t in final_state.debate_log))

            return {
                "experiment": "dermarbiter_full",
                "case_id": case.get("case_id", "UNKNOWN"),
                "predicted": predicted,
                "ground_truth": case.get("ground_truth_label", ""),
                "final_diagnosis": list(final_state.final_diagnosis),
                "consensus_score": final_state.consensus_score,
                "early_exit": final_state.early_exit,
                "num_debate_rounds": debate_rounds,
                "total_tokens": final_state.total_tokens,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "Full-panel case %s failed: %s",
                case.get("case_id"),
                exc,
            )
            return {
                "experiment": "dermarbiter_full",
                "case_id": case.get("case_id", "UNKNOWN"),
                "predicted": "",
                "ground_truth": case.get("ground_truth_label", ""),
                "consensus_score": 0.0,
                "early_exit": False,
                "num_debate_rounds": 0,
                "total_tokens": 0,
                "latency_ms": round(latency_ms, 2),
                "error": str(exc),
            }

    # ----- Main entry point ------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        """Execute the full-panel evaluation and write results.

        Returns:
            List of per-case result dicts.
        """
        cases = _load_cases(self.data_path, max_cases=self.max_cases)
        if not cases:
            logger.warning("No cases loaded from %s", self.data_path)
            return []

        logger.info(
            "Running full DermArbiter panel on %d cases", len(cases)
        )

        orchestrator = self._build_full_orchestrator()

        # Ensure output directory exists
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        results: List[Dict[str, Any]] = []

        with open(self.output_path, "w", encoding="utf-8") as fh:
            for idx, case in enumerate(cases, 1):
                case_id = case.get("case_id", f"case_{idx}")
                logger.info(
                    "[%d/%d] Running full-panel case %s …",
                    idx,
                    len(cases),
                    case_id,
                )
                result = self._run_single_case(orchestrator, case)
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)

        logger.info(
            "Full-panel complete. %d results written to %s",
            len(results),
            self.output_path,
        )
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DermArbiter Full-Panel Debate Runner",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file or directory."
    )
    parser.add_argument(
        "--data", required=True, help="Path to JSONL dataset file."
    )
    parser.add_argument(
        "--output",
        default="results/dermarbiter_full.jsonl",
        help="Output JSONL path (default: results/dermarbiter_full.jsonl).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock agents/tools for CPU-only testing.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the number of cases to process.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = FullPanelRunner(
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        mock=args.mock,
        max_cases=args.max_cases,
    )
    runner.run()


if __name__ == "__main__":
    main()
