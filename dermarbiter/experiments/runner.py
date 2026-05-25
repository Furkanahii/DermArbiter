"""DermArbiter Benchmark Runner — Execute the full pipeline on JSONL test cases.

Loads cases from a JSONL file, runs each through the DermArbiterOrchestrator,
collects telemetry, and writes structured results for downstream analysis.

Usage::

    python -m dermarbiter.experiments.runner \
        --config configs/default.yaml \
        --data   data/sample_cases.jsonl \
        --output results/run_001.jsonl \
        --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dermarbiter.core.blackboard import BlackboardState
from dermarbiter.core.orchestrator import DermArbiterOrchestrator
from dermarbiter.tools.base_tool import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cases(path: str, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read JSONL file and return a list of case dicts.

    Each line must be a JSON object with at least:
        case_id, query, image_path, patient_context, ground_truth_label
    """
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
                continue
            cases.append(obj)
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run the full DermArbiter pipeline on a dataset of clinical cases.

    .. note:: Previously named ``BenchmarkRunner``. The alias
       ``BenchmarkRunner = ExperimentRunner`` is kept for backwards
       compatibility but new code should use ``ExperimentRunner``.

    Args:
        config_path: Path to the YAML config directory (or single file).
        data_path:   Path to the JSONL dataset.
        output_path: Path where results JSONL will be written.
        mock:        If True, use mock agents and tools (no GPU/API needed).
        max_cases:   Limit the number of cases processed.
    """

    def __init__(
        self,
        config_path: str,
        data_path: str,
        output_path: str = "results/benchmark.jsonl",
        mock: bool = False,
        max_cases: Optional[int] = None,
    ) -> None:
        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.mock = mock
        self.max_cases = max_cases

        self._agents: Optional[Dict[str, Any]] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._orchestrator: Optional[DermArbiterOrchestrator] = None

    # ----- Setup -----------------------------------------------------------

    def _setup_mock(self) -> None:
        """Configure mock agents and tools for CPU-only testing."""
        from dermarbiter.core.mock_factory import create_mock_agents, create_mock_registry

        self._agents = create_mock_agents()
        self._tool_registry = create_mock_registry()

    def _build_orchestrator(self) -> DermArbiterOrchestrator:
        """Build (or rebuild) the orchestrator from current agents/tools."""
        if self._agents is None or self._tool_registry is None:
            raise RuntimeError("Agents and tool registry must be initialised first.")
        return DermArbiterOrchestrator(
            agents=self._agents,
            tool_registry=self._tool_registry,
        )

    # ----- Single-case execution ------------------------------------------

    def _run_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one case through the orchestrator and return result dict."""
        assert self._orchestrator is not None

        initial_state = BlackboardState(
            case_id=case.get("case_id", "UNKNOWN"),
            query=case.get("query", ""),
            image_path=case.get("image_path"),
            patient_context=case.get("patient_context", {}),
        )

        t0 = time.monotonic()
        final_state = self._orchestrator.run(initial_state)
        latency_ms = (time.monotonic() - t0) * 1000.0

        predicted = (
            final_state.final_diagnosis[0]
            if final_state.final_diagnosis
            else ""
        )

        # Count debate rounds from the log
        debate_rounds = set()
        for turn in final_state.debate_log:
            debate_rounds.add(turn.round_num)

        return {
            "case_id": case.get("case_id", "UNKNOWN"),
            "predicted": predicted,
            "ground_truth": case.get("ground_truth_label", ""),
            "final_diagnosis": list(final_state.final_diagnosis),
            "consensus_score": final_state.consensus_score,
            "early_exit": final_state.early_exit,
            "num_debate_rounds": len(debate_rounds),
            "total_tokens": final_state.total_tokens,
            "latency_ms": round(latency_ms, 2),
        }

    # ----- Main entry point -----------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        """Execute the benchmark and write results to the output file.

        Returns:
            List of per-case result dicts.
        """
        # Setup
        if self.mock:
            self._setup_mock()
        else:
            raise NotImplementedError(
                "Non-mock mode requires live LLM backends. "
                "Use --mock for CPU-only testing."
            )

        self._orchestrator = self._build_orchestrator()

        # Load cases
        cases = _load_cases(self.data_path, max_cases=self.max_cases)
        if not cases:
            logger.warning("No cases loaded from %s", self.data_path)
            return []

        logger.info("Loaded %d cases from %s", len(cases), self.data_path)

        # Ensure output directory exists
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        results: List[Dict[str, Any]] = []

        with open(self.output_path, "w", encoding="utf-8") as fh:
            for idx, case in enumerate(cases, 1):
                case_id = case.get("case_id", f"case_{idx}")
                logger.info(
                    "[%d/%d] Running case %s …", idx, len(cases), case_id
                )
                try:
                    result = self._run_single_case(case)
                except Exception as exc:
                    logger.error("Case %s failed: %s", case_id, exc, exc_info=True)
                    result = {
                        "case_id": case_id,
                        "predicted": "",
                        "ground_truth": case.get("ground_truth_label", ""),
                        "consensus_score": 0.0,
                        "early_exit": False,
                        "num_debate_rounds": 0,
                        "total_tokens": 0,
                        "latency_ms": 0.0,
                        "error": str(exc),
                    }

                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)

        logger.info(
            "Benchmark complete. %d results written to %s",
            len(results),
            self.output_path,
        )
        return results


# Backwards compatibility alias
BenchmarkRunner = ExperimentRunner


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DermArbiter Benchmark Runner",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file or directory."
    )
    parser.add_argument(
        "--data", required=True, help="Path to JSONL dataset file."
    )
    parser.add_argument(
        "--output",
        default="results/benchmark.jsonl",
        help="Output JSONL path (default: results/benchmark.jsonl).",
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

    runner = ExperimentRunner(
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        mock=args.mock,
        max_cases=args.max_cases,
    )
    runner.run()


if __name__ == "__main__":
    main()
