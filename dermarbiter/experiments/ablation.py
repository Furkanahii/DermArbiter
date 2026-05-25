"""DermArbiter Ablation Runner — Systematic ablation over agents, tools, and rounds.

Runs the full pipeline with controlled variations to measure the contribution
of individual components.

Supported ablation types:
    • ``agent``  — remove one agent at a time from the panel
    • ``tool``   — remove one tool at a time from the registry
    • ``round``  — vary ``max_rounds`` (1, 2, 3, 5)

Usage::

    python -m dermarbiter.experiments.ablation \
        --config configs/default.yaml \
        --data   data/sample_cases.jsonl \
        --ablation-type agent \
        --output results/ablation_agent.jsonl \
        --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from dermarbiter.core.blackboard import BlackboardState
from dermarbiter.core.orchestrator import DermArbiterOrchestrator
from dermarbiter.experiments.runner import ExperimentRunner, _load_cases
from dermarbiter.tools.base_tool import ToolRegistry

logger = logging.getLogger(__name__)

# Ablation type constants
AGENT_ABLATION = "agent"
TOOL_ABLATION = "tool"
ROUND_ABLATION = "round"
VALID_ABLATION_TYPES = {AGENT_ABLATION, TOOL_ABLATION, ROUND_ABLATION}

# Default round values for round ablation
DEFAULT_ROUND_VALUES = [1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

class AblationConfig:
    """Describes a single ablation experiment variant.

    Args:
        ablation_type: One of ``agent``, ``tool``, ``round``.
        variant_name:  Human-readable label for this variant.
        removed_agents: Agents to exclude (agent ablation).
        removed_tools:  Tools to exclude (tool ablation).
        max_rounds:     Override for max debate rounds (round ablation).
    """

    def __init__(
        self,
        ablation_type: str,
        variant_name: str,
        removed_agents: Optional[List[str]] = None,
        removed_tools: Optional[List[str]] = None,
        max_rounds: Optional[int] = None,
    ) -> None:
        if ablation_type not in VALID_ABLATION_TYPES:
            raise ValueError(
                f"Invalid ablation_type {ablation_type!r}. "
                f"Must be one of: {sorted(VALID_ABLATION_TYPES)}"
            )
        self.ablation_type = ablation_type
        self.variant_name = variant_name
        self.removed_agents = removed_agents or []
        self.removed_tools = removed_tools or []
        self.max_rounds = max_rounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ablation_type": self.ablation_type,
            "variant_name": self.variant_name,
            "removed_agents": self.removed_agents,
            "removed_tools": self.removed_tools,
            "max_rounds": self.max_rounds,
        }

    def __repr__(self) -> str:
        return f"<AblationConfig {self.variant_name!r} type={self.ablation_type!r}>"


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Run systematic ablation experiments over the DermArbiter pipeline.

    Args:
        config_path:    Path to the YAML config directory/file.
        data_path:      Path to the JSONL dataset.
        ablation_type:  One of ``agent``, ``tool``, ``round``.
        output_path:    Path for the combined ablation results JSONL.
        mock:           Use mock agents/tools.
        max_cases:      Limit cases per variant.
        round_values:   Custom round values for round ablation.
    """

    def __init__(
        self,
        config_path: str,
        data_path: str,
        ablation_type: str = AGENT_ABLATION,
        output_path: str = "results/ablation.jsonl",
        mock: bool = False,
        max_cases: Optional[int] = None,
        round_values: Optional[List[int]] = None,
    ) -> None:
        if ablation_type not in VALID_ABLATION_TYPES:
            raise ValueError(
                f"Invalid ablation_type {ablation_type!r}. "
                f"Must be one of: {sorted(VALID_ABLATION_TYPES)}"
            )
        self.config_path = config_path
        self.data_path = data_path
        self.ablation_type = ablation_type
        self.output_path = output_path
        self.mock = mock
        self.max_cases = max_cases
        self.round_values = round_values or DEFAULT_ROUND_VALUES

    # ----- Variant generation ----------------------------------------------

    def _generate_variants(self) -> List[AblationConfig]:
        """Generate ablation variants based on the ablation type."""
        if self.ablation_type == AGENT_ABLATION:
            return self._agent_variants()
        elif self.ablation_type == TOOL_ABLATION:
            return self._tool_variants()
        elif self.ablation_type == ROUND_ABLATION:
            return self._round_variants()
        else:
            raise ValueError(f"Unknown ablation type: {self.ablation_type}")

    def _agent_variants(self) -> List[AblationConfig]:
        """Generate one variant per removable agent (skip moderator)."""
        # Baseline (all agents)
        variants = [
            AblationConfig(
                ablation_type=AGENT_ABLATION,
                variant_name="baseline_all_agents",
            )
        ]
        # Remove one non-moderator agent at a time
        for agent_role in ["specialist", "generalist", "skeptic"]:
            variants.append(
                AblationConfig(
                    ablation_type=AGENT_ABLATION,
                    variant_name=f"no_{agent_role}",
                    removed_agents=[agent_role],
                )
            )
        return variants

    def _tool_variants(self) -> List[AblationConfig]:
        """Generate one variant per removable tool."""
        # Get tool names from mock registry
        from dermarbiter.core.mock_factory import create_mock_registry

        registry = create_mock_registry()
        tool_names = registry.tool_names

        variants = [
            AblationConfig(
                ablation_type=TOOL_ABLATION,
                variant_name="baseline_all_tools",
            )
        ]
        for tool_name in tool_names:
            variants.append(
                AblationConfig(
                    ablation_type=TOOL_ABLATION,
                    variant_name=f"no_{tool_name}",
                    removed_tools=[tool_name],
                )
            )
        return variants

    def _round_variants(self) -> List[AblationConfig]:
        """Generate one variant per max_rounds value."""
        return [
            AblationConfig(
                ablation_type=ROUND_ABLATION,
                variant_name=f"max_rounds_{r}",
                max_rounds=r,
            )
            for r in self.round_values
        ]

    # ----- Single-variant execution ----------------------------------------

    def _run_variant(
        self,
        variant: AblationConfig,
        cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run a single ablation variant on all cases."""
        from dermarbiter.core.mock_factory import create_mock_agents, create_mock_registry

        agents = create_mock_agents()
        registry = create_mock_registry()

        # Apply agent ablation
        for role in variant.removed_agents:
            agents.pop(role, None)

        # Apply tool ablation — rebuild registry without removed tools
        if variant.removed_tools:
            filtered_registry = ToolRegistry()
            for tool_name in registry.tool_names:
                if tool_name not in variant.removed_tools:
                    filtered_registry.register(registry.get(tool_name))
            registry = filtered_registry

        # Build orchestrator with optional round override
        kwargs: Dict[str, Any] = {
            "agents": agents,
            "tool_registry": registry,
        }
        if variant.max_rounds is not None:
            kwargs["max_rounds"] = variant.max_rounds

        orchestrator = DermArbiterOrchestrator(**kwargs)

        results: List[Dict[str, Any]] = []
        for case in cases:
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

                result = {
                    "variant": variant.variant_name,
                    "ablation_type": variant.ablation_type,
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
                    "Variant %s / case %s failed: %s",
                    variant.variant_name,
                    case.get("case_id"),
                    exc,
                )
                result = {
                    "variant": variant.variant_name,
                    "ablation_type": variant.ablation_type,
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

            results.append(result)
        return results

    # ----- Main entry point ------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        """Execute all ablation variants and write combined results.

        Returns:
            Combined list of per-case result dicts across all variants.
        """
        cases = _load_cases(self.data_path, max_cases=self.max_cases)
        if not cases:
            logger.warning("No cases loaded from %s", self.data_path)
            return []

        variants = self._generate_variants()
        logger.info(
            "Running %d ablation variants (%s) on %d cases",
            len(variants),
            self.ablation_type,
            len(cases),
        )

        # Ensure output directory exists
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        all_results: List[Dict[str, Any]] = []

        with open(self.output_path, "w", encoding="utf-8") as fh:
            for v_idx, variant in enumerate(variants, 1):
                logger.info(
                    "[%d/%d] Variant: %s", v_idx, len(variants), variant.variant_name
                )
                results = self._run_variant(variant, cases)
                for r in results:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                all_results.extend(results)

        logger.info(
            "Ablation complete. %d total results written to %s",
            len(all_results),
            self.output_path,
        )
        return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DermArbiter Ablation Runner",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file or directory."
    )
    parser.add_argument(
        "--data", required=True, help="Path to JSONL dataset file."
    )
    parser.add_argument(
        "--ablation-type",
        choices=sorted(VALID_ABLATION_TYPES),
        required=True,
        help="Type of ablation: agent, tool, or round.",
    )
    parser.add_argument(
        "--output",
        default="results/ablation.jsonl",
        help="Output JSONL path (default: results/ablation.jsonl).",
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
        help="Limit cases per variant.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = AblationRunner(
        config_path=args.config,
        data_path=args.data,
        ablation_type=args.ablation_type,
        output_path=args.output,
        mock=args.mock,
        max_cases=args.max_cases,
    )
    runner.run()


if __name__ == "__main__":
    main()
