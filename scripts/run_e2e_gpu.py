#!/usr/bin/env python3
"""DermArbiter — End-to-End GPU Test Script

Standalone script that initializes the full DermArbiter multi-agent
diagnostic pipeline, runs all 5 phases (Plan & Probe → Independent
Reading → Reveal & Critique → Targeted Debate → Synthesis), and
prints the final clinical report with consensus scores and timing.

Usage:
    # Full GPU run with real models
    python scripts/run_e2e_gpu.py \
        --config configs/default.yaml \
        --image data/sample.jpg \
        --query "Changing mole on back"

    # Mock mode (no GPU / API keys required)
    python scripts/run_e2e_gpu.py \
        --mock \
        --query "Changing mole on back"

    # With patient context
    python scripts/run_e2e_gpu.py \
        --config configs/default.yaml \
        --image data/sample.jpg \
        --query "Red scaly patch on elbow" \
        --age 45 --sex male --fitzpatrick 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `dermarbiter` is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Mock layer — lightweight stubs that mirror the real pipeline interfaces
# ═══════════════════════════════════════════════════════════════════════════


class _MockToolOutput:
    """Minimal stand-in for ``ToolOutput``."""

    def __init__(self, tool_name: str, result: dict, confidence: float,
                 raw_text: str, metadata: dict | None = None,
                 timestamp: str | None = None):
        self.tool_name = tool_name
        self.result = result
        self.confidence = confidence
        self.raw_text = raw_text
        self.metadata = metadata or {}
        self.timestamp = timestamp or ""


class _MockTool:
    """A tool that returns canned dermatology results."""

    _MOCK_RESULTS: dict[str, dict] = {
        "panderm_classifier": {
            "predictions": [
                {"label": "melanocytic nevus", "score": 0.62},
                {"label": "melanoma", "score": 0.18},
                {"label": "seborrheic keratosis", "score": 0.11},
            ],
            "summary": "PanDerm: melanocytic nevus (62%), melanoma (18%), seborrheic keratosis (11%)",
        },
        "make_annotator": {
            "annotations": {
                "asymmetry": "mild",
                "border": "regular",
                "color": "brown, homogeneous",
                "diameter": "5mm",
                "evolution": "stable",
            },
            "summary": "MAKE ABCDE: A-mild, B-regular, C-homogeneous brown, D-5mm, E-stable",
        },
        "dermogpt_vqa": {
            "answer": "The lesion appears to be a compound melanocytic nevus with regular pigment network.",
            "summary": "DermoGPT VQA: compound melanocytic nevus with regular pigment network",
        },
        "general_vqa": {
            "answer": "Well-circumscribed pigmented lesion with symmetrical borders, consistent with a benign nevus.",
            "summary": "General VQA: benign nevus with symmetrical borders",
        },
        "guideline_rag": {
            "guideline": "BAD 2024: Lesions with ABCDE score < 3 and stable history have low malignancy risk.",
            "summary": "GuidelineRAG: low malignancy risk per BAD 2024 criteria",
        },
        "case_rag": {
            "similar_cases": [
                {"case_id": "HAM-001234", "dx": "melanocytic nevus", "similarity": 0.89},
                {"case_id": "HAM-005678", "dx": "melanoma", "similarity": 0.42},
            ],
            "summary": "CaseRAG: closest match melanocytic nevus (89% sim), melanoma (42% sim)",
        },
        "ontology_graph": {
            "icd10": "D22.9",
            "snomed": "400010006",
            "hierarchy": ["Neoplasm", "Melanocytic neoplasm", "Melanocytic nevus"],
            "summary": "OntologyGraph: D22.9 (ICD-10), melanocytic nevus hierarchy",
        },
        "fairness_probe": {
            "fitzpatrick_bias": {"I-II": 0.02, "III-IV": 0.01, "V-VI": 0.05},
            "calibration_gap": 0.03,
            "summary": "FairnessProbe: max calibration gap 0.05 (FST V-VI), overall 0.03",
        },
        "uncertainty_probe": {
            "epistemic": 0.12,
            "aleatoric": 0.08,
            "total": 0.15,
            "summary": "UncertaintyProbe: total uncertainty 0.15 (epistemic 0.12, aleatoric 0.08)",
        },
    }

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock {self._name} tool for testing."

    def run(self, image_path: str | None = None, query: str = "") -> _MockToolOutput:
        data = self._MOCK_RESULTS.get(self._name, {"summary": f"{self._name}: no mock data"})
        return _MockToolOutput(
            tool_name=self._name,
            result=data,
            confidence=0.75,
            raw_text=data.get("summary", str(data)),
            metadata={"mode": "mock"},
        )

    def validate_input(self, image_path=None, query=""):
        return True

    def to_schema(self):
        return {"name": self._name, "description": self.description}


class _MockModelRouter:
    """Returns deterministic LLM responses keyed by agent role."""

    _BRIEF_TEMPLATES: dict[str, dict] = {
        "specialist": {
            "top3_differential": ["melanocytic nevus", "melanoma", "seborrheic keratosis"],
            "confidence": 0.82,
            "reasoning": "PanDerm classifier indicates melanocytic nevus at 62% confidence. "
                         "ABCDE analysis shows regular borders, homogeneous color, and stable evolution. "
                         "Guideline RAG confirms low malignancy risk per BAD 2024.",
            "cited_cards": [],
            "disagreement_flags": [],
        },
        "generalist": {
            "top3_differential": ["melanocytic nevus", "dermatofibroma", "melanoma"],
            "confidence": 0.74,
            "reasoning": "Visual inspection suggests a well-circumscribed pigmented lesion. "
                         "Case RAG shows 89% similarity to benign nevus. "
                         "Moderate confidence due to limited patient history.",
            "cited_cards": [],
            "disagreement_flags": [],
        },
        "skeptic": {
            "top3_differential": ["melanocytic nevus", "melanoma", "atypical nevus"],
            "confidence": 0.58,
            "reasoning": "While evidence favors benign nevus, the 18% melanoma probability from "
                         "PanDerm warrants caution. Recommend dermoscopic follow-up in 3 months. "
                         "Uncertainty probe shows non-trivial epistemic uncertainty (0.12).",
            "cited_cards": [],
            "disagreement_flags": ["insufficient_followup_plan"],
        },
        "moderator": {
            "top3_differential": ["melanocytic nevus", "melanoma", "seborrheic keratosis"],
            "confidence": 0.78,
            "reasoning": "Panel consensus favors melanocytic nevus. Skeptic raises valid concern "
                         "about melanoma risk requiring follow-up.",
            "cited_cards": [],
            "disagreement_flags": [],
        },
    }

    def call(self, agent_role: str, messages: list, **kwargs) -> str:
        brief = self._BRIEF_TEMPLATES.get(agent_role, self._BRIEF_TEMPLATES["generalist"])
        return json.dumps(brief)


class _MockAgent:
    """Lightweight mock agent compatible with the orchestrator contract."""

    def __init__(self, role: str, router: _MockModelRouter, has_tools: bool = True):
        self._role = role
        self._router = router
        self._has_tools = has_tools
        self._config = type("Cfg", (), {
            "role": role,
            "model_backend": "mock",
            "model_name": "mock-model",
            "has_tool_access": has_tools,
            "allowed_tools": [],
            "system_prompt_path": "",
            "temperature": 0.3,
            "max_output_tokens": 2048,
        })()

    @property
    def role(self) -> str:
        return self._role

    @property
    def has_tool_access(self) -> bool:
        return self._has_tools

    # Phase 1
    def propose_tools(self, case_info: dict) -> list[str]:
        tool_map = {
            "specialist": ["panderm_classifier", "make_annotator", "dermogpt_vqa",
                           "guideline_rag", "uncertainty_probe"],
            "generalist": ["panderm_classifier", "general_vqa", "case_rag",
                           "fairness_probe"],
            "moderator":  ["ontology_graph", "uncertainty_probe"],
            "skeptic":    [],
        }
        return tool_map.get(self._role, [])

    # Phase 2
    def generate_brief(self, evidence_cards: list) -> Any:
        from dermarbiter.core.blackboard import AgentBrief
        tmpl = _MockModelRouter._BRIEF_TEMPLATES.get(
            self._role, _MockModelRouter._BRIEF_TEMPLATES["generalist"]
        )
        return AgentBrief(agent_role=self._role, **tmpl)

    # Phase 4
    def generate_argument(self, topic: str, opponent_brief: Any) -> str:
        return (
            f"[{self._role.upper()}] Regarding '{topic[:80]}': "
            f"I maintain my position based on the evidence. "
            f"The opponent's confidence of {opponent_brief.confidence:.2f} "
            f"is noted but does not account for the full evidence base."
        )

    # Phase 3 (moderator-specific)
    def should_early_exit(self, briefs: dict) -> bool:
        if len(briefs) < 2:
            return False
        primaries = [
            b.top3_differential[0].lower()
            for b in briefs.values()
            if b.top3_differential
        ]
        if not primaries:
            return False
        from collections import Counter
        counts = Counter(primaries)
        top_count = counts.most_common(1)[0][1]
        return top_count >= 2 and all(
            b.confidence >= 0.50 for b in briefs.values()
        )

    def synthesize_final_report(self, state: Any) -> str:
        dx_list = state.final_diagnosis or ["Undetermined"]
        dx_str = "\n".join(f"  {i+1}. {dx.title()}" for i, dx in enumerate(dx_list))
        dissent_str = "\n".join(f"  - {n}" for n in state.dissent_notes) if state.dissent_notes else "  None"
        return (
            f"# DermArbiter Clinical Report\n"
            f"## Case: {state.case_id}\n\n"
            f"## Consensus Differential Diagnosis\n{dx_str}\n\n"
            f"## Consensus Score: {state.consensus_score:.2f}\n\n"
            f"## Clinical Summary\n"
            f"The multi-agent panel reached consensus on the primary diagnosis. "
            f"Evidence from {len(state.evidence_cards)} diagnostic tools was reviewed "
            f"by {len(state.briefs)} agents over {len(state.debate_log)} debate turns.\n\n"
            f"## Dissent Notes\n{dissent_str}\n\n"
            f"## Telemetry\n"
            f"  Total tokens: {state.total_tokens}\n"
            f"  Tool invocations: {state.total_tool_calls}\n"
            f"  Errors: {len(state.errors)}\n"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _build_mock_pipeline(
    query: str,
    image_path: str | None = None,
    patient_context: dict | None = None,
) -> tuple:
    """Build the pipeline using mock components (no GPU / API keys)."""
    from dermarbiter.core.blackboard import BlackboardState
    from dermarbiter.tools.base_tool import ToolRegistry, BaseTool, ToolOutput

    # Wrap _MockTool so it passes isinstance checks with BaseTool
    class MockBaseTool(BaseTool):
        def __init__(self, mock: _MockTool):
            self._mock = mock

        @property
        def name(self) -> str:
            return self._mock.name

        @property
        def description(self) -> str:
            return self._mock.description

        def run(self, image_path=None, query="") -> ToolOutput:
            m = self._mock.run(image_path, query)
            return ToolOutput(
                tool_name=m.tool_name,
                result=m.result,
                confidence=m.confidence,
                raw_text=m.raw_text,
                metadata=m.metadata,
            )

    # Build tool registry
    registry = ToolRegistry()
    all_tools = [
        "panderm_classifier", "make_annotator", "dermogpt_vqa",
        "general_vqa", "guideline_rag", "case_rag",
        "ontology_graph", "fairness_probe", "uncertainty_probe",
    ]
    for tool_name in all_tools:
        registry.register(MockBaseTool(_MockTool(tool_name)))

    # Build mock agents
    router = _MockModelRouter()
    agents = {
        "specialist": _MockAgent("specialist", router, has_tools=True),
        "generalist": _MockAgent("generalist", router, has_tools=True),
        "skeptic":    _MockAgent("skeptic",    router, has_tools=False),
        "moderator":  _MockAgent("moderator",  router, has_tools=True),
    }

    # Build initial state
    state = BlackboardState(
        case_id=f"CASE-{uuid.uuid4().hex[:12]}",
        image_path=image_path,
        query=query,
        patient_context=patient_context or {},
    )

    return agents, registry, state


def _build_real_pipeline(
    config_dir: str,
    query: str,
    image_path: str | None = None,
    patient_context: dict | None = None,
) -> tuple:
    """Build the pipeline using real models and tools."""
    from dermarbiter.core.config import load_config, AgentConfig
    from dermarbiter.core.model_router import ModelRouter
    from dermarbiter.core.blackboard import BlackboardState
    from dermarbiter.tools.base_tool import ToolRegistry
    from dermarbiter.agents import (
        SpecialistAgent,
        GeneralistAgent,
        SkepticAgent,
        ModeratorAgent,
    )
    from dermarbiter.tools import (
        PanDermClassifier,
        MAKEAnnotator,
        DermoGPTVQA,
        MedGemmaVQA,
        GuidelineRAG,
        CaseRAG,
        OntologyGraph,
        FairnessProbe,
        UncertaintyProbe,
    )

    # 1. Load config
    cfg = load_config(config_dir)
    print(f"  Config loaded from: {config_dir}")
    print(f"  Project: {cfg.project_name}  |  Default model: {cfg.default_model}")

    # 2. Create model router
    router = ModelRouter(cfg)
    print(f"  ModelRouter initialized.")

    # 3. Build agent configs from the loaded config
    agent_configs = {}
    for role_key, ac in cfg.agents.items():
        agent_configs[role_key] = ac

    # Fill missing agent configs with defaults
    for role in ["specialist", "generalist", "skeptic", "moderator"]:
        if role not in agent_configs:
            agent_configs[role] = AgentConfig(
                role=role,
                model_backend="google_api",
                model_name=cfg.default_model,
                temperature=cfg.default_temperature,
            )

    # 4. Create tool registry and register all 9 tools
    registry = ToolRegistry()
    tool_classes = [
        PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
        GuidelineRAG, CaseRAG, OntologyGraph, FairnessProbe, UncertaintyProbe,
    ]
    for ToolClass in tool_classes:
        try:
            tool_instance = ToolClass()
            registry.register(tool_instance)
            print(f"  ✓ Registered tool: {tool_instance.name}")
        except Exception as exc:
            print(f"  ✗ Failed to register {ToolClass.__name__}: {exc}")

    # 5. Create agents
    agents = {
        "specialist": SpecialistAgent(config=agent_configs["specialist"], model_router=router, tool_registry=registry),
        "generalist": GeneralistAgent(config=agent_configs["generalist"], model_router=router, tool_registry=registry),
        "skeptic":    SkepticAgent(config=agent_configs["skeptic"],    model_router=router),
        "moderator":  ModeratorAgent(config=agent_configs["moderator"],  model_router=router, tool_registry=registry),
    }
    print(f"  Agents created: {list(agents.keys())}")

    # 6. Build initial state
    state = BlackboardState(
        case_id=f"CASE-{uuid.uuid4().hex[:12]}",
        image_path=image_path,
        query=query,
        patient_context=patient_context or {},
    )

    return agents, registry, state


def run_pipeline(
    agents: dict,
    registry: Any,
    state: Any,
    mock_mode: bool = False,
) -> Any:
    """Execute the full 5-phase DermArbiter pipeline."""
    from dermarbiter.core.blackboard import BlackboardState

    if mock_mode:
        # In mock mode, drive the phases manually (no LangGraph dependency needed)
        from dermarbiter.core.debate_protocol import (
            plan_probe,
            independent_read,
            reveal_critique,
            targeted_debate,
            synthesis,
        )

        print("\n" + "═" * 60)
        print("  Phase 1: Plan & Probe")
        print("═" * 60)
        plan_probe(state, agents, registry)
        print(f"  Evidence cards collected: {len(state.evidence_cards)}")

        print("\n" + "═" * 60)
        print("  Phase 2: Independent Reading")
        print("═" * 60)
        independent_read(state, agents)
        print(f"  Briefs submitted: {list(state.briefs.keys())}")

        print("\n" + "═" * 60)
        print("  Phase 3: Reveal & Critique")
        print("═" * 60)
        reveal_critique(state, agents)
        print(f"  Early exit: {state.early_exit}")

        print("\n" + "═" * 60)
        print("  Phase 4: Targeted Debate")
        print("═" * 60)
        targeted_debate(state, agents, max_rounds=3)
        print(f"  Debate turns: {len(state.debate_log)}")

        print("\n" + "═" * 60)
        print("  Phase 5: Synthesis")
        print("═" * 60)
        synthesis(state, agents)
        print(f"  Final diagnoses: {state.final_diagnosis}")

        return state
    else:
        # Real mode — use the LangGraph orchestrator
        from dermarbiter.core.orchestrator import DermArbiterOrchestrator

        orchestrator = DermArbiterOrchestrator(
            agents=agents,
            tool_registry=registry,
            max_rounds=3,
            max_tokens_per_turn=100,
            global_token_budget=50_000,
        )
        return orchestrator.run(state)


def print_results(state: Any, elapsed: float) -> None:
    """Pretty-print the pipeline results."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  DERMARBITER — FINAL RESULTS".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print(f"\n  Case ID:     {state.case_id}")
    print(f"  Query:       {state.query}")
    print(f"  Image:       {state.image_path or 'N/A'}")

    print("\n" + "─" * 70)
    print("  CONSENSUS DIFFERENTIAL DIAGNOSIS")
    print("─" * 70)
    for i, dx in enumerate(state.final_diagnosis, 1):
        print(f"    {i}. {dx.title()}")
    if not state.final_diagnosis:
        print("    (no diagnoses produced)")

    print(f"\n  Consensus Score:   {state.consensus_score:.2f}")

    if state.dissent_notes:
        print("\n  Dissent Notes:")
        for note in state.dissent_notes:
            print(f"    • {note}")

    print("\n" + "─" * 70)
    print("  CLINICAL REPORT")
    print("─" * 70)
    print(state.clinical_report or "  (no report generated)")

    print("\n" + "─" * 70)
    print("  TELEMETRY")
    print("─" * 70)
    print(f"    Total tokens:      {state.total_tokens}")
    print(f"    Tool invocations:  {state.total_tool_calls}")
    print(f"    Evidence cards:    {len(state.evidence_cards)}")
    print(f"    Debate turns:      {len(state.debate_log)}")
    print(f"    Errors:            {len(state.errors)}")
    print(f"    Wall-clock time:   {elapsed:.2f}s")
    print("─" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DermArbiter — End-to-End GPU/Mock Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_e2e_gpu.py --mock --query 'Changing mole on back'\n"
            "  python scripts/run_e2e_gpu.py --config configs/ --image data/sample.jpg "
            "--query 'Red scaly patch'\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/",
        help="Path to config directory containing YAML files (default: configs/)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the clinical image file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Clinical query describing the patient's concern.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run in mock mode without GPU or API keys.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )

    # Patient context flags
    parser.add_argument("--age", type=int, default=None, help="Patient age.")
    parser.add_argument("--sex", type=str, default=None, help="Patient sex.")
    parser.add_argument("--fitzpatrick", type=int, default=None, help="Fitzpatrick skin type (1-6).")
    parser.add_argument("--location", type=str, default=None, help="Lesion body location.")
    parser.add_argument("--duration", type=str, default=None, help="Duration of the condition.")

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    _setup_logging(args.log_level)

    # Build patient context from CLI flags
    patient_context: dict[str, Any] = {}
    if args.age is not None:
        patient_context["age"] = args.age
    if args.sex is not None:
        patient_context["sex"] = args.sex
    if args.fitzpatrick is not None:
        patient_context["fitzpatrick_type"] = args.fitzpatrick
    if args.location is not None:
        patient_context["location"] = args.location
    if args.duration is not None:
        patient_context["duration"] = args.duration

    print("\n╔" + "═" * 68 + "╗")
    print("║" + "  DERMARBITER — END-TO-END PIPELINE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Mode:    {'MOCK (no GPU / API keys)' if args.mock else 'REAL (GPU / API)'}")
    print(f"  Query:   {args.query}")
    print(f"  Image:   {args.image or 'N/A'}")
    print(f"  Config:  {args.config}")
    if patient_context:
        print(f"  Patient: {patient_context}")

    try:
        # Build pipeline
        print("\n  Initializing pipeline...")
        if args.mock:
            agents, registry, state = _build_mock_pipeline(
                query=args.query,
                image_path=args.image,
                patient_context=patient_context,
            )
        else:
            # Resolve config directory relative to project root if not absolute
            config_dir = args.config
            if not os.path.isabs(config_dir):
                config_dir = str(_PROJECT_ROOT / config_dir)
            agents, registry, state = _build_real_pipeline(
                config_dir=config_dir,
                query=args.query,
                image_path=args.image,
                patient_context=patient_context,
            )

        # Run pipeline
        t0 = time.time()
        result_state = run_pipeline(agents, registry, state, mock_mode=args.mock)
        elapsed = time.time() - t0

        # Print results
        print_results(result_state, elapsed)

        # Optionally save JSON output
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data = result_state.to_dict()
            output_data["_meta"] = {
                "elapsed_seconds": round(elapsed, 3),
                "mock_mode": args.mock,
                "config_dir": args.config,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\n  Results saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logging.getLogger(__name__).error("Pipeline failed: %s", exc, exc_info=True)
        print(f"\n  ✗ Pipeline failed: {exc}")
        print("\n  Troubleshooting tips:")
        print("    1. Set GOOGLE_API_KEY in .env or environment")
        print("    2. Use --mock for testing without GPU/API")
        print("    3. Run 'python scripts/validate_tools.py' to check dependencies")
        print("    4. Use --log-level DEBUG for detailed error info")
        sys.exit(1)


if __name__ == "__main__":
    main()
