#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
DermArbiter Agent Layer Demo
═══════════════════════════════════════════════════════════════
A focused demonstration of the DermArbiter agent layer:
  • Creating mock agents (Specialist, Generalist, Skeptic, Moderator)
  • Exercising each agent's Phase 1, 2, and 4 interfaces
  • Inspecting the AgentBrief data model

Runs entirely on CPU — no GPU or API keys required.

Usage:
    python notebooks/02_agent_demo.py
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _banner(title: str, char: str = "═", width: int = 64) -> None:
    """Print a section banner."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def _sub_header(title: str, char: str = "─", width: int = 56) -> None:
    """Print a sub-section header."""
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Create Mock Agents
# ═══════════════════════════════════════════════════════════════════════════

def demo_agent_creation():
    """
    Instantiate all four DermArbiter mock agents and display
    their role identifiers and tool-access permissions.
    """
    _banner("SECTION 1: Agent Creation")

    from tests.mocks.mock_agents import (
        MockSpecialist,
        MockGeneralist,
        MockSkeptic,
        MockModerator,
        create_mock_agents,
    )

    agents = create_mock_agents()

    print(f"  {'Role':<14s}  {'Class':<22s}  {'Tool Access':<12s}")
    print(f"  {'─' * 14}  {'─' * 22}  {'─' * 12}")
    for role, agent in agents.items():
        cls_name = type(agent).__name__
        access = "✓ Yes" if agent.has_tool_access else "✗ No"
        print(f"  {role:<14s}  {cls_name:<22s}  {access:<12s}")

    print(f"\n  Total agents: {len(agents)}")
    return agents


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Phase 1 — Tool Proposals
# ═══════════════════════════════════════════════════════════════════════════

def demo_propose_tools(agents):
    """
    Each agent proposes diagnostic tools for a sample case.
    The Skeptic has no tool access and returns an empty list.
    """
    _banner("SECTION 2: Phase 1 — Tool Proposals")

    case_info = {
        "case_id": "DEMO-001",
        "query": "55yo male, changing pigmented lesion upper back, 6 months",
        "patient_context": {
            "age": 55,
            "sex": "Male",
            "fitzpatrick_type": "III",
            "location": "upper back",
        },
        "image_path": None,
    }

    print(f"  Case: {case_info['query']}\n")

    for role, agent in agents.items():
        tools = agent.propose_tools(case_info)
        count = len(tools)
        tools_str = ", ".join(tools) if tools else "(no tool access)"
        print(f"  [{role.upper():>12s}]  {count} tool(s): {tools_str}")

    return case_info


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Phase 2 — Independent Diagnostic Briefs
# ═══════════════════════════════════════════════════════════════════════════

def demo_generate_briefs(agents):
    """
    Each agent generates an independent diagnostic brief based on
    mock evidence cards. We display the structured AgentBrief fields.
    """
    _banner("SECTION 3: Phase 2 — Diagnostic Briefs")

    # Create some mock evidence cards for context
    from dermarbiter.core.blackboard import EvidenceCard, ToolOutput
    from tests.mocks.mock_tools import create_mock_registry

    registry = create_mock_registry()
    mock_evidence = []
    for tool_name in ["panderm_classifier", "make_annotator", "case_rag"]:
        tool = registry.get(tool_name)
        output = tool.run(query="melanoma workup")
        card = EvidenceCard(
            tool_output=ToolOutput(
                tool_name=output.tool_name,
                result=output.result,
                confidence=output.confidence,
                raw_text=output.raw_text,
                metadata=output.metadata,
            ),
            requested_by="specialist",
        )
        mock_evidence.append(card)

    print(f"  Evidence cards provided: {len(mock_evidence)}")
    print(f"  Tools: {', '.join(c.tool_output.tool_name for c in mock_evidence)}\n")

    briefs = {}
    for role, agent in agents.items():
        brief = agent.generate_brief(mock_evidence)
        briefs[role] = brief

        _sub_header(f"Agent: {role.upper()}")
        print(f"    Top-3 Differential : {brief.top3_differential}")
        print(f"    Confidence         : {brief.confidence:.2f}")
        print(f"    Cited Cards        : {brief.cited_cards}")
        print(f"    Disagreement Flags : {brief.disagreement_flags or '(none)'}")
        print(f"    Reasoning:")

        # Word-wrap reasoning for readability
        words = brief.reasoning.split()
        line = "      "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                print(line)
                line = "      " + word
            else:
                line += " " + word if line.strip() else "      " + word
        if line.strip():
            print(line)

    return briefs


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Phase 4 — Debate Arguments
# ═══════════════════════════════════════════════════════════════════════════

def demo_generate_arguments(agents, briefs):
    """
    Each debating agent (Specialist, Generalist, Skeptic) generates
    an argument in response to an opponent's brief. We show how the
    generate_argument method constructs evidence-based rebuttals.
    """
    _banner("SECTION 4: Phase 4 — Debate Arguments")

    debate_pairs = [
        ("specialist", "generalist", "Primary diagnosis divergence: melanoma vs BCC"),
        ("generalist", "specialist", "Confidence calibration: 0.68 vs 0.85"),
        ("skeptic", "specialist", "Questioning overconfidence in melanoma diagnosis"),
    ]

    for speaker, opponent, topic in debate_pairs:
        agent = agents[speaker]
        opponent_brief = briefs[opponent]

        _sub_header(f"{speaker.upper()} → {opponent.upper()}")
        print(f"    Topic: {topic}")
        print(f"    Opponent confidence: {opponent_brief.confidence:.2f}")
        print(f"    Opponent top dx: {opponent_brief.top3_differential[0]}")
        print()

        argument = agent.generate_argument(topic, opponent_brief)
        print(f"    Argument:")

        # Word-wrap for readability
        words = argument.split()
        line = "      "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                print(line)
                line = "      " + word
            else:
                line += " " + word if line.strip() else "      " + word
        if line.strip():
            print(line)


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: AgentBrief Data Model Inspection
# ═══════════════════════════════════════════════════════════════════════════

def demo_brief_model():
    """
    Show the AgentBrief Pydantic model structure, including
    JSON serialisation and schema.
    """
    _banner("SECTION 5: AgentBrief Data Model")

    from dermarbiter.core.blackboard import AgentBrief
    import json

    sample = AgentBrief(
        agent_role="specialist",
        top3_differential=["melanoma", "dysplastic_nevus", "BCC"],
        confidence=0.85,
        reasoning="Evidence-based clinical reasoning.",
        cited_cards=["EC-001", "EC-002"],
        disagreement_flags=["possible_overconfidence"],
    )

    print("  JSON representation:")
    serialised = sample.model_dump(mode="json")
    formatted = json.dumps(serialised, indent=4)
    for line in formatted.split("\n"):
        print(f"    {line}")

    print("\n  Model fields:")
    for name, field_info in AgentBrief.model_fields.items():
        ftype = str(field_info.annotation).replace("typing.", "")
        print(f"    {name:<22s}  {ftype}")


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Mock Tool Registry Overview
# ═══════════════════════════════════════════════════════════════════════════

def demo_tool_registry():
    """
    Show all 9 mock tools registered in the DermArbiter tool registry.
    """
    _banner("SECTION 6: Mock Tool Registry")

    from tests.mocks.mock_tools import create_mock_registry

    registry = create_mock_registry()

    print(f"  Registered tools: {len(registry)}\n")
    print(f"  {'#':<4s}  {'Tool Name':<22s}  {'Description (truncated)':<40s}")
    print(f"  {'─' * 4}  {'─' * 22}  {'─' * 40}")
    for i, tool_name in enumerate(registry.tool_names, 1):
        tool = registry.get(tool_name)
        desc = tool.description[:40] + "…" if len(tool.description) > 40 else tool.description
        print(f"  {i:<4d}  {tool_name:<22s}  {desc:<40s}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔" + "═" * 64 + "╗")
    print("║" + "  DERMARBITER — AGENT LAYER DEMO".center(64) + "║")
    print("║" + "  Mock Mode (no GPU / API keys required)".center(64) + "║")
    print("╚" + "═" * 64 + "╝")

    t0 = time.time()

    agents = demo_agent_creation()
    demo_propose_tools(agents)
    briefs = demo_generate_briefs(agents)
    demo_generate_arguments(agents, briefs)
    demo_brief_model()
    demo_tool_registry()

    elapsed = time.time() - t0

    _banner("DEMO COMPLETE")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  All sections executed successfully ✓")
    print()


if __name__ == "__main__":
    main()
