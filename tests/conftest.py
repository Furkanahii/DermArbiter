"""Pytest fixtures for DermArbiter test suite.

Provides reusable fixtures for:
  • sample clinical cases
  • pre-populated mock tool registries
  • realistic EvidenceCard and AgentBrief collections
  • empty and populated BlackboardState instances
"""

from __future__ import annotations

import pytest

from dermarbiter.core.blackboard import (
    AgentBrief,
    BlackboardState,
    DebateTurn,
    EvidenceCard,
    ToolOutput as BBToolOutput,
)
from dermarbiter.tools.base_tool import ToolOutput, ToolRegistry
from tests.mocks.mock_agents import (
    MockGeneralist,
    MockModerator,
    MockSkeptic,
    MockSpecialist,
    create_mock_agents,
)
from tests.mocks.mock_tools import create_mock_registry


# ═══════════════════════════════════════════════════════════════════════════
# Sample case
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_case() -> dict:
    """A realistic melanoma case dict used throughout the test suite."""
    return {
        "case_id": "TEST-CASE-001",
        "query": (
            "55-year-old male presents with a changing pigmented lesion "
            "on the upper back. The lesion has grown over the past 6 months "
            "and now shows irregular borders with colour variegation. "
            "Patient reports occasional itching. No personal history of "
            "skin cancer but positive family history (father had melanoma "
            "at age 60)."
        ),
        "patient_context": {
            "age": 55,
            "sex": "Male",
            "fitzpatrick_type": "III",
            "location": "upper back",
            "duration": "6 months",
            "family_history": "father — melanoma (age 60)",
            "personal_history": "no prior skin cancer",
            "symptoms": "occasional pruritus, size change",
        },
        "image_path": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Mock tool registry
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_tool_registry() -> ToolRegistry:
    """A ``ToolRegistry`` pre-populated with all 9 mock tools."""
    return create_mock_registry()


# ═══════════════════════════════════════════════════════════════════════════
# Sample evidence cards
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_evidence_cards() -> list[EvidenceCard]:
    """Four realistic EvidenceCards from different tool categories."""
    return [
        EvidenceCard(
            card_id="EC-001",
            tool_output=BBToolOutput(
                tool_name="panderm_classifier",
                result={
                    "predictions": [
                        {"disease": "melanoma", "probability": 0.62},
                        {"disease": "basal_cell_carcinoma", "probability": 0.15},
                        {"disease": "melanocytic_nevus", "probability": 0.11},
                    ],
                },
                confidence=0.62,
                raw_text="Top prediction: melanoma (62%), BCC (15%), nevus (11%).",
                metadata={"model": "PanDerm", "latency_ms": 142},
            ),
            requested_by="specialist",
        ),
        EvidenceCard(
            card_id="EC-002",
            tool_output=BBToolOutput(
                tool_name="make_annotator",
                result={
                    "concepts": [
                        {"concept": "atypical_pigment_network", "score": 0.87},
                        {"concept": "blue_white_veil", "score": 0.74},
                        {"concept": "irregular_dots_globules", "score": 0.69},
                    ],
                },
                confidence=0.87,
                raw_text=(
                    "Atypical pigment network (0.87), blue-white veil (0.74), "
                    "irregular dots/globules (0.69)."
                ),
                metadata={"model": "MAKE"},
            ),
            requested_by="specialist",
        ),
        EvidenceCard(
            card_id="EC-003",
            tool_output=BBToolOutput(
                tool_name="case_rag",
                result={
                    "similar_cases": [
                        {"case_id": "derm1m_042187", "diagnosis": "melanoma", "distance": 0.12},
                        {"case_id": "derm1m_118934", "diagnosis": "melanoma", "distance": 0.18},
                        {"case_id": "derm1m_073621", "diagnosis": "dysplastic_nevus", "distance": 0.24},
                    ],
                },
                confidence=0.82,
                raw_text="Top 3 similar cases: 2 melanoma (d=0.12, 0.18), 1 dysplastic nevus (d=0.24).",
                metadata={"database": "Derm1M"},
            ),
            requested_by="generalist",
        ),
        EvidenceCard(
            card_id="EC-004",
            tool_output=BBToolOutput(
                tool_name="guideline_rag",
                result={
                    "chunks": [
                        {
                            "source": "DermNet NZ — Melanoma",
                            "text": "Melanoma should be suspected in any changing mole...",
                            "relevance_score": 0.94,
                        },
                    ],
                },
                confidence=0.91,
                raw_text="Guidelines: suspect melanoma in changing mole with ABCDE features.",
                metadata={"sources": ["DermNet NZ"]},
            ),
            requested_by="specialist",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Sample agent briefs
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_briefs() -> dict[str, AgentBrief]:
    """Dictionary of 4 realistic AgentBriefs — one per role."""
    agents = create_mock_agents()
    return {
        role: agent.generate_brief()
        for role, agent in agents.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# Blackboard fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def empty_blackboard(sample_case: dict) -> BlackboardState:
    """A fresh BlackboardState initialised with the sample case."""
    return BlackboardState(
        case_id=sample_case["case_id"],
        query=sample_case["query"],
        patient_context=sample_case["patient_context"],
        image_path=sample_case["image_path"],
    )


@pytest.fixture
def populated_blackboard(
    empty_blackboard: BlackboardState,
    sample_evidence_cards: list[EvidenceCard],
    sample_briefs: dict[str, AgentBrief],
) -> BlackboardState:
    """A BlackboardState with evidence cards, briefs, and debate turns."""
    bb = empty_blackboard

    # Phase 1 — add evidence cards
    for card in sample_evidence_cards:
        bb.add_evidence_card(card)

    # Phase 2 — add briefs
    for brief in sample_briefs.values():
        bb.add_brief(brief)

    # Phase 3-4 — add debate turns
    bb.add_debate_turn(
        DebateTurn(
            round_num=1,
            speaker="skeptic",
            argument=(
                "A 62% classifier probability does not constitute "
                "certainty. Regression structures (0.52) are equivocal."
            ),
            token_count=22,
        )
    )
    bb.add_debate_turn(
        DebateTurn(
            round_num=1,
            speaker="specialist",
            argument=(
                "Multi-modal convergence (classifier + dermoscopy + RAG) "
                "raises effective confidence beyond the 62% point estimate."
            ),
            token_count=20,
        )
    )

    return bb
