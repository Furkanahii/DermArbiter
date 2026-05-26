"""Mock agent implementations for DermArbiter testing.

Provides deterministic mock versions of the four debate agents
(Specialist, Generalist, Skeptic, Moderator) that return realistic
``AgentBrief`` objects for a melanoma case scenario.

These mocks allow the debate protocol and orchestrator to be tested
independently of any LLM backend.

Usage::

    from tests.mocks.mock_agents import MockSpecialist, MockGeneralist
    agent = MockSpecialist()
    brief = agent.generate_brief(evidence_cards=[])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dermarbiter.core.blackboard import AgentBrief, EvidenceCard


# ---------------------------------------------------------------------------
# Lightweight mock base (mirrors the real BaseAgent interface)
# ---------------------------------------------------------------------------

class MockBaseAgent(ABC):
    """Minimal mock base agent mirroring the real BaseAgent interface.

    The real ``BaseAgent`` lives in ``dermarbiter.agents.base_agent``
    and depends on LLM backends.  This mock skips that dependency.
    """

    @property
    @abstractmethod
    def role(self) -> str:
        """Agent role identifier."""
        ...

    @property
    @abstractmethod
    def has_tool_access(self) -> bool:
        """Whether this agent may request tool calls."""
        ...

    @abstractmethod
    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        """Phase 1 — suggest which tools to run."""
        ...

    @abstractmethod
    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard] | None = None,
    ) -> AgentBrief:
        """Phase 2 — produce an independent diagnostic brief."""
        ...

    @abstractmethod
    def generate_argument(
        self,
        topic: str = "",
        opponent_brief: AgentBrief | None = None,
    ) -> str:
        """Phase 4 — produce a debate argument or rebuttal."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} role={self.role!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# Mock Specialist — Gemini 2.5 Flash
# ═══════════════════════════════════════════════════════════════════════════

class MockSpecialist(MockBaseAgent):
    """Expert dermatologist persona — high-confidence melanoma diagnosis."""

    @property
    def role(self) -> str:
        return "specialist"

    @property
    def has_tool_access(self) -> bool:
        return True

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        return [
            "panderm_classifier",
            "make_annotator",
            "dermogpt_vqa",
            "guideline_rag",
            "uncertainty_probe",
        ]

    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard] | None = None,
    ) -> AgentBrief:
        cited = [c.card_id for c in (evidence_cards or [])[:3]]
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[
                "melanoma",
                "dysplastic_nevus",
                "basal_cell_carcinoma",
            ],
            confidence=0.85,
            cited_cards=cited or ["EC-001", "EC-002", "EC-006"],
            reasoning=(
                "PanDerm classifier yields 62% melanoma probability. "
                "Dermoscopic features (atypical pigment network 0.87, "
                "blue-white veil 0.74) align with revised 7-point "
                "checklist score ≥ 5. Case RAG retrieves 3/5 confirmed "
                "melanoma cases with low embedding distance. Guidelines "
                "recommend excisional biopsy. I assess melanoma as the "
                "leading diagnosis with high confidence."
            ),
            disagreement_flags=[],
        )

    def generate_argument(
        self,
        topic: str = "",
        opponent_brief: AgentBrief | None = None,
    ) -> str:
        return (
            "The convergence of PanDerm classification (62% melanoma), "
            "dermoscopic ABCDE criteria, and case-based retrieval (3/5 "
            "melanoma neighbours) strongly supports melanoma as the "
            "primary diagnosis. Excisional biopsy is indicated."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mock Generalist — MedGemma-4B
# ═══════════════════════════════════════════════════════════════════════════

class MockGeneralist(MockBaseAgent):
    """Non-specialist medical persona — moderate confidence, broader DDx."""

    @property
    def role(self) -> str:
        return "generalist"

    @property
    def has_tool_access(self) -> bool:
        return True

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        return [
            "panderm_classifier",
            "general_vqa",
            "case_rag",
            "fairness_probe",
        ]

    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard] | None = None,
    ) -> AgentBrief:
        cited = [c.card_id for c in (evidence_cards or [])[:2]]
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[
                "melanoma",
                "basal_cell_carcinoma",
                "dysplastic_nevus",
            ],
            confidence=0.68,
            cited_cards=cited or ["EC-001", "EC-004"],
            reasoning=(
                "Visual assessment reveals irregular borders and colour "
                "variegation suggesting melanoma as the top differential. "
                "However, pigmented BCC cannot be fully excluded given "
                "the blue-grey areas. General VQA supports a broad "
                "melanocytic neoplasm differential. Fairness probe shows "
                "Fitzpatrick III — no immediate bias concern for this "
                "skin tone, but per-subgroup monitoring is advised."
            ),
            disagreement_flags=[],
        )

    def generate_argument(
        self,
        topic: str = "",
        opponent_brief: AgentBrief | None = None,
    ) -> str:
        return (
            "While melanoma is the leading candidate, the possibility "
            "of pigmented BCC should not be dismissed. Blue-grey areas "
            "can be seen in both entities. A broader differential "
            "preserves diagnostic safety."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mock Skeptic — Qwen3-8B (no tool access)
# ═══════════════════════════════════════════════════════════════════════════

class MockSkeptic(MockBaseAgent):
    """Devil's advocate — challenges assumptions, no tool access."""

    @property
    def role(self) -> str:
        return "skeptic"

    @property
    def has_tool_access(self) -> bool:
        return False

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        return []  # Skeptic has no tool access by design

    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard] | None = None,
    ) -> AgentBrief:
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[
                "melanoma",
                "dysplastic_nevus",
                "seborrheic_keratosis",
            ],
            confidence=0.55,
            cited_cards=["EC-001"],
            reasoning=(
                "I note that PanDerm assigns only 62% to melanoma — "
                "this is not overwhelmingly high. The 11% melanocytic "
                "nevus probability deserves attention. Dysplastic nevi "
                "can mimic melanoma dermoscopically, and regression "
                "structures (score 0.52) are also seen in involuting "
                "nevi. I flag potential overconfidence in the melanoma "
                "diagnosis. Additionally, seborrheic keratosis with "
                "unusual features should remain in the differential."
            ),
            disagreement_flags=[
                "panderm_confidence_below_70pct",
                "regression_structures_ambiguous",
                "possible_overconfidence_in_melanoma",
            ],
        )

    def generate_argument(
        self,
        topic: str = "",
        opponent_brief: AgentBrief | None = None,
    ) -> str:
        return (
            "A 62% classifier probability does not constitute "
            "certainty. Regression structures (0.52) are equivocal "
            "and seen in both melanoma and involuting nevi. The panel "
            "should ensure biopsy recommendation rather than "
            "over-committing to a single diagnosis."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mock Moderator — Gemini 2.5 Flash
# ═══════════════════════════════════════════════════════════════════════════

class MockModerator(MockBaseAgent):
    """Panel chairperson — synthesises briefs, manages gating and turns."""

    @property
    def role(self) -> str:
        return "moderator"

    @property
    def has_tool_access(self) -> bool:
        return True

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        return [
            "ontology_graph",
            "uncertainty_probe",
        ]

    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard] | None = None,
    ) -> AgentBrief:
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[
                "melanoma",
                "dysplastic_nevus",
                "basal_cell_carcinoma",
            ],
            confidence=0.79,
            cited_cards=["EC-001", "EC-002", "EC-005", "EC-006"],
            reasoning=(
                "Synthesising evidence: PanDerm (62% melanoma), MAKE "
                "(atypical network 0.87 + blue-white veil 0.74), case "
                "RAG (3/5 melanoma neighbours), and guideline alignment "
                "(7-point checklist score ≥ 5) converge on melanoma as "
                "the leading diagnosis. The Skeptic raises valid points "
                "about sub-70% classifier confidence, but the "
                "multi-modal evidence convergence increases overall "
                "panel confidence. Uncertainty probe confirms conformal "
                "set = {melanoma, BCC} at 90% coverage. Recommend "
                "excisional biopsy with a note on the BCC alternative."
            ),
            disagreement_flags=[],
        )

    def generate_argument(
        self,
        topic: str = "",
        opponent_brief: AgentBrief | None = None,
    ) -> str:
        return (
            "The panel has reached majority consensus on melanoma as "
            "the primary diagnosis. While the Skeptic's caution is "
            "noted, multi-modal evidence convergence (classifier + "
            "dermoscopy + RAG + guidelines) supports this conclusion. "
            "Final recommendation: excisional biopsy."
        )

    def should_early_exit(self, briefs: dict[str, AgentBrief]) -> bool:
        """Determine whether mock early exit is warranted."""
        if len(briefs) < 2:
            return False
        primaries = []
        for role, brief in briefs.items():
            if brief.disagreement_flags:
                return False
            if brief.top3_differential:
                primaries.append(brief.top3_differential[0].strip().lower())
        if not primaries:
            return False
        return len(set(primaries)) == 1

    def synthesize_final_report(self, blackboard: Any) -> str:
        """Produce a comprehensive mock clinical report from blackboard state."""
        return (
            f"# DermArbiter Clinical Report — {blackboard.case_id}\n\n"
            f"## Summary\n"
            f"Multi-expert consensus analysis for dermatological case {blackboard.case_id}.\n\n"
            f"## Differential Diagnosis\n"
            f"1. Melanoma (Confidence: 0.85)\n"
            f"2. Dysplastic Nevus (Confidence: 0.68)\n"
            f"3. Basal Cell Carcinoma (Confidence: 0.55)\n\n"
            f"## Recommended Next Steps\n"
            f"Excisional biopsy of the lesion is recommended based on high confidence multi-modal evidence convergence."
        )



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_mock_agents() -> dict[str, MockBaseAgent]:
    """Create a mapping of role → mock agent for all four debate roles.

    Returns:
        Dictionary mapping role strings to mock agent instances.
    """
    agents: list[MockBaseAgent] = [
        MockSpecialist(),
        MockGeneralist(),
        MockSkeptic(),
        MockModerator(),
    ]
    return {agent.role: agent for agent in agents}
