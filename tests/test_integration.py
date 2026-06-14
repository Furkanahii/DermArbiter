"""End-to-end integration tests for DermArbiter pipeline.

Tests the full 5-phase debate protocol with mock tools and mock agents
working together through the LangGraph orchestrator.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import (
    AgentBrief,
    BlackboardState,
    DebateTurn,
    EvidenceCard,
    ToolOutput as BBToolOutput,
)
from dermarbiter.core.debate_protocol import (
    independent_read,
    plan_probe,
    reveal_critique,
    synthesis,
    targeted_debate,
)
from dermarbiter.core.orchestrator import DermArbiterOrchestrator
from dermarbiter.tools.base_tool import ToolOutput, ToolRegistry
from tests.mocks.mock_agents import (
    MockGeneralist,
    MockModerator,
    MockSkeptic,
    MockSpecialist,
    create_mock_agents,
)
from tests.mocks.mock_tools import create_mock_registry


# ---------------------------------------------------------------------------
# Shared fixtures local to this module
# ---------------------------------------------------------------------------

@pytest.fixture
def agents():
    """Create a dict of mock agents with moderator methods injected."""
    agent_map = create_mock_agents()
    moderator = agent_map["moderator"]

    def should_early_exit(briefs: dict[str, AgentBrief]) -> bool:
        """Early exit when all non-moderator agents agree on primary dx
        AND all have confidence >= 0.60 AND no disagreement flags."""
        non_mod = {r: b for r, b in briefs.items() if r != "moderator"}
        if len(non_mod) < 2:
            return False
        primaries = [
            b.top3_differential[0].strip().lower()
            for b in non_mod.values()
            if b.top3_differential
        ]
        if len(set(primaries)) != 1:
            return False
        if any(b.confidence < 0.60 for b in non_mod.values()):
            return False
        if any(b.disagreement_flags for b in non_mod.values()):
            return False
        return True

    def synthesize_final_report(state: BlackboardState) -> str:
        dx_line = ", ".join(state.final_diagnosis) or "Undetermined"
        return (
            f"# DermArbiter Clinical Report — {state.case_id}\n\n"
            f"## Differential Diagnosis\n{dx_line}\n\n"
            f"## Consensus Score\n{state.consensus_score:.2f}\n\n"
            f"## Dissent Notes\n"
            + ("\n".join(state.dissent_notes) or "None")
            + "\n"
        )

    moderator.should_early_exit = should_early_exit
    moderator.synthesize_final_report = synthesize_final_report
    return agent_map


@pytest.fixture
def registry():
    """Create a mock tool registry with all 9 tools."""
    return create_mock_registry()


@pytest.fixture
def bb(sample_case):
    """A fresh BlackboardState for integration testing."""
    return BlackboardState(
        case_id=sample_case["case_id"],
        query=sample_case["query"],
        patient_context=sample_case["patient_context"],
        image_path=sample_case["image_path"],
    )


@pytest.fixture
def disagreeing_agents(agents):
    """Agents where the generalist disagrees on primary diagnosis."""
    agents_copy = agents
    original_gen_brief = agents_copy["generalist"].generate_brief

    def disagree_brief(*args, **kwargs):
        brief = original_gen_brief(*args, **kwargs)
        brief.top3_differential = [
            "basal_cell_carcinoma",
            "melanoma",
            "dysplastic_nevus",
        ]
        return brief

    agents_copy["generalist"].generate_brief = disagree_brief
    return agents_copy


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Plan & Probe
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase1Integration:
    """Phase 1 — Plan & Probe with mock tools."""

    def test_agents_propose_tools(self, agents, sample_case):
        """Each agent with tool access proposes at least one tool."""
        case_info = {
            "case_id": sample_case["case_id"],
            "query": sample_case["query"],
            "patient_context": sample_case["patient_context"],
            "image_path": sample_case["image_path"],
        }
        for role, agent in agents.items():
            proposed = agent.propose_tools(case_info)
            if agent.has_tool_access:
                assert len(proposed) > 0, f"{role} should propose tools"
            else:
                assert len(proposed) == 0, f"{role} (no access) should propose nothing"

    def test_tool_batch_execution(self, agents, registry, bb):
        """Plan & probe executes tools and populates evidence cards."""
        plan_probe(bb, agents, registry)
        assert len(bb.evidence_cards) > 0
        assert bb.total_tool_calls == len(bb.evidence_cards)

    def test_evidence_cards_populated(self, agents, registry, bb):
        """Evidence cards contain valid tool outputs with correct fields."""
        plan_probe(bb, agents, registry)
        for card in bb.evidence_cards:
            assert card.card_id.startswith("EC-")
            assert card.tool_output.tool_name != ""
            assert 0.0 <= card.tool_output.confidence <= 1.0
            assert card.requested_by in ("specialist", "generalist", "moderator")

    def test_tool_deduplication(self, agents, registry, bb):
        """Same tool proposed by multiple agents is only executed once."""
        # Both specialist and generalist propose 'panderm_classifier'
        specialist_tools = agents["specialist"].propose_tools({"query": bb.query})
        generalist_tools = agents["generalist"].propose_tools({"query": bb.query})
        shared = set(specialist_tools) & set(generalist_tools)
        assert "panderm_classifier" in shared, "Expected overlap on panderm_classifier"

        plan_probe(bb, agents, registry)
        panderm_cards = [
            c for c in bb.evidence_cards
            if c.tool_output.tool_name == "panderm_classifier"
        ]
        assert len(panderm_cards) == 1, "panderm_classifier should appear exactly once"
        # First requester (specialist) gets credit
        assert panderm_cards[0].requested_by == "specialist"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Independent Reading
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase2Integration:
    """Phase 2 — Independent Reading with mock agents."""

    def test_all_agents_generate_briefs(self, agents, registry, bb):
        """All four agents produce briefs after independent reading."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        assert set(bb.briefs.keys()) == {"specialist", "generalist", "skeptic", "moderator"}

    def test_brief_format_compliance(self, agents, registry, bb):
        """Every brief has valid top3_differential, confidence, and reasoning."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        for role, brief in bb.briefs.items():
            assert isinstance(brief.top3_differential, list)
            assert len(brief.top3_differential) >= 1, f"{role} must have at least 1 dx"
            assert 0.0 <= brief.confidence <= 1.0
            assert len(brief.reasoning) > 0, f"{role} must provide reasoning"
            assert brief.agent_role == role

    def test_token_tracking(self, agents, registry, bb):
        """Token count increases after independent reading."""
        plan_probe(bb, agents, registry)
        tokens_before = bb.total_tokens
        independent_read(bb, agents)
        assert bb.total_tokens > tokens_before


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Reveal & Critique
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase3Integration:
    """Phase 3 — Reveal & Critique (early exit gating)."""

    def test_early_exit_unanimous_agreement(self, registry, bb):
        """Early exit triggers when all agents agree and have high confidence."""
        # Build agents where everyone agrees, high confidence, no flags
        agent_map = create_mock_agents()
        moderator = agent_map["moderator"]

        def should_early_exit(briefs):
            non_mod = {r: b for r, b in briefs.items() if r != "moderator"}
            if len(non_mod) < 2:
                return False
            primaries = [
                b.top3_differential[0].strip().lower()
                for b in non_mod.values()
                if b.top3_differential
            ]
            if len(set(primaries)) != 1:
                return False
            if any(b.confidence < 0.60 for b in non_mod.values()):
                return False
            if any(b.disagreement_flags for b in non_mod.values()):
                return False
            return True

        moderator.should_early_exit = should_early_exit

        # All agree on melanoma, high confidence, no flags
        for role in ("specialist", "generalist", "skeptic"):
            bb.add_brief(AgentBrief(
                agent_role=role,
                top3_differential=["melanoma", "bcc", "nevus"],
                confidence=0.85,
                reasoning="Agreeing on melanoma.",
                disagreement_flags=[],
            ))
        bb.add_brief(AgentBrief(
            agent_role="moderator",
            top3_differential=["melanoma"],
            confidence=0.80,
            reasoning="Panel agrees.",
        ))

        reveal_critique(bb, agent_map)
        assert bb.early_exit is True

    def test_no_early_exit_on_disagreement(self, agents, registry, bb):
        """Early exit does NOT trigger when primary diagnoses differ."""
        plan_probe(bb, agents, registry)
        # Manually set briefs with disagreement
        bb.add_brief(AgentBrief(
            agent_role="specialist",
            top3_differential=["melanoma"],
            confidence=0.85,
            reasoning="Melanoma.",
        ))
        bb.add_brief(AgentBrief(
            agent_role="generalist",
            top3_differential=["basal_cell_carcinoma"],
            confidence=0.70,
            reasoning="BCC.",
        ))
        bb.add_brief(AgentBrief(
            agent_role="skeptic",
            top3_differential=["melanoma"],
            confidence=0.55,
            reasoning="Uncertain.",
        ))
        bb.add_brief(AgentBrief(
            agent_role="moderator",
            top3_differential=["melanoma"],
            confidence=0.75,
            reasoning="Split panel.",
        ))

        reveal_critique(bb, agents)
        assert bb.early_exit is False

    def test_no_early_exit_low_confidence(self, agents, registry, bb):
        """Early exit blocked if any non-moderator agent has confidence < 0.60."""
        plan_probe(bb, agents, registry)
        for role in ("specialist", "generalist", "skeptic"):
            conf = 0.40 if role == "skeptic" else 0.85
            bb.add_brief(AgentBrief(
                agent_role=role,
                top3_differential=["melanoma"],
                confidence=conf,
                reasoning="Test.",
            ))
        bb.add_brief(AgentBrief(
            agent_role="moderator",
            top3_differential=["melanoma"],
            confidence=0.80,
            reasoning="Low confidence agent.",
        ))

        reveal_critique(bb, agents)
        assert bb.early_exit is False

    def test_no_early_exit_with_flags(self, agents, registry, bb):
        """Early exit blocked if any agent raises disagreement flags."""
        plan_probe(bb, agents, registry)
        bb.add_brief(AgentBrief(
            agent_role="specialist",
            top3_differential=["melanoma"],
            confidence=0.85,
            reasoning="Confident.",
        ))
        bb.add_brief(AgentBrief(
            agent_role="generalist",
            top3_differential=["melanoma"],
            confidence=0.75,
            reasoning="Agreeing.",
        ))
        bb.add_brief(AgentBrief(
            agent_role="skeptic",
            top3_differential=["melanoma"],
            confidence=0.65,
            reasoning="Flagging concern.",
            disagreement_flags=["possible_overconfidence"],
        ))
        bb.add_brief(AgentBrief(
            agent_role="moderator",
            top3_differential=["melanoma"],
            confidence=0.80,
            reasoning="Panel.",
        ))

        reveal_critique(bb, agents)
        assert bb.early_exit is False


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Targeted Debate
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4Integration:
    """Phase 4 — Targeted Debate."""

    def test_debate_runs_on_disagreement(self, disagreeing_agents, registry, bb):
        """Debate produces turns when agents disagree."""
        plan_probe(bb, disagreeing_agents, registry)
        independent_read(bb, disagreeing_agents)
        reveal_critique(bb, disagreeing_agents)
        assert bb.early_exit is False

        targeted_debate(
            bb, disagreeing_agents, max_rounds=2,
            turn_order=["specialist", "generalist"],
        )
        assert len(bb.debate_log) == 4  # 2 rounds × 2 speakers

    def test_debate_skipped_on_early_exit(self, agents, registry, bb):
        """Phase 4 is a no-op when early_exit is True."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        # Force early exit
        bb.early_exit = True

        targeted_debate(bb, agents, max_rounds=3)
        assert len(bb.debate_log) == 0

    def test_debate_respects_max_rounds(self, disagreeing_agents, registry, bb):
        """Debate stops after max_rounds even if agents still disagree."""
        plan_probe(bb, disagreeing_agents, registry)
        independent_read(bb, disagreeing_agents)
        reveal_critique(bb, disagreeing_agents)

        targeted_debate(
            bb, disagreeing_agents, max_rounds=1,
            turn_order=["specialist", "generalist", "skeptic"],
        )
        # 1 round × 3 speakers = 3 turns
        assert len(bb.debate_log) == 3
        assert all(t.round_num == 1 for t in bb.debate_log)

    def test_debate_token_budget_enforcement(self, disagreeing_agents, registry, bb):
        """Debate terminates early when global token budget is exhausted."""
        plan_probe(bb, disagreeing_agents, registry)
        independent_read(bb, disagreeing_agents)
        reveal_critique(bb, disagreeing_agents)

        # Set a tiny budget that will already be exceeded after Phase 2
        targeted_debate(
            bb, disagreeing_agents,
            max_rounds=5,
            global_token_budget=1,  # Already exceeded
        )
        # Budget exceeded immediately → 0 debate turns
        assert len(bb.debate_log) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Synthesis
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase5Integration:
    """Phase 5 — Synthesis."""

    def test_final_diagnosis_ranked(self, agents, registry, bb):
        """Synthesis produces a non-empty ranked list of diagnoses."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        synthesis(bb, agents)
        assert len(bb.final_diagnosis) > 0
        # Top diagnosis should be melanoma (all mock agents agree)
        assert bb.final_diagnosis[0] == "melanoma"

    def test_consensus_score_calculated(self, agents, registry, bb):
        """Synthesis computes a consensus score in [0, 1]."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        synthesis(bb, agents)
        assert 0.0 <= bb.consensus_score <= 1.0
        # All non-moderator agents agree on melanoma → consensus = 1.0
        assert bb.consensus_score == 1.0

    def test_dissent_notes_captured(self, agents, registry, bb):
        """Synthesis collects disagreement flags as dissent notes."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        synthesis(bb, agents)
        # MockSkeptic raises 3 disagreement flags
        assert len(bb.dissent_notes) > 0
        assert any("skeptic" in note for note in bb.dissent_notes)

    def test_clinical_report_generated(self, agents, registry, bb):
        """Synthesis produces a non-empty clinical report string."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        synthesis(bb, agents)
        assert len(bb.clinical_report) > 0
        assert bb.case_id in bb.clinical_report

    def test_synthesis_aggregates_mappings(self, agents, registry, bb):
        """Synthesis aggregates ICD-10 and SNOMED mappings from all briefs."""
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        
        # Manually add mappings to the briefs before synthesis
        bb.briefs["specialist"].icd10_mappings = {"melanoma": "C43.9", "basal_cell_carcinoma": "C44.9"}
        bb.briefs["specialist"].snomed_mappings = {"melanoma": "372132005", "basal_cell_carcinoma": "13331008"}
        
        bb.briefs["generalist"].icd10_mappings = {"melanoma": "C43.9", "nevus": "D22.9"}
        bb.briefs["generalist"].snomed_mappings = {"melanoma": "372132005", "nevus": "400192004"}
        
        synthesis(bb, agents)
        
        # The final diagnoses should have mapping entries, others filtered out
        final_dx = [dx.lower() for dx in bb.final_diagnosis]
        
        # Verify icd10 mappings
        for dx, code in bb.final_icd10_mappings.items():
            assert dx in final_dx
            if dx == "melanoma":
                assert code == "C43.9"
            elif dx == "basal_cell_carcinoma":
                assert code == "C44.9"
                
        # Verify snomed mappings
        for dx, code in bb.final_snomed_mappings.items():
            assert dx in final_dx
            if dx == "melanoma":
                assert code == "372132005"
            elif dx == "basal_cell_carcinoma":
                assert code == "13331008"


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndPipeline:
    """Full pipeline tests through all 5 phases."""

    def test_full_pipeline_with_agreement(self, agents, registry, bb):
        """Full pipeline with unanimous agreement → early exit path.

        Default mock agents: specialist, generalist, skeptic all have
        'melanoma' as primary. Skeptic has flags but our early-exit
        function checks flags → no early exit. We override to remove flags.
        """
        # Override skeptic to remove disagreement flags
        original_skeptic_brief = agents["skeptic"].generate_brief

        def no_flag_brief(*args, **kwargs):
            brief = original_skeptic_brief(*args, **kwargs)
            brief.disagreement_flags = []
            brief.confidence = 0.75  # above 0.60 threshold
            return brief

        agents["skeptic"].generate_brief = no_flag_brief

        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        reveal_critique(bb, agents)

        assert bb.early_exit is True

        targeted_debate(bb, agents, max_rounds=3)
        assert len(bb.debate_log) == 0  # Skipped due to early exit

        synthesis(bb, agents)

        assert len(bb.final_diagnosis) > 0
        assert bb.final_diagnosis[0] == "melanoma"
        assert bb.consensus_score == 1.0
        assert len(bb.clinical_report) > 0

    def test_full_pipeline_with_disagreement(self, disagreeing_agents, registry, bb):
        """Full pipeline with disagreement → full debate path."""
        plan_probe(bb, disagreeing_agents, registry)
        independent_read(bb, disagreeing_agents)
        reveal_critique(bb, disagreeing_agents)

        assert bb.early_exit is False

        targeted_debate(
            bb, disagreeing_agents,
            max_rounds=2,
            turn_order=["specialist", "generalist", "skeptic"],
        )
        assert len(bb.debate_log) == 6  # 2 rounds × 3 speakers

        synthesis(bb, disagreeing_agents)

        assert len(bb.final_diagnosis) > 0
        assert bb.consensus_score < 1.0  # Not everyone agrees
        assert len(bb.clinical_report) > 0

    def test_full_pipeline_with_tool_failure(self, agents, bb):
        """Pipeline degrades gracefully when a tool fails."""
        # Create a registry with one broken tool
        registry = create_mock_registry()

        # Inject a failing tool
        from dermarbiter.tools.base_tool import BaseTool

        class FailingTool(BaseTool):
            @property
            def name(self) -> str:
                return "panderm_classifier"

            @property
            def description(self) -> str:
                return "Broken tool for testing."

            def run(self, image_path=None, query="") -> ToolOutput:
                raise RuntimeError("Simulated GPU OOM error")

        registry.register(FailingTool())  # Overwrites the mock

        plan_probe(bb, agents, registry)
        # panderm_classifier should produce an error card instead of crashing
        panderm_cards = [
            c for c in bb.evidence_cards
            if c.tool_output.tool_name == "panderm_classifier"
        ]
        assert len(panderm_cards) == 1
        assert panderm_cards[0].tool_output.confidence == 0.0
        assert "error" in panderm_cards[0].tool_output.result

        # Rest of pipeline still works
        independent_read(bb, agents)
        reveal_critique(bb, agents)
        targeted_debate(bb, agents, max_rounds=1)
        synthesis(bb, agents)
        assert len(bb.final_diagnosis) > 0
        assert len(bb.clinical_report) > 0

    def test_full_pipeline_state_preservation(self, agents, registry, bb):
        """All blackboard fields are populated after a full run."""
        # Remove flags for early-exit check to be consistent
        plan_probe(bb, agents, registry)
        independent_read(bb, agents)
        reveal_critique(bb, agents)
        targeted_debate(bb, agents, max_rounds=1)
        synthesis(bb, agents)

        # Case identity
        assert bb.case_id == "TEST-CASE-001"
        assert bb.query != ""
        assert bb.patient_context != {}

        # Evidence layer
        assert len(bb.evidence_cards) > 0
        assert bb.total_tool_calls > 0

        # Agent opinions
        assert len(bb.briefs) == 4

        # Final outputs
        assert len(bb.final_diagnosis) > 0
        assert 0.0 <= bb.consensus_score <= 1.0
        assert isinstance(bb.dissent_notes, list)
        assert len(bb.clinical_report) > 0

        # Telemetry
        assert bb.total_tokens > 0

    def test_orchestrator_run(self, agents, registry, bb):
        """Full pipeline through the LangGraph orchestrator."""
        orchestrator = DermArbiterOrchestrator(
            agents=agents,
            tool_registry=registry,
            max_rounds=2,
            max_tokens_per_turn=100,
            global_token_budget=100_000,
        )

        final_state = orchestrator.run(bb)

        assert isinstance(final_state, BlackboardState)
        # Evidence was gathered
        assert len(final_state.evidence_cards) > 0
        # Briefs were produced
        assert len(final_state.briefs) == 4
        # Synthesis ran
        assert len(final_state.final_diagnosis) > 0
        assert 0.0 <= final_state.consensus_score <= 1.0
        assert len(final_state.clinical_report) > 0
        # Tokens were tracked
        assert final_state.total_tokens > 0
