"""Unit tests for DermArbiter Blackboard data models.

Tests cover:
  • BlackboardState creation and defaults
  • Adding evidence cards, briefs, and debate turns
  • Early exit gating default
  • Serialisation (to_dict)
  • Pydantic validation (confidence bounds, brief constraints)
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
from dermarbiter.tools.base_tool import ToolOutput


# ═══════════════════════════════════════════════════════════════════════════
# BlackboardState — creation
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateEmptyBlackboard:
    """Tests for creating an empty BlackboardState."""

    def test_create_empty_blackboard(self, empty_blackboard: BlackboardState):
        """A fresh blackboard should have no evidence, briefs, or debate."""
        bb = empty_blackboard
        assert bb.case_id == "TEST-CASE-001"
        assert bb.image_path is None
        assert len(bb.evidence_cards) == 0
        assert len(bb.briefs) == 0
        assert len(bb.debate_log) == 0
        assert bb.total_tokens == 0
        assert bb.total_tool_calls == 0
        assert bb.errors == []

    def test_empty_blackboard_has_query(self, empty_blackboard: BlackboardState):
        """The query field should be populated from the sample case."""
        assert "55-year-old male" in empty_blackboard.query

    def test_empty_blackboard_has_patient_context(
        self, empty_blackboard: BlackboardState
    ):
        """Patient context should contain demographics."""
        ctx = empty_blackboard.patient_context
        assert ctx["fitzpatrick_type"] == "III"
        assert ctx["age"] == 55

    def test_empty_blackboard_defaults(self):
        """A completely default BlackboardState should be valid."""
        bb = BlackboardState()
        assert bb.consensus_score == 0.0
        assert bb.final_diagnosis == []
        assert bb.clinical_report == ""
        assert bb.early_exit is False


# ═══════════════════════════════════════════════════════════════════════════
# Adding evidence cards
# ═══════════════════════════════════════════════════════════════════════════

class TestAddEvidenceCard:
    """Tests for adding EvidenceCards to the blackboard."""

    def test_add_evidence_card(
        self,
        empty_blackboard: BlackboardState,
        sample_evidence_cards: list[EvidenceCard],
    ):
        """Adding a card should append to evidence_cards and bump counter."""
        bb = empty_blackboard
        card = sample_evidence_cards[0]

        bb.add_evidence_card(card)

        assert len(bb.evidence_cards) == 1
        assert bb.evidence_cards[0].card_id == "EC-001"
        assert bb.total_tool_calls == 1

    def test_add_multiple_evidence_cards(
        self,
        empty_blackboard: BlackboardState,
        sample_evidence_cards: list[EvidenceCard],
    ):
        """Adding multiple cards should accumulate correctly."""
        bb = empty_blackboard
        for card in sample_evidence_cards:
            bb.add_evidence_card(card)

        assert len(bb.evidence_cards) == 4
        assert bb.total_tool_calls == 4

    def test_evidence_card_ids_unique(
        self, sample_evidence_cards: list[EvidenceCard]
    ):
        """All sample evidence cards should have unique IDs."""
        ids = [c.card_id for c in sample_evidence_cards]
        assert len(ids) == len(set(ids))

    def test_evidence_card_has_tool_output(
        self, sample_evidence_cards: list[EvidenceCard]
    ):
        """EvidenceCard should contain a valid ToolOutput."""
        card = sample_evidence_cards[0]
        assert card.tool_output.tool_name == "panderm_classifier"
        assert card.tool_output.confidence == 0.62


# ═══════════════════════════════════════════════════════════════════════════
# Adding briefs
# ═══════════════════════════════════════════════════════════════════════════

class TestAddBrief:
    """Tests for adding AgentBriefs to the blackboard."""

    def test_add_brief(
        self,
        empty_blackboard: BlackboardState,
        sample_briefs: dict[str, AgentBrief],
    ):
        """Adding a brief should key it by agent role."""
        bb = empty_blackboard
        brief = sample_briefs["specialist"]

        bb.add_brief(brief)

        assert "specialist" in bb.briefs
        assert bb.briefs["specialist"].confidence == 0.85

    def test_add_all_briefs(
        self,
        empty_blackboard: BlackboardState,
        sample_briefs: dict[str, AgentBrief],
    ):
        """All four roles should be present after adding all briefs."""
        bb = empty_blackboard
        for brief in sample_briefs.values():
            bb.add_brief(brief)

        assert set(bb.briefs.keys()) == {
            "specialist", "generalist", "skeptic", "moderator"
        }

    def test_brief_overwrite(
        self,
        empty_blackboard: BlackboardState,
        sample_briefs: dict[str, AgentBrief],
    ):
        """Re-adding a brief with the same role should overwrite."""
        bb = empty_blackboard
        bb.add_brief(sample_briefs["specialist"])

        updated = AgentBrief(
            agent_role="specialist",
            top3_differential=["melanoma"],
            confidence=0.95,
        )
        bb.add_brief(updated)

        assert bb.briefs["specialist"].confidence == 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Adding debate turns
# ═══════════════════════════════════════════════════════════════════════════

class TestAddDebateTurn:
    """Tests for adding DebateTurns to the blackboard."""

    def test_add_debate_turn(self, empty_blackboard: BlackboardState):
        """A debate turn should be appended to the log."""
        bb = empty_blackboard
        turn = DebateTurn(
            round_num=1,
            speaker="skeptic",
            argument="I disagree with the melanoma assessment.",
            token_count=8,
        )

        bb.add_debate_turn(turn)

        assert len(bb.debate_log) == 1
        assert bb.debate_log[0].speaker == "skeptic"
        assert bb.debate_log[0].round_num == 1

    def test_debate_turn_ordering(self, empty_blackboard: BlackboardState):
        """Turns should preserve insertion order."""
        bb = empty_blackboard
        roles = ["skeptic", "specialist", "moderator"]
        for i, role in enumerate(roles):
            bb.add_debate_turn(
                DebateTurn(
                    round_num=i + 1,
                    speaker=role,
                    argument=f"Turn from {role}",
                    token_count=4,
                )
            )

        assert [t.speaker for t in bb.debate_log] == roles

    def test_debate_turn_token_accumulation(self, empty_blackboard: BlackboardState):
        """Token counts should accumulate in total_tokens."""
        bb = empty_blackboard
        bb.add_debate_turn(DebateTurn(round_num=1, speaker="skeptic", argument="a", token_count=10))
        bb.add_debate_turn(DebateTurn(round_num=1, speaker="specialist", argument="b", token_count=15))
        assert bb.total_tokens == 25


# ═══════════════════════════════════════════════════════════════════════════
# Early exit
# ═══════════════════════════════════════════════════════════════════════════

class TestEarlyExit:
    """Tests for the early exit gating flag."""

    def test_early_exit_default_false(self, empty_blackboard: BlackboardState):
        """Early exit should default to False."""
        assert empty_blackboard.early_exit is False

    def test_early_exit_can_be_set(self, empty_blackboard: BlackboardState):
        """Early exit should be settable."""
        empty_blackboard.early_exit = True
        assert empty_blackboard.early_exit is True


# ═══════════════════════════════════════════════════════════════════════════
# Serialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestBlackboardToDict:
    """Tests for BlackboardState serialisation."""

    def test_blackboard_to_dict(
        self, populated_blackboard: BlackboardState
    ):
        """to_dict() should return a JSON-serialisable dictionary."""
        d = populated_blackboard.to_dict()

        assert isinstance(d, dict)
        assert d["case_id"] == "TEST-CASE-001"
        assert len(d["evidence_cards"]) == 4
        assert len(d["briefs"]) == 4
        assert len(d["debate_log"]) == 2

    def test_to_dict_contains_all_keys(
        self, populated_blackboard: BlackboardState
    ):
        """Serialised dict should contain all expected top-level keys."""
        d = populated_blackboard.to_dict()
        expected_keys = {
            "case_id", "image_path", "query", "patient_context",
            "evidence_cards", "briefs", "debate_log", "early_exit",
            "final_diagnosis", "consensus_score",
            "clinical_report", "total_tokens", "total_tool_calls",
            "errors",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_evidence_card_structure(
        self, populated_blackboard: BlackboardState
    ):
        """Each evidence card in the dict should have required fields."""
        d = populated_blackboard.to_dict()
        card = d["evidence_cards"][0]
        assert "card_id" in card
        assert "tool_output" in card
        assert "requested_by" in card

    def test_to_dict_briefs_structure(
        self, populated_blackboard: BlackboardState
    ):
        """Each brief in the dict should have required fields."""
        d = populated_blackboard.to_dict()
        specialist = d["briefs"]["specialist"]
        assert "agent_role" in specialist
        assert "top3_differential" in specialist
        assert "confidence" in specialist


# ═══════════════════════════════════════════════════════════════════════════
# ToolOutput validation (from tools.base_tool)
# ═══════════════════════════════════════════════════════════════════════════

class TestToolOutputValidation:
    """Pydantic validation tests for ToolOutput confidence bounds."""

    def test_tool_output_valid_confidence(self):
        """Confidence in [0, 1] should be accepted."""
        output = ToolOutput(
            tool_name="test_tool",
            result={"key": "value"},
            confidence=0.75,
            raw_text="Test output",
        )
        assert output.confidence == 0.75

    def test_tool_output_confidence_zero(self):
        """Confidence = 0.0 should be valid."""
        output = ToolOutput(
            tool_name="test_tool",
            result={},
            confidence=0.0,
            raw_text="Zero confidence",
        )
        assert output.confidence == 0.0

    def test_tool_output_confidence_one(self):
        """Confidence = 1.0 should be valid."""
        output = ToolOutput(
            tool_name="test_tool",
            result={},
            confidence=1.0,
            raw_text="Full confidence",
        )
        assert output.confidence == 1.0

    def test_tool_output_confidence_too_high(self):
        """Confidence > 1.0 should be rejected."""
        with pytest.raises(Exception):  # ValidationError
            ToolOutput(
                tool_name="test_tool",
                result={},
                confidence=1.5,
                raw_text="Invalid",
            )

    def test_tool_output_confidence_negative(self):
        """Confidence < 0.0 should be rejected."""
        with pytest.raises(Exception):  # ValidationError
            ToolOutput(
                tool_name="test_tool",
                result={},
                confidence=-0.1,
                raw_text="Invalid",
            )

    def test_tool_output_has_timestamp(self):
        """ToolOutput should auto-generate a timestamp."""
        output = ToolOutput(
            tool_name="test_tool",
            result={},
            confidence=0.5,
            raw_text="Test",
        )
        assert output.timestamp is not None
        assert len(output.timestamp) > 0


# ═══════════════════════════════════════════════════════════════════════════
# AgentBrief validation (from core.blackboard)
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentBriefValidation:
    """Pydantic validation tests for AgentBrief."""

    def test_agent_brief_valid(self):
        """A well-formed AgentBrief should be accepted."""
        brief = AgentBrief(
            agent_role="specialist",
            top3_differential=["melanoma", "bcc", "nevus"],
            confidence=0.85,
            cited_cards=["EC-001"],
        )
        assert brief.agent_role == "specialist"
        assert len(brief.top3_differential) == 3

    def test_agent_brief_confidence_clamped_high(self):
        """Confidence > 1.0 should be clamped to 1.0 by the validator."""
        brief = AgentBrief(
            agent_role="test",
            top3_differential=["melanoma"],
            confidence=1.5,
        )
        assert brief.confidence == 1.0

    def test_agent_brief_confidence_clamped_low(self):
        """Confidence < 0.0 should be clamped to 0.0 by the validator."""
        brief = AgentBrief(
            agent_role="test",
            top3_differential=["melanoma"],
            confidence=-0.1,
        )
        assert brief.confidence == 0.0

    def test_agent_brief_empty_differential_allowed(self):
        """Empty differential list should be allowed (min_length=0)."""
        brief = AgentBrief(
            agent_role="test",
            top3_differential=[],
            confidence=0.5,
        )
        assert brief.top3_differential == []

    def test_agent_brief_disagreement_flags_default_empty(self):
        """Disagreement flags should default to an empty list."""
        brief = AgentBrief(
            agent_role="generalist",
            top3_differential=["melanoma"],
            confidence=0.7,
        )
        assert brief.disagreement_flags == []

    def test_agent_brief_has_reasoning(self):
        """AgentBrief reasoning should default to empty string."""
        brief = AgentBrief(
            agent_role="test",
            top3_differential=["melanoma"],
            confidence=0.5,
        )
        assert brief.reasoning == ""


# ═══════════════════════════════════════════════════════════════════════════
# Blackboard helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestBlackboardHelpers:
    """Tests for BlackboardState helper methods."""

    def test_get_brief_summary_empty(self, empty_blackboard: BlackboardState):
        """Brief summary should indicate no briefs when empty."""
        assert "No briefs" in empty_blackboard.get_brief_summary()

    def test_get_brief_summary_populated(self, populated_blackboard: BlackboardState):
        """Brief summary should include all agent roles."""
        summary = populated_blackboard.get_brief_summary()
        assert "SPECIALIST" in summary
        assert "GENERALIST" in summary
        assert "SKEPTIC" in summary

    def test_get_evidence_summary_empty(self, empty_blackboard: BlackboardState):
        """Evidence summary should indicate no cards when empty."""
        assert "No evidence" in empty_blackboard.get_evidence_summary()

    def test_get_evidence_summary_populated(self, populated_blackboard: BlackboardState):
        """Evidence summary should reference card IDs."""
        summary = populated_blackboard.get_evidence_summary()
        assert "EC-001" in summary
        assert "panderm_classifier" in summary
