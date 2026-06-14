"""Comprehensive unit tests for the four concrete DermArbiter agents.

Tests cover:
    • SpecialistAgent — tool proposals, brief generation, fallback logic, debate
    • GeneralistAgent — tool proposals, brief generation, debate
    • SkepticAgent    — empty tool list, no-tool-access semantics, brief generation
    • ModeratorAgent  — tool proposals, early-exit gating, report synthesis
    • Common checks   — BaseAgent subclass assertion, system prompts, token counting

All tests use ``unittest.mock`` to avoid any real LLM calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.agents.generalist import GeneralistAgent
from dermarbiter.agents.moderator import ModeratorAgent
from dermarbiter.agents.skeptic import SkepticAgent
from dermarbiter.agents.specialist import SpecialistAgent
from dermarbiter.core.blackboard import AgentBrief, BlackboardState, EvidenceCard
from dermarbiter.core.config import AgentConfig


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

# --- Controlled LLM responses used across tests ---

_VALID_BRIEF_JSON = json.dumps(
    {
        "top3_differential": ["melanoma", "bcc", "nevus"],
        "confidence": 0.85,
        "reasoning": "Multi-modal evidence converges on melanoma.",
        "cited_cards": ["EC-001", "EC-002"],
        "disagreement_flags": [],
    }
)

_VALID_REPORT_TEXT = (
    "CLINICAL REPORT — Primary diagnosis: melanoma (confidence 0.85). "
    "Differential: BCC, nevus. Recommend excisional biopsy."
)


@pytest.fixture
def mock_router() -> MagicMock:
    """A ``MagicMock`` standing in for ``ModelRouter``.

    ``router.call(...)`` returns valid brief JSON by default.
    """
    router = MagicMock()
    router.call.return_value = _VALID_BRIEF_JSON
    return router


@pytest.fixture
def specialist_config() -> AgentConfig:
    return AgentConfig(
        role="specialist",
        model_backend="google_api",
        model_name="gemini-2.5-flash",
        temperature=0.1,
        system_prompt_path="",
        has_tool_access=True,
    )


@pytest.fixture
def generalist_config() -> AgentConfig:
    return AgentConfig(
        role="generalist",
        model_backend="google_api",
        model_name="gemini-2.0-flash",
        temperature=0.3,
        system_prompt_path="",
        has_tool_access=True,
    )


@pytest.fixture
def skeptic_config() -> AgentConfig:
    return AgentConfig(
        role="skeptic",
        model_backend="google_api",
        model_name="gemini-2.0-flash",
        temperature=0.5,
        system_prompt_path="",
        has_tool_access=False,
    )


@pytest.fixture
def moderator_config() -> AgentConfig:
    return AgentConfig(
        role="moderator",
        model_backend="google_api",
        model_name="gemini-2.5-flash",
        temperature=0.2,
        system_prompt_path="",
        has_tool_access=True,
    )


# --- Agent instance fixtures ---


@pytest.fixture
def specialist(specialist_config: AgentConfig, mock_router: MagicMock) -> SpecialistAgent:
    return SpecialistAgent(config=specialist_config, model_router=mock_router)


@pytest.fixture
def generalist(generalist_config: AgentConfig, mock_router: MagicMock) -> GeneralistAgent:
    return GeneralistAgent(config=generalist_config, model_router=mock_router)


@pytest.fixture
def skeptic(skeptic_config: AgentConfig, mock_router: MagicMock) -> SkepticAgent:
    return SkepticAgent(config=skeptic_config, model_router=mock_router)


@pytest.fixture
def moderator(moderator_config: AgentConfig, mock_router: MagicMock) -> ModeratorAgent:
    return ModeratorAgent(config=moderator_config, model_router=mock_router)


@pytest.fixture
def sample_case_info() -> dict:
    """Minimal case info dict for ``propose_tools``."""
    return {
        "query": "Pigmented lesion on upper back, 55yo male.",
        "image_path": "/tmp/test_lesion.jpg",
        "patient_context": {"age": 55, "sex": "Male"},
    }


@pytest.fixture
def opponent_brief() -> AgentBrief:
    """A brief to use as the opponent in debate tests."""
    return AgentBrief(
        agent_role="generalist",
        top3_differential=["melanoma", "bcc", "nevus"],
        confidence=0.68,
        reasoning="Broad differential based on visual assessment.",
        cited_cards=["EC-001"],
        disagreement_flags=[],
    )


# ═══════════════════════════════════════════════════════════════════════════
# TestSpecialistAgent
# ═══════════════════════════════════════════════════════════════════════════


class TestSpecialistAgent:
    """Tests for ``SpecialistAgent``."""

    def test_specialist_role_is_specialist(
        self, specialist: SpecialistAgent
    ) -> None:
        assert specialist.role == "specialist"

    def test_specialist_propose_tools(
        self, specialist: SpecialistAgent, sample_case_info: dict
    ) -> None:
        """Proposed tools include panderm_classifier and make_annotator."""
        tools = specialist.propose_tools(sample_case_info)
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "panderm_classifier" in tools
        assert "make_annotator" in tools

    def test_specialist_propose_tools_always_includes_core(
        self, specialist: SpecialistAgent, sample_case_info: dict
    ) -> None:
        """Core tools (panderm, guideline_rag) are always present."""
        tools = specialist.propose_tools(sample_case_info)
        assert "panderm_classifier" in tools
        assert "guideline_rag" in tools

    def test_specialist_propose_tools_no_duplicates(
        self, specialist: SpecialistAgent, sample_case_info: dict
    ) -> None:
        """Tool list contains no duplicates."""
        tools = specialist.propose_tools(sample_case_info)
        assert len(tools) == len(set(tools))

    def test_specialist_generate_brief_returns_agent_brief(
        self,
        specialist: SpecialistAgent,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """With valid LLM JSON, ``generate_brief`` returns an AgentBrief."""
        brief = specialist.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == "specialist"
        assert brief.top3_differential == ["melanoma", "bcc", "nevus"]
        assert brief.confidence == pytest.approx(0.85)
        assert len(brief.cited_cards) > 0

    def test_specialist_generate_brief_fallback_on_bad_json(
        self,
        specialist: SpecialistAgent,
        mock_router: MagicMock,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """When the LLM returns garbage, a fallback brief is produced."""
        mock_router.call.return_value = "NOT VALID JSON {{{garbage"
        brief = specialist.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == "specialist"
        assert brief.confidence <= 0.5  # fallback is conservative
        assert "parse_error" in brief.disagreement_flags

    def test_specialist_generate_brief_calls_llm(
        self,
        specialist: SpecialistAgent,
        mock_router: MagicMock,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """Verify the LLM is actually invoked during brief generation."""
        specialist.generate_brief(sample_evidence_cards)
        mock_router.call.assert_called_once()

    def test_specialist_generate_argument_returns_string(
        self,
        specialist: SpecialistAgent,
        mock_router: MagicMock,
        opponent_brief: AgentBrief,
    ) -> None:
        """``generate_argument`` returns a string."""
        mock_router.call.return_value = "Rebuttal: evidence supports melanoma."
        result = specialist.generate_argument("primary diagnosis", opponent_brief)
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestGeneralistAgent
# ═══════════════════════════════════════════════════════════════════════════


class TestGeneralistAgent:
    """Tests for ``GeneralistAgent``."""

    def test_generalist_role_is_generalist(
        self, generalist: GeneralistAgent
    ) -> None:
        assert generalist.role == "generalist"

    def test_generalist_propose_tools(
        self, generalist: GeneralistAgent, sample_case_info: dict
    ) -> None:
        """Proposed tools include general_vqa and fairness_probe."""
        tools = generalist.propose_tools(sample_case_info)
        assert isinstance(tools, list)
        assert "general_vqa" in tools
        assert "fairness_probe" in tools

    def test_generalist_propose_tools_includes_classifier(
        self, generalist: GeneralistAgent, sample_case_info: dict
    ) -> None:
        """Generalist also uses panderm_classifier."""
        tools = generalist.propose_tools(sample_case_info)
        assert "panderm_classifier" in tools

    def test_generalist_generate_brief(
        self,
        generalist: GeneralistAgent,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """``generate_brief`` returns a properly typed AgentBrief."""
        brief = generalist.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == "generalist"
        assert 0.0 <= brief.confidence <= 1.0

    def test_generalist_has_tool_access(
        self, generalist: GeneralistAgent
    ) -> None:
        """Generalist config has ``has_tool_access=True``."""
        assert generalist.has_tool_access is True

    def test_generalist_generate_brief_fallback(
        self,
        generalist: GeneralistAgent,
        mock_router: MagicMock,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """Garbage LLM response triggers fallback brief."""
        mock_router.call.return_value = "<html>error</html>"
        brief = generalist.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert "parse_error" in brief.disagreement_flags

    def test_generalist_generate_argument(
        self,
        generalist: GeneralistAgent,
        mock_router: MagicMock,
        opponent_brief: AgentBrief,
    ) -> None:
        """``generate_argument`` returns a non-empty string."""
        mock_router.call.return_value = "Consider broader differential."
        result = generalist.generate_argument("diagnostic breadth", opponent_brief)
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestSkepticAgent
# ═══════════════════════════════════════════════════════════════════════════


class TestSkepticAgent:
    """Tests for ``SkepticAgent``."""

    def test_skeptic_role_is_skeptic(self, skeptic: SkepticAgent) -> None:
        assert skeptic.role == "skeptic"

    def test_skeptic_propose_tools_empty(
        self, skeptic: SkepticAgent, sample_case_info: dict
    ) -> None:
        """Skeptic proposes zero tools."""
        tools = skeptic.propose_tools(sample_case_info)
        assert tools == []

    def test_skeptic_has_no_tool_access(self, skeptic: SkepticAgent) -> None:
        """Skeptic config has ``has_tool_access=False``."""
        assert skeptic.has_tool_access is False

    def test_skeptic_generate_brief(
        self,
        skeptic: SkepticAgent,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """Skeptic can generate a brief even without tool access."""
        brief = skeptic.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == "skeptic"
        assert 0.0 <= brief.confidence <= 1.0

    def test_skeptic_generate_brief_with_empty_cards(
        self, skeptic: SkepticAgent
    ) -> None:
        """Skeptic handles empty evidence card list gracefully."""
        brief = skeptic.generate_brief([])
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == "skeptic"

    def test_skeptic_generate_argument(
        self,
        skeptic: SkepticAgent,
        mock_router: MagicMock,
        opponent_brief: AgentBrief,
    ) -> None:
        """Skeptic produces string arguments."""
        mock_router.call.return_value = "Overconfidence detected."
        result = skeptic.generate_argument("confidence level", opponent_brief)
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestModeratorAgent
# ═══════════════════════════════════════════════════════════════════════════


class TestModeratorAgent:
    """Tests for ``ModeratorAgent``."""

    def test_moderator_role_is_moderator(
        self, moderator: ModeratorAgent
    ) -> None:
        assert moderator.role == "moderator"

    def test_moderator_propose_tools(
        self, moderator: ModeratorAgent, sample_case_info: dict
    ) -> None:
        """Moderator proposes ontology_graph."""
        tools = moderator.propose_tools(sample_case_info)
        assert isinstance(tools, list)
        assert "ontology_graph" in tools

    def test_moderator_propose_tools_includes_uncertainty(
        self, moderator: ModeratorAgent, sample_case_info: dict
    ) -> None:
        """Moderator also requests uncertainty_probe."""
        tools = moderator.propose_tools(sample_case_info)
        assert "uncertainty_probe" in tools

    def test_moderator_generate_brief(
        self,
        moderator: ModeratorAgent,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """Moderator produces a valid AgentBrief."""
        brief = moderator.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == "moderator"

    # --- Early Exit Gating ---

    def test_moderator_should_early_exit_unanimous(
        self, moderator: ModeratorAgent
    ) -> None:
        """All agents agree on melanoma, no flags → early exit True."""
        briefs = {
            "specialist": AgentBrief(
                agent_role="specialist",
                top3_differential=["melanoma", "bcc", "nevus"],
                confidence=0.85,
                reasoning="Strong evidence.",
                disagreement_flags=[],
            ),
            "generalist": AgentBrief(
                agent_role="generalist",
                top3_differential=["melanoma", "nevus", "bcc"],
                confidence=0.70,
                reasoning="Agrees on melanoma.",
                disagreement_flags=[],
            ),
            "skeptic": AgentBrief(
                agent_role="skeptic",
                top3_differential=["melanoma", "dysplastic_nevus"],
                confidence=0.55,
                reasoning="Reluctantly agrees.",
                disagreement_flags=[],
            ),
        }
        assert moderator.should_early_exit(briefs) is True

    def test_moderator_should_early_exit_disagreement(
        self, moderator: ModeratorAgent
    ) -> None:
        """Agents disagree on top diagnosis → early exit False."""
        briefs = {
            "specialist": AgentBrief(
                agent_role="specialist",
                top3_differential=["melanoma", "bcc"],
                confidence=0.85,
                reasoning="Melanoma primary.",
                disagreement_flags=[],
            ),
            "generalist": AgentBrief(
                agent_role="generalist",
                top3_differential=["bcc", "melanoma"],
                confidence=0.60,
                reasoning="BCC primary.",
                disagreement_flags=[],
            ),
        }
        assert moderator.should_early_exit(briefs) is False

    def test_moderator_should_early_exit_with_flags(
        self, moderator: ModeratorAgent
    ) -> None:
        """Even unanimous diagnosis with disagreement flags → no early exit."""
        briefs = {
            "specialist": AgentBrief(
                agent_role="specialist",
                top3_differential=["melanoma"],
                confidence=0.85,
                disagreement_flags=[],
            ),
            "skeptic": AgentBrief(
                agent_role="skeptic",
                top3_differential=["melanoma"],
                confidence=0.50,
                disagreement_flags=["overconfidence:specialist"],
            ),
        }
        assert moderator.should_early_exit(briefs) is False

    def test_moderator_should_early_exit_single_brief(
        self, moderator: ModeratorAgent
    ) -> None:
        """Only one brief submitted → cannot declare consensus."""
        briefs = {
            "specialist": AgentBrief(
                agent_role="specialist",
                top3_differential=["melanoma"],
                confidence=0.85,
                disagreement_flags=[],
            ),
        }
        assert moderator.should_early_exit(briefs) is False

    def test_moderator_should_early_exit_empty_briefs(
        self, moderator: ModeratorAgent
    ) -> None:
        """Empty briefs dict → no early exit."""
        assert moderator.should_early_exit({}) is False

    def test_moderator_should_early_exit_case_insensitive(
        self, moderator: ModeratorAgent
    ) -> None:
        """Diagnosis comparison is case-insensitive."""
        briefs = {
            "specialist": AgentBrief(
                agent_role="specialist",
                top3_differential=["Melanoma"],
                confidence=0.85,
                disagreement_flags=[],
            ),
            "generalist": AgentBrief(
                agent_role="generalist",
                top3_differential=["melanoma"],
                confidence=0.70,
                disagreement_flags=[],
            ),
        }
        assert moderator.should_early_exit(briefs) is True

    # --- Final Report Synthesis ---

    def test_moderator_synthesize_final_report(
        self,
        moderator: ModeratorAgent,
        mock_router: MagicMock,
    ) -> None:
        """``synthesize_final_report`` returns a string via LLM."""
        mock_router.call.return_value = _VALID_REPORT_TEXT
        bb = BlackboardState(
            case_id="TEST-001",
            query="Test case",
        )
        bb.add_brief(
            AgentBrief(
                agent_role="specialist",
                top3_differential=["melanoma"],
                confidence=0.85,
            )
        )
        report = moderator.synthesize_final_report(bb)
        assert isinstance(report, str)
        assert len(report) > 0
        mock_router.call.assert_called_once()

    def test_moderator_synthesize_final_report_uses_blackboard_content(
        self,
        moderator: ModeratorAgent,
        mock_router: MagicMock,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """Verify the prompt includes blackboard brief/evidence summaries."""
        mock_router.call.return_value = _VALID_REPORT_TEXT
        bb = BlackboardState(case_id="TEST-002", query="Test")
        for card in sample_evidence_cards:
            bb.add_evidence_card(card)
        bb.add_brief(
            AgentBrief(
                agent_role="specialist",
                top3_differential=["melanoma"],
                confidence=0.85,
            )
        )

        moderator.synthesize_final_report(bb)

        # Inspect the prompt sent to the LLM
        call_args = mock_router.call.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages", call_args[0][1])
        prompt_text = " ".join(m.get("content", "") for m in messages)
        assert "BRIEFS" in prompt_text or "EVIDENCE" in prompt_text


# ═══════════════════════════════════════════════════════════════════════════
# TestAgentCommon — cross-cutting checks for all agents
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentCommon:
    """Cross-cutting tests that apply to all four agent types."""

    @pytest.fixture(
        params=[
            ("specialist", SpecialistAgent),
            ("generalist", GeneralistAgent),
            ("skeptic", SkepticAgent),
            ("moderator", ModeratorAgent),
        ],
        ids=["specialist", "generalist", "skeptic", "moderator"],
    )
    def agent_pair(
        self, request: pytest.FixtureRequest, mock_router: MagicMock
    ) -> tuple[str, BaseAgent]:
        """Parametrized fixture yielding (role, agent_instance)."""
        role, cls = request.param
        config = AgentConfig(
            role=role,
            model_backend="google_api",
            model_name="gemini-2.0-flash",
            temperature=0.3,
            system_prompt_path="",
            has_tool_access=(role != "skeptic"),
        )
        return role, cls(config=config, model_router=mock_router)

    def test_all_agents_are_base_agent_subclass(
        self, agent_pair: tuple[str, BaseAgent]
    ) -> None:
        """Every concrete agent is a subclass of BaseAgent."""
        role, agent = agent_pair
        assert isinstance(agent, BaseAgent), (
            f"{agent.__class__.__name__} is not a BaseAgent subclass"
        )

    def test_all_agents_have_system_prompt(
        self, agent_pair: tuple[str, BaseAgent]
    ) -> None:
        """Every agent exposes a non-empty system prompt string."""
        role, agent = agent_pair
        prompt = agent.system_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 10, (
            f"System prompt for {role} is too short: {prompt!r}"
        )

    def test_all_agents_have_role_in_system_prompt(
        self, agent_pair: tuple[str, BaseAgent]
    ) -> None:
        """The default system prompt mentions the agent's role."""
        role, agent = agent_pair
        assert role in agent.system_prompt

    def test_count_tokens_not_zero(
        self, agent_pair: tuple[str, BaseAgent]
    ) -> None:
        """``_count_tokens`` returns a positive integer for non-empty text."""
        _, agent = agent_pair
        count = agent._count_tokens("This is a test string for token counting.")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_zero_for_empty(
        self, agent_pair: tuple[str, BaseAgent]
    ) -> None:
        """``_count_tokens`` returns 0 for an empty string."""
        _, agent = agent_pair
        assert agent._count_tokens("") == 0

    def test_repr_includes_role(
        self, agent_pair: tuple[str, BaseAgent]
    ) -> None:
        """The repr string includes the agent's role."""
        role, agent = agent_pair
        r = repr(agent)
        assert role in r

    def test_propose_tools_returns_list(
        self,
        agent_pair: tuple[str, BaseAgent],
        sample_case_info: dict,
    ) -> None:
        """``propose_tools`` always returns a list."""
        _, agent = agent_pair
        tools = agent.propose_tools(sample_case_info)
        assert isinstance(tools, list)
        for t in tools:
            assert isinstance(t, str)

    def test_generate_brief_returns_agent_brief(
        self,
        agent_pair: tuple[str, BaseAgent],
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """``generate_brief`` always returns an AgentBrief."""
        role, agent = agent_pair
        brief = agent.generate_brief(sample_evidence_cards)
        assert isinstance(brief, AgentBrief)
        assert brief.agent_role == role

    def test_generate_brief_parses_mappings(
        self,
        agent_pair: tuple[str, BaseAgent],
        mock_router: MagicMock,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """Verify that generate_brief parses ICD-10 and SNOMED mappings from LLM response."""
        role, agent = agent_pair
        mapping_brief_json = json.dumps(
            {
                "top3_differential": ["melanoma", "bcc", "nevus"],
                "confidence": 0.85,
                "reasoning": f"Multi-modal evidence converges on melanoma for {role}.",
                "cited_cards": ["EC-001", "EC-002"],
                "disagreement_flags": [],
                "icd10_mappings": {"melanoma": "C43.9", "bcc": "C44.9"},
                "snomed_mappings": {"melanoma": "372132005", "bcc": "13331008"},
            }
        )
        mock_router.call.return_value = mapping_brief_json
        brief = agent.generate_brief(sample_evidence_cards)
        assert brief.icd10_mappings == {"melanoma": "C43.9", "bcc": "C44.9"}
        assert brief.snomed_mappings == {"melanoma": "372132005", "bcc": "13331008"}

    def test_generate_argument_returns_string(
        self,
        agent_pair: tuple[str, BaseAgent],
        mock_router: MagicMock,
        opponent_brief: AgentBrief,
    ) -> None:
        """``generate_argument`` always returns a string."""
        mock_router.call.return_value = "A debate argument."
        _, agent = agent_pair
        result = agent.generate_argument("test topic", opponent_brief)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# TestEdgeCases — boundary conditions and robustness
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_specialist_brief_with_empty_evidence(
        self, specialist: SpecialistAgent
    ) -> None:
        """Specialist handles empty evidence cards without crashing."""
        brief = specialist.generate_brief([])
        assert isinstance(brief, AgentBrief)

    def test_brief_confidence_clamped_to_0_1(
        self,
        specialist: SpecialistAgent,
        mock_router: MagicMock,
    ) -> None:
        """Out-of-range confidence is clamped by AgentBrief validator."""
        mock_router.call.return_value = json.dumps(
            {
                "top3_differential": ["melanoma"],
                "confidence": 1.5,  # out of range
                "reasoning": "Test",
                "cited_cards": [],
                "disagreement_flags": [],
            }
        )
        brief = specialist.generate_brief([])
        assert 0.0 <= brief.confidence <= 1.0

    def test_agent_model_backend_property(
        self, specialist: SpecialistAgent
    ) -> None:
        """The ``model_backend`` property reflects config."""
        assert specialist.model_backend == "google_api"

    def test_agent_init_with_tool_registry(
        self, specialist_config: AgentConfig, mock_router: MagicMock
    ) -> None:
        """Agent can be initialised with an optional tool_registry."""
        mock_registry = MagicMock()
        agent = SpecialistAgent(
            config=specialist_config,
            model_router=mock_router,
            tool_registry=mock_registry,
        )
        assert agent._tool_registry is mock_registry

    def test_call_llm_prepends_system_prompt(
        self,
        specialist: SpecialistAgent,
        mock_router: MagicMock,
    ) -> None:
        """``_call_llm`` automatically prepends the system prompt."""
        specialist._call_llm([{"role": "user", "content": "Hello"}])
        call_args = mock_router.call.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages", call_args[0][1])
        assert messages[0]["role"] == "system"
        assert "specialist" in messages[0]["content"]

    def test_call_llm_does_not_double_system_prompt(
        self,
        specialist: SpecialistAgent,
        mock_router: MagicMock,
    ) -> None:
        """If a system message already exists, don't duplicate it."""
        msgs = [
            {"role": "system", "content": "Custom system prompt."},
            {"role": "user", "content": "Hello"},
        ]
        specialist._call_llm(msgs)
        call_args = mock_router.call.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages", call_args[0][1])
        # Only one system message
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Custom system prompt."

    def test_build_evidence_context_with_no_cards(
        self, specialist: SpecialistAgent
    ) -> None:
        """``_build_evidence_context`` handles empty list."""
        ctx = specialist._build_evidence_context([])
        assert "No evidence" in ctx

    def test_build_evidence_context_truncation(
        self,
        specialist: SpecialistAgent,
        sample_evidence_cards: list[EvidenceCard],
    ) -> None:
        """``_build_evidence_context`` truncates after max_cards."""
        ctx = specialist._build_evidence_context(sample_evidence_cards, max_cards=2)
        assert "omitted" in ctx
