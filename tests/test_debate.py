"""
Unit and integration tests for DermArbiter debate protocol and LangGraph orchestrator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from dermarbiter.core.blackboard import AgentBrief, BlackboardState, DebateTurn, EvidenceCard
from dermarbiter.core.orchestrator import DermArbiterOrchestrator
from tests.mocks.mock_agents import create_mock_agents, MockModerator


@pytest.fixture
def mock_agents_for_debate():
    """Returns a dict of mock agents. We inject the missing methods to MockModerator for integration testing."""
    agents = create_mock_agents()
    moderator = agents["moderator"]

    # Add should_early_exit to moderator dynamically for testing
    def should_early_exit(briefs: dict[str, AgentBrief]) -> bool:
        # Simple rule: early exit if specialist and generalist agree on top-1
        spec = briefs.get("specialist")
        gen = briefs.get("generalist")
        if spec and gen and spec.top3_differential and gen.top3_differential:
            return spec.top3_differential[0].lower() == gen.top3_differential[0].lower()
        return False

    def synthesize_final_report(state: BlackboardState) -> str:
        return f"Synthesized report for {state.case_id}: Consensus score {state.consensus_score:.2f}."

    # Bind methods
    moderator.should_early_exit = should_early_exit
    moderator.synthesize_final_report = synthesize_final_report

    return agents


class TestOrchestratorFlow:
    """Tests covering the orchestrator initialization and full debate execution flow."""

    def test_orchestrator_init(self, mock_agents_for_debate, mock_tool_registry):
        """Verify the orchestrator is properly initialized with default parameters."""
        orchestrator = DermArbiterOrchestrator(
            agents=mock_agents_for_debate,
            tool_registry=mock_tool_registry,
            max_rounds=2,
            max_tokens_per_turn=50,
            global_token_budget=10000,
        )
        assert orchestrator.agents == mock_agents_for_debate
        assert orchestrator.tool_registry == mock_tool_registry
        assert orchestrator.max_rounds == 2
        assert orchestrator.max_tokens_per_turn == 50
        assert orchestrator.global_token_budget == 10000
        assert orchestrator.graph is not None

    def test_run_debate_early_exit(self, mock_agents_for_debate, mock_tool_registry, empty_blackboard):
        """
        Verify that if early exit conditions are met (moderator.should_early_exit returns True),
        Phase 4 (targeted debate) is skipped, and we proceed directly to synthesis.
        """
        orchestrator = DermArbiterOrchestrator(
            agents=mock_agents_for_debate,
            tool_registry=mock_tool_registry,
            max_rounds=3,
        )

        # In mock_agents_for_debate, specialist and generalist both agree on "melanoma" as top-1.
        # This will trigger our dynamic should_early_exit -> True.
        final_state = orchestrator.run(empty_blackboard)

        assert isinstance(final_state, BlackboardState)
        assert final_state.early_exit is True
        # Since early exit was triggered, there should be no debate turns in the log
        assert len(final_state.debate_log) == 0
        # Synthesis should run and produce a report
        assert "Synthesized report" in final_state.clinical_report
        assert len(final_state.final_diagnosis) > 0
        assert final_state.consensus_score > 0.0

    def test_run_debate_no_early_exit(self, mock_agents_for_debate, mock_tool_registry, empty_blackboard):
        """
        Verify that if early exit is false, targeted debate runs for the specified max rounds
        before going to synthesis.
        """
        # Force specialist and generalist to disagree so should_early_exit returns False
        agents = mock_agents_for_debate
        original_gen_brief_method = agents["generalist"].generate_brief

        def disagree_brief(*args, **kwargs):
            brief = original_gen_brief_method(*args, **kwargs)
            brief.top3_differential = ["basal_cell_carcinoma", "melanoma"]
            return brief

        agents["generalist"].generate_brief = disagree_brief

        orchestrator = DermArbiterOrchestrator(
            agents=agents,
            tool_registry=mock_tool_registry,
            max_rounds=2,
            turn_order=["specialist", "generalist"],
        )

        final_state = orchestrator.run(empty_blackboard)

        assert final_state.early_exit is False
        # 2 rounds * 2 speakers = 4 turns
        assert len(final_state.debate_log) == 4
        assert final_state.debate_log[0].round_num == 1
        assert final_state.debate_log[3].round_num == 2
        assert "Synthesized report" in final_state.clinical_report

    def test_global_token_budget_enforcement(self, mock_agents_for_debate, mock_tool_registry, empty_blackboard):
        """
        Verify that if the cumulative token count exceeds the global budget mid-debate,
        Phase 4 halts immediately.
        """
        agents = mock_agents_for_debate
        # Disagree to ensure we enter debate phase
        original_gen_brief_method = agents["generalist"].generate_brief

        def disagree_brief(*args, **kwargs):
            brief = original_gen_brief_method(*args, **kwargs)
            brief.top3_differential = ["basal_cell_carcinoma"]
            return brief

        agents["generalist"].generate_brief = disagree_brief

        # Set a very low global token budget (e.g. 500 tokens).
        # Phase 2 (independent read) adds some estimated tokens (typically ~400+ per agent).
        # We start Phase 4 already near or over budget, so it should abort debate turns immediately.
        orchestrator = DermArbiterOrchestrator(
            agents=agents,
            tool_registry=mock_tool_registry,
            max_rounds=3,
            global_token_budget=500,
        )

        final_state = orchestrator.run(empty_blackboard)
        # Debate should be cut short (likely 0 or very few turns)
        assert len(final_state.debate_log) < 9  # Max possible turns is 9 (3 rounds * 3 speakers)

    def test_turn_token_limit_truncation(self, mock_agents_for_debate, mock_tool_registry, empty_blackboard):
        """
        Verify that debate turn arguments exceeding the turn-level token budget are truncated.
        """
        agents = mock_agents_for_debate
        # Disagree to enter debate
        original_gen_brief = agents["generalist"].generate_brief

        def disagree_brief(*args, **kwargs):
            brief = original_gen_brief(*args, **kwargs)
            brief.top3_differential = ["basal_cell_carcinoma"]
            return brief

        agents["generalist"].generate_brief = disagree_brief

        # Specialist generate_argument returns ~30 tokens.
        # Let's set a very tiny turn limit (e.g., 5 tokens) and check if it gets truncated with trailing "..."
        orchestrator = DermArbiterOrchestrator(
            agents=agents,
            tool_registry=mock_tool_registry,
            max_rounds=1,
            max_tokens_per_turn=5,
            turn_order=["specialist"],
        )

        final_state = orchestrator.run(empty_blackboard)
        assert len(final_state.debate_log) == 1
        argument = final_state.debate_log[0].argument
        assert argument.endswith("...")
        assert len(argument) < 5 * 4 + 5  # Char limit estimate: 5 tokens * 3.8 chars/token + safety margin

    def test_run_returns_blackboard_state(self, mock_agents_for_debate, mock_tool_registry, empty_blackboard):
        """Ensure the orchestrator run method returns a properly typed BlackboardState."""
        orchestrator = DermArbiterOrchestrator(
            agents=mock_agents_for_debate,
            tool_registry=mock_tool_registry,
        )
        result = orchestrator.run(empty_blackboard)
        assert isinstance(result, BlackboardState)
