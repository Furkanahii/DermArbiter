"""
DermArbiter Orchestrator — LangGraph State Machine for Multi-Agent Clinical Debate

Stitches the 5 phases of the debate protocol together using LangGraph.
It manages state transitions, gating checks, and global token/turn budgets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import BlackboardState
from dermarbiter.core.debate_protocol import (
    independent_read,
    plan_probe,
    reveal_critique,
    synthesis,
    targeted_debate,
)
from dermarbiter.tools.base_tool import ToolRegistry

logger = logging.getLogger(__name__)


class DermArbiterOrchestrator:
    """
    Orchestrates the multi-agent diagnostic debate panel using a LangGraph StateGraph.

    Nodes:
        • plan_probe: Phase 1 — Proposal & batch execution of tools.
        • independent_read: Phase 2 — Agents generate independent diagnostic briefs.
        • reveal_critique: Phase 3 — Evaluate briefs for consensus and early-exit.
        • targeted_debate: Phase 4 — Structured sequential argument rounds.
        • synthesis: Phase 5 — Synthesize final diagnoses and clinical report.
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        tool_registry: ToolRegistry,
        max_rounds: int = 3,
        max_tokens_per_turn: int = 100,
        global_token_budget: int = 50000,
        turn_order: List[str] | None = None,
        specialist_weight: float = 1.2,
        rank_weights: List[float] | None = None,
        top_k_diagnoses: int = 5,
    ) -> None:
        """
        Initializes the orchestrator with required agents, tools, and execution limits.

        Args:
            agents: Dict mapping role strings to agents.
            tool_registry: Registry containing the diagnostic tool pool.
            max_rounds: Maximum debate rounds in Phase 4.
            max_tokens_per_turn: Max token count limit per debate turn.
            global_token_budget: Maximum allowed tokens for the entire panel run.
            turn_order: Explicit turn sequence for Phase 4.
            specialist_weight: Weight multiplier for specialist in synthesis.
            rank_weights: Positional weights for rank 1, 2, 3 in synthesis.
            top_k_diagnoses: Max diagnoses in final ranking.
        """
        self.agents = agents
        self.tool_registry = tool_registry
        self.max_rounds = max_rounds
        self.max_tokens_per_turn = max_tokens_per_turn
        self.global_token_budget = global_token_budget
        self.turn_order = turn_order
        self.specialist_weight = specialist_weight
        self.rank_weights = rank_weights
        self.top_k_diagnoses = top_k_diagnoses

        # Compile the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Constructs and compiles the StateGraph workflow."""
        workflow = StateGraph(BlackboardState)

        # Define graph nodes
        workflow.add_node("plan_probe", self._node_plan_probe)
        workflow.add_node("independent_read", self._node_independent_read)
        workflow.add_node("reveal_critique", self._node_reveal_critique)
        workflow.add_node("targeted_debate", self._node_targeted_debate)
        workflow.add_node("synthesis", self._node_synthesis)

        # Entry point
        workflow.set_entry_point("plan_probe")

        # Linear edges
        workflow.add_edge("plan_probe", "independent_read")
        workflow.add_edge("independent_read", "reveal_critique")

        # Conditional routing from reveal_critique based on early-exit gate
        workflow.add_conditional_edges(
            "reveal_critique",
            self._decide_next_phase,
            {
                "synthesis": "synthesis",
                "targeted_debate": "targeted_debate",
            },
        )

        # Complete debate to synthesis transition
        workflow.add_edge("targeted_debate", "synthesis")
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    def _ensure_state_object(self, state: Any) -> BlackboardState:
        """Converts raw state input to BlackboardState if needed."""
        if isinstance(state, BlackboardState):
            return state
        if isinstance(state, dict):
            return BlackboardState(**state)
        raise TypeError(f"Expected state to be dict or BlackboardState, got {type(state)}")

    def _node_plan_probe(self, state: Any) -> BlackboardState:
        """Phase 1 node wrapper."""
        state_obj = self._ensure_state_object(state)
        plan_probe(state_obj, self.agents, self.tool_registry)
        return state_obj

    def _node_independent_read(self, state: Any) -> BlackboardState:
        """Phase 2 node wrapper."""
        state_obj = self._ensure_state_object(state)
        independent_read(state_obj, self.agents)
        return state_obj

    def _node_reveal_critique(self, state: Any) -> BlackboardState:
        """Phase 3 node wrapper."""
        state_obj = self._ensure_state_object(state)
        reveal_critique(state_obj, self.agents)
        return state_obj

    def _node_targeted_debate(self, state: Any) -> BlackboardState:
        """Phase 4 node wrapper."""
        state_obj = self._ensure_state_object(state)
        targeted_debate(
            state_obj,
            self.agents,
            max_rounds=self.max_rounds,
            max_tokens_per_turn=self.max_tokens_per_turn,
            global_token_budget=self.global_token_budget,
            turn_order=self.turn_order,
        )
        return state_obj

    def _node_synthesis(self, state: Any) -> BlackboardState:
        """Phase 5 node wrapper."""
        state_obj = self._ensure_state_object(state)
        synthesis(
            state_obj,
            self.agents,
            specialist_weight=self.specialist_weight,
            rank_weights=self.rank_weights,
            top_k=self.top_k_diagnoses,
        )
        return state_obj

    def _decide_next_phase(self, state: Any) -> str:
        """Conditional routing decision logic."""
        state_obj = self._ensure_state_object(state)
        if state_obj.early_exit:
            logger.info("Consensus met in Phase 3. Routing directly to Synthesis.")
            return "synthesis"
        logger.info("Consensus not met. Routing to Targeted Debate.")
        return "targeted_debate"

    def run(self, initial_state: BlackboardState) -> BlackboardState:
        """
        Executes the compiled LangGraph workflow.

        Args:
            initial_state: The initial BlackboardState containing the patient case.

        Returns:
            The finalized BlackboardState populated with all outputs and clinical report.
        """
        logger.info("Executing DermArbiter Orchestration workflow...")
        result = self.graph.invoke(initial_state)
        return self._ensure_state_object(result)
