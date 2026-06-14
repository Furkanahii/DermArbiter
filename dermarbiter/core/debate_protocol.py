"""
DermArbiter Debate Protocol — Multi-Agent 5-Phase Diagnostic Panel Loop

Implements the clinical debate flow logic:
    Phase 1: Plan & Probe (tool proposals & batch execution)
    Phase 2: Independent Reading (initial diagnostic briefs)
    Phase 3: Reveal & Critique (early exit gating)
    Phase 4: Targeted Debate (structured arguments and rebuttals)
    Phase 5: Synthesis (consensus rankings, dissent notes, and clinical report)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import (
    AgentBrief,
    BlackboardState,
    DebateTurn,
    EvidenceCard,
    ToolOutput as BBToolOutput,
)
from dermarbiter.tools.base_tool import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Plan & Probe
# ---------------------------------------------------------------------------

def plan_probe(
    state: BlackboardState,
    agents: Dict[str, BaseAgent],
    tool_registry: ToolRegistry,
) -> None:
    """
    Phase 1: Plan & Probe
    
    Queries each agent for diagnostic tool proposals, resolves duplicates by prioritizing
    first requester (Specialist -> Generalist -> Skeptic -> Moderator), executes unique 
    tools via ToolRegistry, and appends outputs as EvidenceCards to the Blackboard.
    """
    logger.info("Starting Phase 1: Plan & Probe")
    
    case_info = {
        "case_id": state.case_id,
        "query": state.query,
        "patient_context": state.patient_context,
        "image_path": state.image_path,
    }
    
    # Map proposed tools to the first agent requesting them.
    # Order of priority: Specialist -> Generalist -> Skeptic -> Moderator
    priority_order = ["specialist", "generalist", "skeptic", "moderator"]
    tool_to_requester: Dict[str, str] = {}
    
    for role in priority_order:
        agent = agents.get(role)
        if agent and agent.has_tool_access:
            proposed = agent.propose_tools(case_info)
            # Filter by whitelist if configured
            if hasattr(agent, "_config") and agent._config.allowed_tools:
                proposed = [t for t in proposed if t in agent._config.allowed_tools]
                
            for tool_name in proposed:
                if tool_name in tool_registry and tool_name not in tool_to_requester:
                    tool_to_requester[tool_name] = role
                    
    unique_tools = list(tool_to_requester.keys())
    if not unique_tools:
        logger.warning("No unique diagnostic tools proposed by any agent.")
        return
        
    logger.info("Executing tool batch: %s", unique_tools)
    outputs = tool_registry.run_batch(
        tool_names=unique_tools,
        image_path=state.image_path,
        query=state.query,
    )
    
    # Wrap in EvidenceCards and attach to blackboard
    for out in outputs:
        bb_out = BBToolOutput(
            tool_name=out.tool_name,
            result=out.result,
            confidence=out.confidence,
            raw_text=out.raw_text,
            metadata=out.metadata,
            timestamp=out.timestamp,
        )
        req_by = tool_to_requester.get(out.tool_name, "unknown")
        card = EvidenceCard(
            tool_output=bb_out,
            requested_by=req_by,
        )
        state.add_evidence_card(card)
        
    logger.info("Phase 1 complete. Evidence cards added: %d", len(state.evidence_cards))


# ---------------------------------------------------------------------------
# Phase 2: Independent Reading
# ---------------------------------------------------------------------------

def independent_read(
    state: BlackboardState,
    agents: Dict[str, BaseAgent],
) -> None:
    """
    Phase 2: Independent Reading
    
    Instructs each agent to generate an initial diagnostic brief based on the
    accumulated evidence cards, registers the briefs, and tracks token consumption.
    """
    logger.info("Starting Phase 2: Independent Reading")
    
    for role, agent in agents.items():
        logger.info("Agent [%s] is reviewing evidence and generating brief...", role)
        brief = agent.generate_brief(state.evidence_cards)
        state.add_brief(brief)
        
        # Estimate token usage (prompt context + response brief)
        prompt_tokens = BaseAgent._count_tokens(state.get_evidence_summary()) + 300
        response_tokens = BaseAgent._count_tokens(brief.reasoning) + 100
        state.total_tokens += (prompt_tokens + response_tokens)
        
    logger.info("Phase 2 complete. Briefs submitted for roles: %s", list(state.briefs.keys()))


# ---------------------------------------------------------------------------
# Phase 3: Reveal & Critique
# ---------------------------------------------------------------------------

def reveal_critique(
    state: BlackboardState,
    agents: Dict[str, BaseAgent],
) -> None:
    """
    Phase 3: Reveal & Critique
    
    Unveils the diagnostic briefs and queries the Moderator agent to determine 
    if a consensus exists, triggering an early exit if conditions are satisfied.
    """
    logger.info("Starting Phase 3: Reveal & Critique")
    
    moderator = agents.get("moderator")
    if not moderator:
        logger.warning("Moderator agent missing. Skipping early exit check.")
        return
        
    if hasattr(moderator, "should_early_exit"):
        state.early_exit = moderator.should_early_exit(state.briefs)
        logger.info("Early exit evaluation result: %s", state.early_exit)
    else:
        logger.warning("Moderator agent lacks should_early_exit method.")


# ---------------------------------------------------------------------------
# Phase 4: Targeted Debate
# ---------------------------------------------------------------------------

def _get_opponent_role(speaker: str) -> str:
    """Determine the logical opponent role for a debate speaker."""
    if speaker == "specialist":
        return "generalist"
    elif speaker == "generalist":
        return "specialist"
    else:
        return "specialist"


def targeted_debate(
    state: BlackboardState,
    agents: Dict[str, BaseAgent],
    max_rounds: int = 3,
    max_tokens_per_turn: int = 100,
    global_token_budget: int = 50000,
    turn_order: List[str] | None = None,
) -> None:
    """
    Phase 4: Targeted Debate
    
    Orchestrates sequential rounds of debate between agents. If state.early_exit is 
    True, this phase is skipped. Truncates turn arguments exceeding turn token budget,
    and terminates early if the global token budget is exhausted.
    """
    if state.early_exit:
        logger.info("Phase 4: Targeted Debate skipped due to early exit.")
        return
        
    if turn_order is None:
        turn_order = ["specialist", "generalist", "skeptic"]
        
    logger.info("Starting Phase 4: Targeted Debate (Max Rounds: %d)", max_rounds)
    
    for round_num in range(1, max_rounds + 1):
        logger.info("Debate Round %d / %d", round_num, max_rounds)
        
        for speaker in turn_order:
            # Check global token budget constraint
            if state.total_tokens >= global_token_budget:
                logger.warning(
                    "Global token budget exceeded (%d >= %d). Forcing debate termination.",
                    state.total_tokens,
                    global_token_budget,
                )
                return
                
            agent = agents.get(speaker)
            if not agent:
                continue
                
            opponent_role = _get_opponent_role(speaker)
            opponent_brief = state.briefs.get(opponent_role)
            speaker_brief = state.briefs.get(speaker)
            
            if not opponent_brief:
                logger.warning("Opponent brief missing for speaker [%s]. Skipping turn.", speaker)
                continue
                
            # Construct a dynamic clinical topic based on the differences
            if speaker_brief:
                speaker_dx = speaker_brief.top3_differential[0] if speaker_brief.top3_differential else "None"
                opponent_dx = opponent_brief.top3_differential[0] if opponent_brief.top3_differential else "None"
                if speaker_dx.lower() != opponent_dx.lower():
                    topic = (
                        f"Divergence in primary diagnosis: [{speaker}] proposes '{speaker_dx}' "
                        f"whereas [{opponent_role}] prefers '{opponent_dx}'."
                    )
                else:
                    topic = (
                        f"Consensus on primary diagnosis '{speaker_dx}' but differing confidence levels: "
                        f"[{speaker}] has {speaker_brief.confidence:.2f} whereas [{opponent_role}] has {opponent_brief.confidence:.2f}."
                    )
            else:
                topic = "Clinical review of the patient case and available tool evidence."
                
            logger.info("Agent [%s] is debating on topic: %s", speaker, topic)
            argument = agent.generate_argument(topic, opponent_brief)
            
            # Enforce turn token limit (truncate text if it exceeds limit)
            raw_tokens = BaseAgent._count_tokens(argument)
            if raw_tokens > max_tokens_per_turn:
                # Approximately 3.8 characters per token
                char_limit = int(max_tokens_per_turn * 3.8)
                argument = argument[:char_limit].rstrip() + "..."
                tokens = max_tokens_per_turn
            else:
                tokens = raw_tokens
                
            # Add turn to blackboard (which automatically increments state.total_tokens)
            turn = DebateTurn(
                round_num=round_num,
                speaker=speaker,
                argument=argument,
                token_count=tokens,
            )
            state.add_debate_turn(turn)
            
            # Additional prompt token estimation for the turn generation call
            prompt_tokens_est = BaseAgent._count_tokens(topic) + 200
            state.total_tokens += prompt_tokens_est


# ---------------------------------------------------------------------------
# Phase 5: Synthesis
# ---------------------------------------------------------------------------

def synthesis(
    state: BlackboardState,
    agents: Dict[str, BaseAgent],
    specialist_weight: float = 1.2,
    rank_weights: List[float] | None = None,
    top_k: int = 5,
) -> None:
    """
    Phase 5: Synthesis
    
    Aggregates expert agent opinions to produce consensus diagnoses ranking,
    calculates consensus score, extracts dissent notes, and asks the Moderator
    to compile the final clinical report.

    Args:
        state: The shared blackboard state.
        agents: Dict of role → agent.
        specialist_weight: Multiplier for specialist's contribution (default 1.2).
        rank_weights: Positional weights for rank 1, 2, 3 (default [1.0, 0.6, 0.3]).
        top_k: Maximum number of diagnoses in the final ranking (default 5).
    """
    logger.info("Starting Phase 5: Synthesis")
    
    if rank_weights is None:
        rank_weights = [1.0, 0.6, 0.3]
    
    # 1. Rank consensus diagnoses based on weighted confidence
    scores: Dict[str, float] = {}
    for role, brief in state.briefs.items():
        if role == "moderator":
            continue
        role_weight = specialist_weight if role == "specialist" else 1.0
        for rank, dx in enumerate(brief.top3_differential):
            dx_clean = dx.strip().lower()
            rw = rank_weights[rank] if rank < len(rank_weights) else 0.1
            scores[dx_clean] = scores.get(dx_clean, 0.0) + brief.confidence * rw * role_weight
            
    sorted_dx = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    state.final_diagnosis = [dx for dx, _ in sorted_dx[:top_k]]
    
    # Aggregate ICD-10 and SNOMED-CT mappings from all briefs
    final_icd10 = {}
    final_snomed = {}
    for role in ["skeptic", "generalist", "specialist", "moderator"]:
        brief = state.briefs.get(role)
        if brief:
            if hasattr(brief, "icd10_mappings") and brief.icd10_mappings:
                for dx, code in brief.icd10_mappings.items():
                    final_icd10[dx.strip().lower()] = code
            if hasattr(brief, "snomed_mappings") and brief.snomed_mappings:
                for dx, code in brief.snomed_mappings.items():
                    final_snomed[dx.strip().lower()] = code
                    
    final_dx_set = {dx.lower() for dx in state.final_diagnosis}
    state.final_icd10_mappings = {dx: code for dx, code in final_icd10.items() if dx in final_dx_set}
    state.final_snomed_mappings = {dx: code for dx, code in final_snomed.items() if dx in final_dx_set}
    
    # 2. Compute consensus score based on primary diagnoses agreement
    primaries = [
        brief.top3_differential[0].strip().lower()
        for role, brief in state.briefs.items()
        if role != "moderator" and brief.top3_differential
    ]
    if primaries:
        counts = Counter(primaries)
        most_common_count = counts.most_common(1)[0][1]
        state.consensus_score = most_common_count / len(primaries)
    else:
        state.consensus_score = 0.0
        
    # 3. Compile dissent notes from agent disagreement flags
    dissent = []
    for role, brief in state.briefs.items():
        for flag in brief.disagreement_flags:
            dissent.append(f"{role}: {flag}")
    state.dissent_notes = dissent
    
    # 4. Generate final report via Moderator
    moderator = agents.get("moderator")
    if moderator and hasattr(moderator, "synthesize_final_report"):
        logger.info("Moderator is synthesizing final clinical report...")
        report = moderator.synthesize_final_report(state)
        state.clinical_report = report
        
        # Accumulate report synthesis token counts
        state.total_tokens += BaseAgent._count_tokens(report)
    else:
        logger.warning("Moderator agent missing or lacks synthesize_final_report. Generating fallback.")
        dx_line = ", ".join(state.final_diagnosis) or "Undetermined"
        fallback_report = (
            f"# DermArbiter Clinical Report — {state.case_id}\n\n"
            f"## Differential Diagnosis\n{dx_line}\n\n"
            f"## Consensus Score\n{state.consensus_score:.2f}\n"
        )
        state.clinical_report = fallback_report
        state.total_tokens += BaseAgent._count_tokens(fallback_report)
        
    logger.info("Phase 5 complete. Report generated (%d chars)", len(state.clinical_report))
