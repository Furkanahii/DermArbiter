"""
DermArbiter Blackboard — Shared State Data Models

Implements the Blackboard architectural pattern for multi-agent dermatological
diagnosis. All agents read from and write to this shared state, enabling
asynchronous evidence accumulation, structured debate, and consensus formation.

Uses Pydantic v2 for strict validation, serialization, and schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Atomic data models
# ---------------------------------------------------------------------------

class ToolOutput(BaseModel):
    """Result produced by a single diagnostic tool invocation."""

    tool_name: str = Field(
        ...,
        description="Canonical name of the tool that produced this output "
                    "(e.g. 'isic_search', 'dermatoscopy_analyzer').",
    )
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured result payload returned by the tool.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tool-reported confidence score in [0, 1].",
    )
    raw_text: str = Field(
        default="",
        description="Unprocessed textual output from the tool, preserved for "
                    "auditability and downstream LLM consumption.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (latency_ms, model_version, etc.).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the tool produced its output.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        """Clamp confidence to [0, 1] even if an upstream tool mis-reports."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))


class EvidenceCard(BaseModel):
    """
    An immutable evidence artifact attached to the blackboard.

    Each card wraps a single ToolOutput and records which agent requested
    the tool execution. Cards are referenced by ``card_id`` in briefs and
    debate turns so that provenance is fully traceable.
    """

    card_id: str = Field(
        default_factory=lambda: f"EC-{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this evidence card.",
    )
    tool_output: ToolOutput = Field(
        ...,
        description="The tool output wrapped by this card.",
    )
    requested_by: str = Field(
        ...,
        description="Role identifier of the agent that requested the tool "
                    "execution (e.g. 'specialist', 'generalist').",
    )

    @field_validator("card_id", mode="before")
    @classmethod
    def _ensure_card_id(cls, v: Any) -> str:
        if not v:
            return f"EC-{uuid.uuid4().hex[:8]}"
        return str(v)


class AgentBrief(BaseModel):
    """
    Structured diagnostic opinion produced by a single agent after reviewing
    the available evidence cards.
    """

    agent_role: str = Field(
        ...,
        description="Role of the agent that authored this brief.",
    )
    top3_differential: list[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
        description="Ordered list of up to 3 (max 5) differential diagnoses.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's overall confidence in the primary (first) diagnosis.",
    )
    reasoning: str = Field(
        default="",
        description="Free-text clinical reasoning that supports the differential.",
    )
    cited_cards: list[str] = Field(
        default_factory=list,
        description="List of EvidenceCard.card_id values referenced in reasoning.",
    )
    disagreement_flags: list[str] = Field(
        default_factory=list,
        description="Flags raised against other agents' conclusions "
                    "(e.g. 'overconfidence:specialist', 'missing_evidence:biopsy').",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))


class DebateTurn(BaseModel):
    """Single turn within the structured multi-agent debate phase."""

    round_num: int = Field(
        ...,
        ge=1,
        description="1-indexed round number within the debate.",
    )
    speaker: str = Field(
        ...,
        description="Role of the agent taking this turn.",
    )
    argument: str = Field(
        default="",
        description="The agent's argument or rebuttal text.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Approximate token count of the argument text.",
    )


# ---------------------------------------------------------------------------
# Top-level blackboard
# ---------------------------------------------------------------------------

class BlackboardState(BaseModel):
    """
    Central shared state for a single diagnostic case.

    Lifecycle:
        1. Created with case_id, image_path, query, and optional patient_context.
        2. Populated with EvidenceCards during the tool-use phase.
        3. Enriched with AgentBriefs during the independent-analysis phase.
        4. Extended with DebateTurns during structured debate.
        5. Finalized with consensus diagnosis, scores, and clinical report.
    """

    # --- Case identity ---
    case_id: str = Field(
        default_factory=lambda: f"CASE-{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this diagnostic case.",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Filesystem or URL path to the primary clinical image.",
    )
    query: str = Field(
        default="",
        description="Free-text clinical query submitted by the user.",
    )
    patient_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured patient demographics and history "
                    "(age, sex, fitzpatrick_type, location, duration, etc.).",
    )

    # --- Evidence layer ---
    evidence_cards: list[EvidenceCard] = Field(
        default_factory=list,
        description="Ordered list of all evidence cards on the blackboard.",
    )

    # --- Agent opinions ---
    briefs: dict[str, AgentBrief] = Field(
        default_factory=dict,
        description="Mapping from agent role to its submitted brief.",
    )

    # --- Debate layer ---
    debate_log: list[DebateTurn] = Field(
        default_factory=list,
        description="Chronological log of debate turns.",
    )
    early_exit: bool = Field(
        default=False,
        description="Whether the moderator triggered an early-exit "
                    "(unanimous agreement before max rounds).",
    )

    # --- Final outputs ---
    final_diagnosis: list[str] = Field(
        default_factory=list,
        description="Consensus-ranked list of diagnoses after debate.",
    )
    consensus_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Moderator-assigned consensus score in [0, 1].",
    )
    dissent_notes: list[str] = Field(
        default_factory=list,
        description="Unresolved disagreements or minority opinions.",
    )
    clinical_report: str = Field(
        default="",
        description="Final synthesized clinical report for the end user.",
    )

    # --- Telemetry ---
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Cumulative token usage across all LLM calls.",
    )
    total_tool_calls: int = Field(
        default=0,
        ge=0,
        description="Total number of tool invocations during the case.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal errors encountered during pipeline execution.",
    )

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def add_evidence_card(self, card: EvidenceCard) -> str:
        """
        Append an evidence card to the blackboard.

        Returns:
            The ``card_id`` of the newly added card.
        """
        self.evidence_cards.append(card)
        self.total_tool_calls += 1
        return card.card_id

    def add_brief(self, brief: AgentBrief) -> None:
        """
        Register an agent's brief.  Overwrites any prior brief from the
        same ``agent_role``.
        """
        self.briefs[brief.agent_role] = brief

    def add_debate_turn(self, turn: DebateTurn) -> None:
        """Append a debate turn and accumulate its token count."""
        self.debate_log.append(turn)
        self.total_tokens += turn.token_count

    def get_brief_summary(self) -> str:
        """
        Return a human-readable summary of all submitted briefs.

        Useful for injecting into moderator / debate prompts so that every
        agent can see the panel's current state at a glance.
        """
        if not self.briefs:
            return "No briefs submitted yet."

        lines: list[str] = []
        for role, brief in self.briefs.items():
            dx_str = ", ".join(brief.top3_differential) or "N/A"
            flags = ", ".join(brief.disagreement_flags) or "none"
            lines.append(
                f"[{role.upper()}] "
                f"Top-3: {dx_str} | "
                f"Confidence: {brief.confidence:.2f} | "
                f"Flags: {flags}"
            )
        return "\n".join(lines)

    def get_evidence_summary(self) -> str:
        """
        Return a compact summary of all evidence cards for prompt injection.
        """
        if not self.evidence_cards:
            return "No evidence cards available."

        lines: list[str] = []
        for card in self.evidence_cards:
            to = card.tool_output
            lines.append(
                f"[{card.card_id}] "
                f"tool={to.tool_name} | "
                f"confidence={to.confidence:.2f} | "
                f"requested_by={card.requested_by} | "
                f"summary={to.raw_text[:120]}..."
                if len(to.raw_text) > 120
                else f"[{card.card_id}] "
                     f"tool={to.tool_name} | "
                     f"confidence={to.confidence:.2f} | "
                     f"requested_by={card.requested_by} | "
                     f"summary={to.raw_text}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the entire blackboard to a plain dict suitable for JSON
        export, logging, or database persistence.

        Uses Pydantic v2's ``model_dump`` with ISO-formatted datetimes.
        """
        return self.model_dump(mode="json")

    @model_validator(mode="after")
    def _validate_consensus_score(self) -> "BlackboardState":
        """Ensure consensus_score stays in [0, 1]."""
        self.consensus_score = max(0.0, min(1.0, self.consensus_score))
        return self
