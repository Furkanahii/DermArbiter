"""
DermArbiter Specialist Agent — Deep Dermatological Expertise

The ``SpecialistAgent`` represents a board-certified dermatologist with
access to the full diagnostic tool suite.  It proposes the most
comprehensive set of tools, produces evidence-heavy briefs with specialist
clinical vocabulary, and defends its position with detailed rebuttals
during debate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import AgentBrief, EvidenceCard
from dermarbiter.core.utils import extract_json

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# SpecialistAgent
# ---------------------------------------------------------------------------

class SpecialistAgent(BaseAgent):
    """
    Deep-domain dermatology specialist with comprehensive tool access.

    Behaviour summary:
        - **Tool proposal**: Requests the widest set of specialist tools
          (image classification, lesion annotation, VQA, guideline RAG,
          and uncertainty probing).
        - **Brief generation**: Produces a strongly evidence-backed
          differential diagnosis with high expected confidence.
        - **Debate style**: Authoritative and citation-heavy; pushes back
          on vague generalist or skeptic critiques with specific evidence.
    """

    # Full specialist tool suite — ordered by diagnostic utility.
    _DEFAULT_TOOLS: list[str] = [
        "panderm_classifier",
        "make_annotator",
        "dermogpt_vqa",
        "guideline_rag",
        "uncertainty_probe",
    ]

    # ------------------------------------------------------------------
    # Phase 1 — Tool Proposal
    # ------------------------------------------------------------------

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        """
        Propose the full specialist diagnostic tool suite.

        The specialist always requests its complete tool set.  Tools that
        are not available or not in the agent's allow-list will be filtered
        downstream by the orchestrator.

        Args:
            case_info: Dict with keys such as ``query``, ``image_path``,
                ``patient_context``.

        Returns:
            Ordered list of tool name strings.
        """
        logger.debug(
            "[%s] Proposing tools for case: %s",
            self.role,
            case_info.get("query", "N/A")[:80],
        )
        return list(self._DEFAULT_TOOLS)

    # ------------------------------------------------------------------
    # Phase 2 — Independent Analysis
    # ------------------------------------------------------------------

    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard],
    ) -> AgentBrief:
        """
        Synthesize evidence cards into a specialist-grade diagnostic brief.

        Calls the LLM with a structured JSON-output prompt.  On parse
        failure the method returns a low-confidence fallback brief so that
        the pipeline never crashes.

        Args:
            evidence_cards: All evidence cards currently on the blackboard.

        Returns:
            A fully populated ``AgentBrief``.
        """
        evidence_context = self._build_evidence_context(evidence_cards)
        card_ids = [c.card_id for c in evidence_cards]

        prompt = (
            "You are a board-certified dermatologist acting as the SPECIALIST "
            "agent in a multi-expert diagnostic panel.\n\n"
            "Review the following evidence cards carefully and produce a "
            "structured diagnostic opinion.\n\n"
            f"{evidence_context}\n\n"
            "Respond with ONLY a JSON object in this exact schema:\n"
            "```json\n"
            "{\n"
            '  "top3_differential": ["<most likely>", "<second>", "<third>"],\n'
            '  "confidence": 0.XX,\n'
            '  "reasoning": "<detailed clinical reasoning citing card IDs>",\n'
            '  "cited_cards": ["<card_id_1>", "<card_id_2>"],\n'
            '  "disagreement_flags": []\n'
            "}\n"
            "```\n\n"
            "Guidelines:\n"
            "- Confidence must be a float between 0.0 and 1.0.\n"
            "- Cite specific evidence card IDs (e.g. EC-xxxx) in your reasoning.\n"
            "- Use precise dermatological terminology.\n"
            "- List disagreement flags only if you see contradictory evidence.\n"
            "- Top3 differential should be ordered from most to least likely."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            raw_response = self._call_llm(messages)
            parsed = extract_json(raw_response)

            if parsed and "top3_differential" in parsed:
                # Validate cited_cards against actual card IDs
                cited = [
                    cid for cid in parsed.get("cited_cards", [])
                    if cid in card_ids
                ]
                return AgentBrief(
                    agent_role=self.role,
                    top3_differential=parsed["top3_differential"][:5],
                    confidence=parsed.get("confidence", 0.5),
                    reasoning=parsed.get("reasoning", ""),
                    cited_cards=cited,
                    disagreement_flags=parsed.get("disagreement_flags", []),
                )

            # JSON parsed but missing expected keys
            logger.warning(
                "[%s] LLM JSON response missing required keys. "
                "Parsed keys: %s",
                self.role,
                list(parsed.keys()),
            )
        except Exception as exc:
            logger.error(
                "[%s] Brief generation failed: %s", self.role, exc,
            )

        # Fallback brief
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[],
            confidence=0.3,
            reasoning="LLM output could not be parsed.",
            cited_cards=[],
            disagreement_flags=["parse_error"],
        )

    # ------------------------------------------------------------------
    # Phase 4 — Structured Debate
    # ------------------------------------------------------------------

    def generate_argument(
        self,
        topic: str,
        opponent_brief: AgentBrief,
    ) -> str:
        """
        Generate an authoritative, evidence-based rebuttal.

        The specialist challenges the opponent's differential with specific
        evidence citations and domain expertise.

        Args:
            topic: The clinical question or point of contention.
            opponent_brief: The brief from the agent being challenged.

        Returns:
            Free-text argument string.
        """
        opponent_dx = ", ".join(opponent_brief.top3_differential) or "N/A"
        opponent_flags = ", ".join(opponent_brief.disagreement_flags) or "none"

        prompt = (
            "You are the SPECIALIST dermatologist in a structured diagnostic "
            "debate.  Provide an authoritative, evidence-based rebuttal.\n\n"
            f"DEBATE TOPIC: {topic}\n\n"
            f"OPPONENT ({opponent_brief.agent_role}) POSITION:\n"
            f"  Differential: {opponent_dx}\n"
            f"  Confidence: {opponent_brief.confidence:.2f}\n"
            f"  Reasoning: {opponent_brief.reasoning}\n"
            f"  Cited cards: {', '.join(opponent_brief.cited_cards) or 'none'}\n"
            f"  Disagreement flags: {opponent_flags}\n\n"
            "YOUR TASK:\n"
            "1. Identify specific weaknesses in the opponent's reasoning.\n"
            "2. Cite evidence cards that support your position.\n"
            "3. Use precise dermatological terminology.\n"
            "4. If you agree with any points, acknowledge them.\n"
            "5. Keep your rebuttal concise and clinically focused."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            return self._call_llm(messages)
        except Exception as exc:
            logger.error(
                "[%s] Argument generation failed: %s", self.role, exc,
            )
            return (
                f"[{self.role}] Unable to generate argument due to an "
                f"internal error.  Deferring to evidence on record."
            )
