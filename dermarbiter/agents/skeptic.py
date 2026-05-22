"""
DermArbiter Skeptic Agent — Adversarial Evidence Critic

The ``SkepticAgent`` acts as a contrarian examiner with **no tool access**.
Its role is to stress-test the other agents' conclusions by probing for
logical fallacies, insufficient evidence, overconfidence, anchoring bias,
and alternative diagnoses that may have been prematurely dismissed.

By design the skeptic never invokes tools — it operates purely on the
evidence cards placed on the blackboard by others, ensuring an independent
critical perspective.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import AgentBrief, EvidenceCard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of a JSON object from free-form LLM output.

    Strategy:
        1. Look for a fenced ``json`` code block (```json ... ```).
        2. Fall back to the first ``{ ... }`` substring.
        3. Return an empty dict if nothing parseable is found.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict, or ``{}`` on failure.
    """
    # Strategy 1: fenced code block
    fence_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: raw { ... } block (outermost braces)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# SkepticAgent
# ---------------------------------------------------------------------------

class SkepticAgent(BaseAgent):
    """
    Adversarial critic that reviews evidence without any tool access.

    Behaviour summary:
        - **Tool proposal**: Returns an empty list — no tools by design.
        - **Brief generation**: Critically reviews all evidence cards,
          identifies weaknesses, contradictions, and gaps.  Typically
          reports lower confidence than peer agents.
        - **Debate style**: Most aggressive of all agents.  Directly
          challenges overconfidence, anchoring bias, and insufficient
          evidence.  Forces the panel to defend its conclusions rigorously.
    """

    # ------------------------------------------------------------------
    # Phase 1 — Tool Proposal
    # ------------------------------------------------------------------

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        """
        The skeptic intentionally proposes no tools.

        All evidence must come from other agents.  This ensures the
        skeptic's analysis is fully independent of tool-selection bias.

        Args:
            case_info: Dict with keys such as ``query``, ``image_path``,
                ``patient_context``.

        Returns:
            Empty list — the skeptic has no tool access.
        """
        logger.debug(
            "[%s] No tools proposed (by design) for case: %s",
            self.role,
            case_info.get("query", "N/A")[:80],
        )
        return []

    # ------------------------------------------------------------------
    # Phase 2 — Independent Analysis
    # ------------------------------------------------------------------

    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard],
    ) -> AgentBrief:
        """
        Critically review evidence and produce a skeptic's diagnostic brief.

        The skeptic focuses on:
            - Evidence gaps and missing modalities.
            - Contradictions between tool outputs.
            - Overconfidence signals (single high-confidence tool driving
              the diagnosis).
            - Alternative diagnoses that share morphological features.

        Args:
            evidence_cards: All evidence cards currently on the blackboard.

        Returns:
            A fully populated ``AgentBrief`` with emphasis on
            disagreement flags and conservative confidence.
        """
        evidence_context = self._build_evidence_context(evidence_cards)
        card_ids = [c.card_id for c in evidence_cards]

        prompt = (
            "You are the SKEPTIC agent in a multi-expert dermatological "
            "diagnostic panel.  You have NO tool access and must rely "
            "entirely on the evidence cards produced by other agents.\n\n"
            "Your role is to critically evaluate the evidence:\n"
            "- Identify contradictions between tool outputs.\n"
            "- Flag insufficient or missing evidence.\n"
            "- Challenge overconfident conclusions.\n"
            "- Consider alternative diagnoses that share similar features.\n"
            "- Note any anchoring bias (over-reliance on a single tool).\n\n"
            f"{evidence_context}\n\n"
            "Respond with ONLY a JSON object in this exact schema:\n"
            "```json\n"
            "{\n"
            '  "top3_differential": ["<most likely>", "<second>", "<third>"],\n'
            '  "confidence": 0.XX,\n'
            '  "reasoning": "<critical analysis of the evidence>",\n'
            '  "cited_cards": ["<card_id_1>", "<card_id_2>"],\n'
            '  "disagreement_flags": ["<flag_1>", "<flag_2>"]\n'
            "}\n"
            "```\n\n"
            "Guidelines:\n"
            "- Your confidence should reflect genuine uncertainty — be "
            "conservative.\n"
            "- Disagreement flags are REQUIRED — always find at least one "
            "concern (e.g. 'overconfidence:specialist', "
            "'missing_evidence:dermoscopy', 'anchoring_bias:classifier').\n"
            "- Cite evidence card IDs to ground your critique.\n"
            "- If all evidence aligns, note that but still flag the risk "
            "of groupthink.\n"
            "- Top3 differential should be ordered from most to least likely."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            raw_response = self._call_llm(messages)
            parsed = _extract_json(raw_response)

            if parsed and "top3_differential" in parsed:
                cited = [
                    cid for cid in parsed.get("cited_cards", [])
                    if cid in card_ids
                ]
                # Skeptic's disagreement_flags should never be empty
                flags = parsed.get("disagreement_flags", [])
                if not flags:
                    flags = ["insufficient_critique:auto_generated"]

                return AgentBrief(
                    agent_role=self.role,
                    top3_differential=parsed["top3_differential"][:5],
                    confidence=parsed.get("confidence", 0.4),
                    reasoning=parsed.get("reasoning", ""),
                    cited_cards=cited,
                    disagreement_flags=flags,
                )

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

        # Fallback brief — skeptic defaults to low confidence
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[],
            confidence=0.3,
            reasoning="LLM output could not be parsed.",
            cited_cards=[],
            disagreement_flags=["parse_failure:skeptic_brief"],
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
        Generate an aggressive, evidence-probing rebuttal.

        The skeptic is the most confrontational debater.  It directly
        attacks overconfidence, gaps in reasoning, and potential biases
        while demanding stronger evidence for any diagnosis.

        Args:
            topic: The clinical question or point of contention.
            opponent_brief: The brief from the agent being challenged.

        Returns:
            Free-text argument string.
        """
        opponent_dx = ", ".join(opponent_brief.top3_differential) or "N/A"
        opponent_flags = ", ".join(opponent_brief.disagreement_flags) or "none"

        prompt = (
            "You are the SKEPTIC in a structured diagnostic debate.  Your "
            "role is to be the most rigorous critic on the panel.  Challenge "
            "every weak point aggressively but fairly.\n\n"
            f"DEBATE TOPIC: {topic}\n\n"
            f"OPPONENT ({opponent_brief.agent_role}) POSITION:\n"
            f"  Differential: {opponent_dx}\n"
            f"  Confidence: {opponent_brief.confidence:.2f}\n"
            f"  Reasoning: {opponent_brief.reasoning}\n"
            f"  Cited cards: {', '.join(opponent_brief.cited_cards) or 'none'}\n"
            f"  Disagreement flags: {opponent_flags}\n\n"
            "YOUR TASK:\n"
            "1. Identify ALL logical fallacies and unsupported claims.\n"
            "2. Challenge overconfidence — is the confidence justified by "
            "the evidence quality?\n"
            "3. Propose at least one alternative diagnosis the opponent "
            "may have dismissed.\n"
            "4. Demand specific evidence for any strong claims.\n"
            "5. Flag anchoring bias if the opponent relies too heavily on "
            "a single tool output.\n"
            "6. Be direct, concise, and unsparing — but stay clinical."
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
                f"internal error.  The panel should treat the current "
                f"consensus with additional caution."
            )
