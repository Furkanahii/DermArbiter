"""
DermArbiter Moderator Agent — Panel Coordinator & Synthesizer

The ``ModeratorAgent`` oversees the diagnostic panel.  It does not
advocate for its own diagnosis but instead:

    • Synthesizes all agents' briefs into a coherent consensus.
    • Mediates debate by identifying genuine disagreements vs. noise.
    • Decides whether an early exit is warranted (unanimous agreement).
    • Produces the final clinical report for the end user.

The moderator has limited tool access — only ``ontology_graph`` (for
terminology alignment) and ``uncertainty_probe`` (to quantify panel
disagreement).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import (
    AgentBrief,
    BlackboardState,
    EvidenceCard,
)

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
# ModeratorAgent
# ---------------------------------------------------------------------------

class ModeratorAgent(BaseAgent):
    """
    Panel moderator that synthesizes, mediates, and produces final reports.

    Behaviour summary:
        - **Tool proposal**: Requests only ``ontology_graph`` and
          ``uncertainty_probe`` — minimal footprint.
        - **Brief generation**: Synthesizes all evidence into a balanced,
          meta-analytic brief rather than advocating a personal diagnosis.
        - **Debate style**: Mediative, not adversarial.  Identifies common
          ground and reframes disagreements constructively.
        - **Extra**: ``should_early_exit`` and ``synthesize_final_report``
          provide moderator-only control-flow decisions.
    """

    _DEFAULT_TOOLS: list[str] = [
        "ontology_graph",
        "uncertainty_probe",
    ]

    # ------------------------------------------------------------------
    # Phase 1 — Tool Proposal
    # ------------------------------------------------------------------

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        """
        Propose moderator-specific tools only.

        The moderator does not request primary diagnostic tools; it uses
        ``ontology_graph`` for terminology alignment and
        ``uncertainty_probe`` to quantify panel disagreement.

        Args:
            case_info: Dict with keys such as ``query``, ``image_path``,
                ``patient_context``.

        Returns:
            Ordered list of tool name strings.
        """
        logger.debug(
            "[%s] Proposing moderator tools for case: %s",
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
        Synthesize all available evidence into a meta-analytic brief.

        Unlike the specialist and generalist, the moderator does not
        advocate for a single diagnosis.  Instead, it identifies the
        consensus differential and flags unresolved disagreements.

        Args:
            evidence_cards: All evidence cards currently on the blackboard.

        Returns:
            A fully populated ``AgentBrief`` reflecting the panel's
            aggregate view.
        """
        evidence_context = self._build_evidence_context(evidence_cards)
        card_ids = [c.card_id for c in evidence_cards]

        prompt = (
            "You are the MODERATOR of a multi-expert dermatological "
            "diagnostic panel.  Your role is NOT to advocate for your own "
            "diagnosis but to synthesize all available evidence into a "
            "balanced meta-analysis.\n\n"
            f"{evidence_context}\n\n"
            "Respond with ONLY a JSON object in this exact schema:\n"
            "```json\n"
            "{\n"
            '  "top3_differential": ["<consensus most likely>", "<second>", "<third>"],\n'
            '  "confidence": 0.XX,\n'
            '  "reasoning": "<synthesis of all evidence, noting agreements and gaps>",\n'
            '  "cited_cards": ["<card_id_1>", "<card_id_2>"],\n'
            '  "disagreement_flags": ["<unresolved issue 1>"]\n'
            "}\n"
            "```\n\n"
            "Guidelines:\n"
            "- Confidence reflects how well the evidence converges on a "
            "single diagnosis.\n"
            "- The top3 differential should reflect panel consensus, not "
            "your personal opinion.\n"
            "- Cite evidence cards that most strongly support each "
            "differential entry.\n"
            "- Disagreement flags should capture genuine unresolved "
            "issues, not nitpicks.\n"
            "- Be impartial — weight evidence quality over agent authority."
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
                return AgentBrief(
                    agent_role=self.role,
                    top3_differential=parsed["top3_differential"][:5],
                    confidence=parsed.get("confidence", 0.5),
                    reasoning=parsed.get("reasoning", ""),
                    cited_cards=cited,
                    disagreement_flags=parsed.get("disagreement_flags", []),
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

        # Fallback brief
        return AgentBrief(
            agent_role=self.role,
            top3_differential=[],
            confidence=0.3,
            reasoning="LLM output could not be parsed.",
            cited_cards=[],
            disagreement_flags=[],
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
        Generate a mediative, consensus-seeking response.

        Unlike other agents the moderator does not attack the opponent.
        Instead it identifies common ground, reframes the disagreement
        constructively, and proposes a path to resolution.

        Args:
            topic: The clinical question or point of contention.
            opponent_brief: The brief from the agent being addressed.

        Returns:
            Free-text mediation string.
        """
        opponent_dx = ", ".join(opponent_brief.top3_differential) or "N/A"
        opponent_flags = ", ".join(opponent_brief.disagreement_flags) or "none"

        prompt = (
            "You are the MODERATOR in a structured diagnostic debate.  "
            "Your role is to mediate, NOT to attack.  Identify common "
            "ground, clarify misunderstandings, and guide the panel toward "
            "consensus.\n\n"
            f"DEBATE TOPIC: {topic}\n\n"
            f"AGENT ({opponent_brief.agent_role}) POSITION:\n"
            f"  Differential: {opponent_dx}\n"
            f"  Confidence: {opponent_brief.confidence:.2f}\n"
            f"  Reasoning: {opponent_brief.reasoning}\n"
            f"  Cited cards: {', '.join(opponent_brief.cited_cards) or 'none'}\n"
            f"  Disagreement flags: {opponent_flags}\n\n"
            "YOUR TASK:\n"
            "1. Acknowledge valid points in the agent's position.\n"
            "2. Identify where this position aligns with other agents.\n"
            "3. Clarify any terminology or framing misunderstandings.\n"
            "4. Suggest how remaining disagreements could be resolved "
            "(e.g. additional evidence, biopsy recommendation).\n"
            "5. Assess whether consensus is forming or diverging.\n"
            "6. Keep the tone constructive and impartial."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            return self._call_llm(messages)
        except Exception as exc:
            logger.error(
                "[%s] Argument generation failed: %s", self.role, exc,
            )
            return (
                f"[{self.role}] Unable to generate mediation response.  "
                f"The panel should proceed with the current evidence."
            )

    # ------------------------------------------------------------------
    # Moderator-only methods
    # ------------------------------------------------------------------

    def should_early_exit(self, briefs: dict[str, AgentBrief]) -> bool:
        """
        Determine whether the panel has reached sufficient consensus to
        skip remaining debate rounds.

        Early exit conditions (ALL must be true):
            1. At least 2 briefs submitted.
            2. All agents agree on the primary (first) diagnosis.
            3. The minimum confidence across all briefs exceeds the
               ``early_exit_threshold`` from agent config (default 0.50).
            4. No agent has any disagreement flags.

        Args:
            briefs: Mapping from agent role to its submitted
                ``AgentBrief``.

        Returns:
            ``True`` if the moderator recommends early exit.
        """
        if len(briefs) < 2:
            logger.debug("[%s] Too few briefs (%d) for early exit.", self.role, len(briefs))
            return False

        # Collect primary diagnoses (first element of top3) and check disagreement flags
        primaries: list[str] = []
        confidences: list[float] = []
        for role, brief in briefs.items():
            if brief.disagreement_flags:
                logger.info(
                    "[%s] Early exit check failed: agent %s has disagreement flags: %s",
                    self.role,
                    role,
                    brief.disagreement_flags,
                )
                return False
            if brief.top3_differential:
                primaries.append(brief.top3_differential[0].strip().lower())
            confidences.append(brief.confidence)

        if not primaries:
            return False

        # Check unanimity on primary diagnosis
        unanimous = len(set(primaries)) == 1

        # Check minimum confidence threshold
        threshold = self._config.extra.get("early_exit_threshold", 0.50)
        above_threshold = all(c >= threshold for c in confidences)

        should_exit = unanimous and above_threshold

        logger.info(
            "[%s] Early exit check — unanimous=%s, min_conf=%.2f, "
            "threshold=%.2f, exit=%s",
            self.role,
            unanimous,
            min(confidences) if confidences else 0.0,
            threshold,
            should_exit,
        )

        return should_exit

    def synthesize_final_report(self, blackboard: BlackboardState) -> str:
        """
        Produce a comprehensive clinical report from the full blackboard.

        The report integrates:
            - Evidence summary
            - Agent briefs and their confidence levels
            - Debate highlights and resolutions
            - Consensus differential diagnosis
            - Unresolved disagreements and recommended next steps

        Args:
            blackboard: The complete ``BlackboardState`` for this case.

        Returns:
            A formatted clinical report string suitable for the end user.
        """
        evidence_summary = blackboard.get_evidence_summary()
        brief_summary = blackboard.get_brief_summary()

        # Build debate excerpt
        debate_lines: list[str] = []
        for turn in blackboard.debate_log[-6:]:  # last 6 turns max
            debate_lines.append(
                f"  Round {turn.round_num} [{turn.speaker}]: "
                f"{turn.argument[:200]}..."
                if len(turn.argument) > 200
                else f"  Round {turn.round_num} [{turn.speaker}]: "
                     f"{turn.argument}"
            )
        debate_excerpt = "\n".join(debate_lines) if debate_lines else "No debate conducted."

        # Collect dissent notes
        dissent = "\n".join(
            f"  - {note}" for note in blackboard.dissent_notes
        ) if blackboard.dissent_notes else "  None recorded."

        prompt = (
            "You are the MODERATOR producing the FINAL CLINICAL REPORT "
            "for a multi-expert dermatological diagnostic case.\n\n"
            "Synthesize the following into a professional, patient-facing "
            "clinical report:\n\n"
            f"=== CASE ===\n"
            f"Case ID: {blackboard.case_id}\n"
            f"Query: {blackboard.query}\n\n"
            f"=== EVIDENCE ===\n{evidence_summary}\n\n"
            f"=== AGENT BRIEFS ===\n{brief_summary}\n\n"
            f"=== DEBATE HIGHLIGHTS ===\n{debate_excerpt}\n\n"
            f"=== DISSENT NOTES ===\n{dissent}\n\n"
            "REPORT FORMAT:\n"
            "1. **Summary**: One-paragraph clinical summary.\n"
            "2. **Differential Diagnosis**: Ranked list with confidence.\n"
            "3. **Evidence Basis**: Key evidence supporting each diagnosis.\n"
            "4. **Areas of Agreement**: Points all agents converged on.\n"
            "5. **Unresolved Issues**: Any remaining disagreements.\n"
            "6. **Recommended Next Steps**: Suggested follow-up "
            "(biopsy, imaging, referral, etc.).\n"
            "7. **Confidence Assessment**: Overall panel confidence and "
            "any caveats.\n\n"
            "Write in clear, professional medical language suitable for "
            "both clinicians and informed patients."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            return self._call_llm(messages)
        except Exception as exc:
            logger.error(
                "[%s] Final report synthesis failed: %s", self.role, exc,
            )
            # Fallback structured report
            dx_line = ", ".join(blackboard.final_diagnosis) or "Undetermined"
            return (
                f"# DermArbiter Clinical Report — {blackboard.case_id}\n\n"
                f"## Summary\n"
                f"Automated report generation encountered an error.  "
                f"The panel's preliminary consensus is presented below.\n\n"
                f"## Differential Diagnosis\n{dx_line}\n\n"
                f"## Consensus Score\n{blackboard.consensus_score:.2f}\n\n"
                f"## Agent Briefs\n{brief_summary}\n\n"
                f"## Dissent Notes\n{dissent}\n\n"
                f"*Report generated with reduced fidelity due to LLM "
                f"failure.  Clinical review is strongly recommended.*"
            )
