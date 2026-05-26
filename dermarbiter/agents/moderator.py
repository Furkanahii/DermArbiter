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

import logging
from typing import Any, Optional

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import (
    AgentBrief,
    BlackboardState,
    EvidenceCard,
)
from dermarbiter.core.utils import extract_json

logger = logging.getLogger(__name__)



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
            raw_response = self._call_llm(messages, json_mode=True)
            parsed = extract_json(raw_response)

            if parsed and "top3_differential" in parsed:
                cited = [
                    cid for cid in parsed.get("cited_cards", [])
                    if cid in card_ids
                ]
                return AgentBrief(
                    agent_role=self.role,
                    top3_differential=parsed["top3_differential"][:5],
                    confidence=self._calibrate_confidence(
                        raw_confidence=parsed.get("confidence", 0.5),
                        n_evidence_cards=len(evidence_cards),
                        n_disagreement_flags=len(
                            parsed.get("disagreement_flags", [])
                        ),
                    ),
                    reasoning=parsed.get("reasoning", ""),
                    cited_cards=cited,
                    disagreement_flags=parsed.get("disagreement_flags", []),
                )

            logger.warning(
                "[%s] LLM JSON response missing required keys. "
                "Parsed keys: %s | raw_response[:300]: %r",
                self.role, list(parsed.keys()), raw_response[:300],
            )
        except Exception as exc:
            logger.error(
                "[%s] Brief generation failed: %s | raw_response[:300]: %r",
                self.role, exc, locals().get("raw_response", "")[:300],
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
                f"[{self.role}] LLM mediation unavailable.  The panel "
                f"should rely on the evidence cards on record and each "
                f"agent's stated reasoning.  Key considerations: "
                f"(1) verify agreement on primary diagnosis across agents, "
                f"(2) review any disagreement flags, "
                f"(3) weight evidence quality over agent authority."
            )

    # ------------------------------------------------------------------
    # Confidence calibration
    # ------------------------------------------------------------------

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        n_evidence_cards: int,
        n_disagreement_flags: int = 0,
    ) -> float:
        """Adjust raw LLM-reported confidence using evidence convergence.

        The LLM's self-reported confidence tends to be poorly calibrated
        (typically overconfident).  This method applies two heuristic
        corrections:

        1. **Evidence penalty**: If fewer than 3 evidence cards are
           available, the confidence is dampened — sparse evidence does
           not warrant high certainty.
        2. **Disagreement penalty**: Each unresolved disagreement flag
           reduces confidence, reflecting genuine panel divergence.

        The result is clamped to [0.1, 0.95] to avoid pathological
        extremes that would distort downstream consensus scoring.

        Args:
            raw_confidence: The confidence value produced by the LLM
                (expected range [0, 1]).
            n_evidence_cards: Number of evidence cards considered.
            n_disagreement_flags: Number of unresolved disagreement flags.

        Returns:
            Calibrated confidence in [0.1, 0.95].
        """
        adjusted = float(raw_confidence)

        # Sparse-evidence penalty: reduce by up to 20% when evidence is thin
        if n_evidence_cards < 3:
            evidence_factor = 0.8 + 0.2 * (n_evidence_cards / 3.0)
            adjusted *= evidence_factor

        # Disagreement penalty: 5% per flag, capped at 25%
        flag_penalty = min(n_disagreement_flags * 0.05, 0.25)
        adjusted -= flag_penalty

        return max(0.1, min(0.95, adjusted))

    # ------------------------------------------------------------------
    # Moderator-only methods
    # ------------------------------------------------------------------

    def should_early_exit(self, briefs: dict[str, AgentBrief]) -> bool:
        """
        Determine whether the panel has reached sufficient consensus to
        skip remaining debate rounds.

        The gating criteria are read from the agent config's ``extra``
        dict (populated from ``agents.yaml``'s ``debate.early_exit``
        section).  Defaults are used when keys are missing.

        Criteria (ALL must pass):
            1. At least ``min_briefs`` (2) briefs submitted.
            2. At least ``min_agreement`` agents agree on the primary dx.
               If ``require_unanimous`` is True, ALL must agree.
            3. Every agent's confidence ≥ ``confidence_floor`` (0.50).
            4. Average confidence ≥ ``early_exit_threshold`` (0.70).
            5. If ``require_no_flags`` is True, no agent may have
               disagreement flags.

        Args:
            briefs: Mapping from agent role to its submitted
                ``AgentBrief``.

        Returns:
            ``True`` if the moderator recommends early exit.
        """
        # --- Read thresholds from config ---
        extra = self._config.extra
        min_agreement = int(extra.get("min_agreement", 2))
        confidence_threshold = float(extra.get("early_exit_threshold", 0.70))
        confidence_floor = float(extra.get("confidence_floor", 0.50))
        require_no_flags = bool(extra.get("require_no_flags", True))
        require_unanimous = bool(extra.get("require_unanimous", False))

        # --- Criterion 1: Minimum briefs ---
        if len(briefs) < 2:
            logger.debug("[%s] Too few briefs (%d) for early exit.", self.role, len(briefs))
            return False

        # --- Collect data ---
        primaries: list[str] = []
        confidences: list[float] = []
        for role, brief in briefs.items():
            # Criterion 5: No disagreement flags
            if require_no_flags and brief.disagreement_flags:
                logger.info(
                    "[%s] Early exit FAILED: agent %s has flags: %s",
                    self.role, role, brief.disagreement_flags,
                )
                return False

            if brief.top3_differential:
                primaries.append(brief.top3_differential[0].strip().lower())
            confidences.append(brief.confidence)

        if not primaries:
            return False

        # --- Criterion 3: Confidence floor ---
        min_conf = min(confidences) if confidences else 0.0
        if min_conf < confidence_floor:
            logger.info(
                "[%s] Early exit FAILED: min confidence %.2f < floor %.2f",
                self.role, min_conf, confidence_floor,
            )
            return False

        # --- Criterion 4: Average confidence ---
        avg_conf = sum(confidences) / len(confidences)
        if avg_conf < confidence_threshold:
            logger.info(
                "[%s] Early exit FAILED: avg confidence %.2f < threshold %.2f",
                self.role, avg_conf, confidence_threshold,
            )
            return False

        # --- Criterion 2: Agreement on primary dx ---
        from collections import Counter
        counts = Counter(primaries)
        most_common_dx, most_common_count = counts.most_common(1)[0]

        if require_unanimous:
            unanimous = len(set(primaries)) == 1
            if not unanimous:
                logger.info(
                    "[%s] Early exit FAILED: unanimity required but primaries=%s",
                    self.role, primaries,
                )
                return False
        else:
            if most_common_count < min_agreement:
                logger.info(
                    "[%s] Early exit FAILED: only %d agents agree on '%s' (need %d)",
                    self.role, most_common_count, most_common_dx, min_agreement,
                )
                return False

        # --- All criteria passed ---
        logger.info(
            "[%s] Early exit PASSED — agreement=%d/%d on '%s', "
            "avg_conf=%.2f, min_conf=%.2f",
            self.role,
            most_common_count,
            len(primaries),
            most_common_dx,
            avg_conf,
            min_conf,
        )
        return True

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
