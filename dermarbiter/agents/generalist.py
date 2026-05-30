"""
DermArbiter Generalist Agent — Broad Clinical Perspective

The ``GeneralistAgent`` represents a primary-care physician or general
practitioner with solid dermatological knowledge but a broader clinical
lens.  It favours common diagnoses, considers systemic differentials that
a specialist might overlook, and actively monitors for fairness and
demographic bias in tool outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dermarbiter.agents.base_agent import BaseAgent
from dermarbiter.core.blackboard import AgentBrief, EvidenceCard
from dermarbiter.core.utils import extract_json

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# GeneralistAgent
# ---------------------------------------------------------------------------

class GeneralistAgent(BaseAgent):
    """
    Broad-spectrum clinician with balanced diagnostic tool access.

    Behaviour summary:
        - **Tool proposal**: Requests a balanced set including image
          classification, general-purpose VQA, case-based RAG, and a
          fairness probe to guard against demographic bias.
        - **Brief generation**: Favours common diagnoses (Bayesian prior
          weighting), considers systemic differentials, and flags potential
          fairness concerns.
        - **Debate style**: Pragmatic and patient-centred; challenges
          over-specialised diagnoses with epidemiological reasoning.
    """

    _DEFAULT_TOOLS: list[str] = [
        "panderm_classifier",
        "general_vqa",
        "case_rag",
        "fairness_probe",
    ]

    # ------------------------------------------------------------------
    # Phase 1 — Tool Proposal
    # ------------------------------------------------------------------

    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        """
        Propose a generalist-oriented diagnostic tool set.

        Includes a fairness probe by default to ensure equitable
        performance across skin types and demographics.

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
        Synthesize evidence cards into a generalist diagnostic brief.

        The generalist weighs prevalence and epidemiological priors more
        heavily than a specialist, and actively monitors for fairness
        signals in tool outputs.

        Args:
            evidence_cards: All evidence cards currently on the blackboard.

        Returns:
            A fully populated ``AgentBrief``.
        """
        evidence_context = self._build_evidence_context(evidence_cards)
        card_ids = [c.card_id for c in evidence_cards]

        # MedGemma-4B (the configured Generalist backend) is a local model
        # and ignores `response_mime_type` — Gemini's JSON-mode escape hatch
        # doesn't apply here. We instead steer it with a tight one-shot
        # example, explicit "no prose" delimiters, and a JSON-only stop
        # condition the parser can latch onto.

        prompt = (
            "You are a primary-care physician with strong dermatological "
            "training, acting as the GENERALIST agent in a multi-expert "
            "diagnostic panel.\n\n"
            "Review the evidence cards below and produce a diagnostic "
            "opinion that balances specialist tool output with clinical "
            "prevalence and patient context.\n\n"
            "═══ CLASS PRIORS (HAM10000) ═══\n"
            "Apply these as Bayesian priors when evidence is non-specific:\n"
            "  nv=67%, mel=11%, bkl=11%, bcc=5%, akiec=3%, df=1%, vasc=1%\n"
            "Default to 'nv' when uncertain. Don't over-call mel/bkl on "
            "ambiguous pigmented lesions.\n\n"
            f"{evidence_context}\n\n"
            "═══ OUTPUT FORMAT ═══\n"
            "Output ONE valid JSON object and NOTHING ELSE. No markdown, "
            "no code fences, no explanation before or after. Begin your "
            "response with `{` and end with `}`.\n\n"
            "Required keys (use these EXACTLY — do not rename or omit):\n"
            "  top3_differential : list of 3 HAM10000 codes (nv/mel/bkl/bcc/akiec/df/vasc)\n"
            "  confidence        : float 0.0 to 1.0\n"
            "  reasoning         : single string, no newlines inside\n"
            "  cited_cards       : list of evidence-card IDs you reference\n"
            "  disagreement_flags: list of strings (empty if no concerns)\n\n"
            "═══ EXAMPLE (shape only — do NOT copy values) ═══\n"
            '{"top3_differential": ["nv", "bkl", "mel"], '
            '"confidence": 0.72, "reasoning": "Tool outputs EC-12ab and '
            "EC-34cd suggest a benign pigmented lesion with regular borders; "
            'prevalence in adults > 30 favors melanocytic nevus.", '
            '"cited_cards": ["EC-12ab", "EC-34cd"], '
            '"disagreement_flags": []}\n\n'
            "═══ YOUR OUTPUT (JSON object only) ═══\n"
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
                    confidence=parsed.get("confidence", 0.5),
                    reasoning=parsed.get("reasoning", ""),
                    cited_cards=cited,
                    disagreement_flags=parsed.get("disagreement_flags", []),
                )

            logger.warning(
                "[%s] LLM JSON response missing required keys. "
                "Parsed keys: %s | raw_response[:800]: %r",
                self.role, list(parsed.keys()), raw_response[:800],
            )
        except Exception as exc:
            logger.error(
                "[%s] Brief generation failed: %s | raw_response[:800]: %r",
                self.role, exc, locals().get("raw_response", "")[:300],
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
        Generate a pragmatic, prevalence-aware rebuttal.

        The generalist challenges over-specialised diagnoses with
        epidemiological reasoning and patient-centred concerns.

        .. note:: **Image embedding in debate prompts**

           Direct image embedding was considered here but intentionally
           omitted.  Debate arguments operate on inter-agent reasoning
           and structured evidence cards — not raw pixel data.  Image
           features are already captured by upstream tools (PanDerm,
           DermoGPT) and surfaced via ``EvidenceCard.tool_output``.
           Adding raw images to debate prompts would inflate token cost
           without meaningfully improving argument quality.

        Args:
            topic: The clinical question or point of contention.
            opponent_brief: The brief from the agent being challenged.

        Returns:
            Free-text argument string.
        """
        opponent_dx = ", ".join(opponent_brief.top3_differential) or "N/A"
        opponent_flags = ", ".join(opponent_brief.disagreement_flags) or "none"

        prompt = (
            "You are the GENERALIST primary-care physician in a structured "
            "diagnostic debate.  Provide a pragmatic, patient-centred "
            "rebuttal.\n\n"
            f"DEBATE TOPIC: {topic}\n\n"
            f"OPPONENT ({opponent_brief.agent_role}) POSITION:\n"
            f"  Differential: {opponent_dx}\n"
            f"  Confidence: {opponent_brief.confidence:.2f}\n"
            f"  Reasoning: {opponent_brief.reasoning}\n"
            f"  Cited cards: {', '.join(opponent_brief.cited_cards) or 'none'}\n"
            f"  Disagreement flags: {opponent_flags}\n\n"
            "YOUR TASK:\n"
            "1. Consider whether common diagnoses have been overlooked.\n"
            "2. Challenge overconfidence with prevalence data.\n"
            "3. Flag any demographic or fairness concerns.\n"
            "4. Suggest additional workup if the differential is uncertain.\n"
            "5. Keep your rebuttal concise and clinically practical."
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
