"""
DermArbiter Base Agent — Abstract Agent Contract

All agents in the DermArbiter panel (Specialist, Generalist, Skeptic,
Moderator) inherit from ``BaseAgent``.  The class defines:

    • **Abstract methods** that each concrete agent must implement to
      participate in the four pipeline phases (tool proposal, brief
      generation, debate, and synthesis).
    • **Concrete helper methods** for LLM invocation, prompt loading,
      and approximate token counting — shared across all agents.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dermarbiter.core.blackboard import AgentBrief, EvidenceCard

if TYPE_CHECKING:
    from dermarbiter.core.config import AgentConfig
    from dermarbiter.core.model_router import ModelRouter

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for English text.
# GPT / Gemini tokenizers average ~4 chars per token; we use 3.8 for a
# slightly conservative estimate.
_CHARS_PER_TOKEN: float = 3.8


class BaseAgent(ABC):
    """
    Abstract base class for every agent in the DermArbiter diagnostic panel.

    Subclasses must implement:
        - ``propose_tools``      — Phase 1: request diagnostic evidence
        - ``generate_brief``     — Phase 2: synthesize evidence into an opinion
        - ``generate_argument``  — Phase 4: participate in structured debate

    The class provides ready-to-use helpers:
        - ``_call_llm``          — route a prompt through the ModelRouter
        - ``_load_system_prompt``— read and cache a system prompt file
        - ``_count_tokens``      — fast approximate token count
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: AgentConfig,
        model_router: ModelRouter,
        tool_registry: Optional[Any] = None,
    ) -> None:
        """
        Args:
            config: Agent-level configuration (role, model, temperature, …).
            model_router: Shared ``ModelRouter`` instance for LLM dispatch.
            tool_registry: Optional reference to the tool registry so the
                agent can enumerate and invoke diagnostic tools.
        """
        self._config = config
        self._model_router = model_router
        self._tool_registry = tool_registry

        # Cache the system prompt on first access
        self._system_prompt_cache: Optional[str] = None

        logger.info(
            "Initialized agent [%s] — backend=%s, model=%s, tools=%s",
            self.role,
            self.model_backend,
            self._config.model_name,
            self.has_tool_access,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def role(self) -> str:
        """Agent role identifier (e.g. 'specialist')."""
        return self._config.role

    @property
    def model_backend(self) -> str:
        """Name of the LLM backend this agent uses."""
        return self._config.model_backend

    @property
    def has_tool_access(self) -> bool:
        """Whether this agent is allowed to invoke diagnostic tools."""
        return self._config.has_tool_access

    @property
    def system_prompt(self) -> str:
        """
        Lazily load and return the agent's system prompt.

        Falls back to a minimal default if the prompt file is missing.
        """
        if self._system_prompt_cache is None:
            if self._config.system_prompt_path:
                self._system_prompt_cache = self._load_system_prompt(
                    self._config.system_prompt_path
                )
            else:
                self._system_prompt_cache = (
                    f"You are the {self.role} agent in a multi-expert "
                    f"dermatological diagnostic panel."
                )
        return self._system_prompt_cache

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def propose_tools(self, case_info: dict[str, Any]) -> list[str]:
        """
        **Phase 1 — Tool Proposal**

        Given preliminary case information (image description, patient
        context, query), return an ordered list of tool names that should
        be executed to gather diagnostic evidence.

        Args:
            case_info: Dict with keys such as ``query``, ``image_path``,
                ``patient_context``.

        Returns:
            Ordered list of tool name strings (e.g.
            ``['isic_search', 'dermatoscopy_analyzer']``).
        """
        ...

    @abstractmethod
    def generate_brief(
        self,
        evidence_cards: list[EvidenceCard],
    ) -> AgentBrief:
        """
        **Phase 2 — Independent Analysis**

        Review the supplied evidence cards and produce a structured
        ``AgentBrief`` containing a ranked differential diagnosis,
        confidence score, clinical reasoning, and any disagreement flags.

        Args:
            evidence_cards: All evidence cards currently on the blackboard.

        Returns:
            A fully populated ``AgentBrief``.
        """
        ...

    @abstractmethod
    def generate_argument(
        self,
        topic: str,
        opponent_brief: AgentBrief,
    ) -> str:
        """
        **Phase 4 — Structured Debate**

        Generate a debate argument or rebuttal on *topic* in response to
        the *opponent_brief*.  The argument should be evidence-based,
        cite specific ``card_id``s where possible, and stay within the
        configured token budget.

        Args:
            topic: The specific clinical question or point of contention
                being debated in this turn.
            opponent_brief: The brief from the agent whose position is
                being challenged.

        Returns:
            Free-text argument string.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helper methods
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat-completion request through the ``ModelRouter``.

        Automatically prepends the system prompt if the first message
        is not already a system message.

        Args:
            messages: List of ``{"role": "...", "content": "..."}`` dicts.
            temperature: Optional temperature override.
            max_tokens: Optional max-token override.
            json_mode: When True, ask the backend to emit only a JSON
                payload (Gemini sets ``response_mime_type=application/json``;
                local/Groq backends ignore it). Used by brief-generation
                paths that feed extract_json() downstream.

        Returns:
            The LLM response text.
        """
        # Ensure system prompt is present
        if not messages or messages[0].get("role") != "system":
            messages = [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ]

        return self._model_router.call(
            agent_role=self.role,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def _load_system_prompt(self, prompt_path: str) -> str:
        """
        Read a system prompt from a text file.

        The path can be absolute or relative to the project root.
        Falls back to a generic prompt if the file is not found.

        Args:
            prompt_path: Filesystem path to the ``.txt`` prompt file.

        Returns:
            The prompt text.
        """
        path = Path(prompt_path)

        # Try as-is, then relative to the package directory
        candidates = [
            path,
            Path(__file__).resolve().parent.parent / path,
            Path(__file__).resolve().parent.parent / "core" / "prompts" / path.name,
        ]

        for candidate in candidates:
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8").strip()
                logger.debug(
                    "Loaded system prompt for [%s] from %s (%d chars)",
                    self.role,
                    candidate,
                    len(text),
                )
                return text

        logger.warning(
            "System prompt file not found for [%s]: %s. Using fallback.",
            self.role,
            prompt_path,
        )
        return (
            f"You are the {self.role} agent in a multi-expert "
            f"dermatological diagnostic panel.  Provide careful, "
            f"evidence-based clinical reasoning."
        )

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Fast approximate token count.

        Uses a character-based heuristic (÷ 3.8) which is accurate to
        ~±10 % for English medical text.  For precise counts, replace
        with a real tokenizer (e.g. ``tiktoken`` or the model-specific
        ``tokenizer.encode``).

        Args:
            text: Arbitrary string.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0
        return max(1, int(len(text) / _CHARS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Convenience: build a user message with evidence context
    # ------------------------------------------------------------------

    def _build_evidence_context(
        self,
        evidence_cards: list[EvidenceCard],
        max_cards: int = 20,
    ) -> str:
        """
        Format evidence cards into a structured text block suitable for
        injection into an LLM prompt.

        Args:
            evidence_cards: Cards to summarise.
            max_cards: Truncate after this many cards to stay within
                context window limits.

        Returns:
            Formatted multi-line string.
        """
        if not evidence_cards:
            return "No evidence cards available."

        lines: list[str] = ["=== EVIDENCE CARDS ==="]
        for i, card in enumerate(evidence_cards[:max_cards]):
            to = card.tool_output
            lines.append(
                f"\n--- Card {card.card_id} (requested by: {card.requested_by}) ---\n"
                f"Tool: {to.tool_name}\n"
                f"Confidence: {to.confidence:.2f}\n"
                f"Result: {to.raw_text or str(to.result)}\n"
            )

        if len(evidence_cards) > max_cards:
            lines.append(
                f"\n[... {len(evidence_cards) - max_cards} additional cards omitted ...]"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} role={self.role!r} "
            f"backend={self.model_backend!r}>"
        )
