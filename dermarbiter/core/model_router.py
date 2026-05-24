"""
DermArbiter Model Router — Multi-Backend LLM Dispatch

Routes LLM calls to the appropriate backend based on per-agent configuration:
    • Google Gemini API  (via langchain_google_genai)
    • Local HuggingFace  (placeholder — transformers / vLLM)
    • Groq Cloud API     (placeholder — groq SDK)

Includes automatic fallback: if the primary backend fails, the router
retries on the next configured backend before raising.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Optional

from dermarbiter.core.config import AgentConfig, DermArbiterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class ModelBackend(str, Enum):
    """Supported LLM inference backends."""

    GOOGLE_API = "google_api"
    LOCAL_HF = "local_hf"
    GROQ_API = "groq_api"


# ---------------------------------------------------------------------------
# Cost table (USD per 1 K tokens, approximate)
# ---------------------------------------------------------------------------

_COST_PER_1K_TOKENS: dict[str, float] = {
    # Google
    "gemini-2.0-flash": 0.00010,
    "gemini-2.0-flash-lite": 0.00005,
    "gemini-2.5-pro": 0.00125,
    "gemini-2.5-flash": 0.00015,
    # Groq (hosted open-source)
    "llama-3.3-70b-versatile": 0.00059,
    "llama-3.1-8b-instant": 0.00005,
    "mixtral-8x7b-32768": 0.00024,
    # Local models are "free" in dollar terms
    "local": 0.0,
}

# Fallback order when a backend is unavailable
_FALLBACK_ORDER: list[ModelBackend] = [
    ModelBackend.GOOGLE_API,
    ModelBackend.GROQ_API,
    ModelBackend.LOCAL_HF,
]


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Dispatches LLM inference requests to the correct backend based on
    agent-level configuration.  Provides transparent fallback and cost
    estimation.

    Usage::

        router = ModelRouter(config)
        reply = router.call("specialist", messages=[
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
        ])
    """

    def __init__(self, config: DermArbiterConfig) -> None:
        self._config = config
        self._agent_configs: dict[str, AgentConfig] = config.agents
        self._gemini_llm_cache: dict[str, Any] = {}
        self._initialized_backends: set[ModelBackend] = set()

        # Eagerly validate that at least one backend is plausible
        self._init_backends()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_backends(self) -> None:
        """Pre-check which backends can be initialized."""
        # Google API
        if self._config.google_api_key:
            try:
                import google.generativeai  # noqa: F401
                self._initialized_backends.add(ModelBackend.GOOGLE_API)
                logger.info("Google Gemini API backend available.")
            except ImportError:
                logger.warning(
                    "langchain_google_genai / google-generativeai not installed. "
                    "Google API backend unavailable."
                )
        else:
            logger.info("GOOGLE_API_KEY not set — Google API backend disabled.")

        # Groq API
        if self._config.groq_api_key:
            try:
                import groq  # noqa: F401
                self._initialized_backends.add(ModelBackend.GROQ_API)
                logger.info("Groq API backend available.")
            except ImportError:
                logger.warning("groq SDK not installed. Groq backend unavailable.")

        # Local HF (always "available" as a placeholder)
        self._initialized_backends.add(ModelBackend.LOCAL_HF)

        if not self._initialized_backends:
            logger.error("No LLM backends could be initialized!")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        agent_role: str,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Route an LLM call for *agent_role* to the configured backend.

        Args:
            agent_role: Role identifier matching a key in ``config.agents``.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Override the agent's default temperature.
            max_tokens: Override the agent's default max_output_tokens.

        Returns:
            The LLM's response text.

        Raises:
            RuntimeError: If all backends fail.
        """
        agent_cfg = self._agent_configs.get(agent_role)
        if agent_cfg is None:
            # Use global defaults
            agent_cfg = AgentConfig(
                role=agent_role,
                model_backend=ModelBackend.GOOGLE_API.value,
                model_name=self._config.default_model,
                temperature=self._config.default_temperature,
            )

        backend = ModelBackend(agent_cfg.model_backend)
        temp = temperature if temperature is not None else agent_cfg.temperature
        max_tok = max_tokens if max_tokens is not None else agent_cfg.max_output_tokens
        model = agent_cfg.model_name

        # Try primary backend, then fallbacks
        errors: list[str] = []
        backends_to_try = [backend] + [
            b for b in _FALLBACK_ORDER if b != backend
        ]

        for be in backends_to_try:
            if be not in self._initialized_backends:
                continue
            try:
                return self._dispatch(be, messages, model, temp, max_tok, **kwargs)
            except Exception as exc:
                msg = f"Backend {be.value} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        raise RuntimeError(
            f"All LLM backends failed for agent '{agent_role}'. "
            f"Errors: {'; '.join(errors)}"
        )

    def get_cost_estimate(self, agent_role: str) -> float:
        """
        Estimate the USD cost per 1 K tokens for the given agent's
        configured model.

        Returns 0.0 if the model is not in the cost table.
        """
        agent_cfg = self._agent_configs.get(agent_role)
        if agent_cfg is None:
            model = self._config.default_model
        else:
            model = agent_cfg.model_name

        return _COST_PER_1K_TOKENS.get(model, 0.0)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        backend: ModelBackend,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        if backend == ModelBackend.GOOGLE_API:
            return self._call_gemini(messages, model, temperature, max_tokens)
        elif backend == ModelBackend.LOCAL_HF:
            device = kwargs.get("device", "cpu")
            quantization = kwargs.get("quantization")
            return self._call_local(messages, model, device, quantization)
        elif backend == ModelBackend.GROQ_API:
            return self._call_groq(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------
    # Google Gemini (via langchain_google_genai)
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call Google Gemini via ``langchain_google_genai.ChatGoogleGenerativeAI``.

        Caches the LLM object per (model, temperature) to avoid repeated
        initialization overhead.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        cache_key = f"{model}:{temperature}"
        if cache_key not in self._gemini_llm_cache:
            self._gemini_llm_cache[cache_key] = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self._config.google_api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        llm = self._gemini_llm_cache[cache_key]

        # Convert dict messages → LangChain message objects
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        start = time.monotonic()
        response = llm.invoke(lc_messages)
        elapsed = time.monotonic() - start
        logger.debug(
            "Gemini call (%s, T=%.2f) completed in %.2fs",
            model,
            temperature,
            elapsed,
        )

        return response.content if hasattr(response, "content") else str(response)

    # ------------------------------------------------------------------
    # Local HuggingFace (placeholder)
    # ------------------------------------------------------------------

    def _call_local(
        self,
        messages: list[dict[str, str]],
        model: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
    ) -> str:
        """
        Placeholder for local HuggingFace Transformers / vLLM inference.

        In production this would:
            1. Load or retrieve a cached ``AutoModelForCausalLM``.
            2. Apply quantization (bitsandbytes 4-bit / 8-bit).
            3. Tokenize the chat template.
            4. Generate and decode.

        Raises:
            NotImplementedError: Always, until a local model is configured.
        """
        logger.warning(
            "Local HF inference requested (model=%s, device=%s, quant=%s) "
            "but the backend is not yet implemented.",
            model,
            device,
            quantization,
        )
        raise NotImplementedError(
            f"Local HuggingFace inference is not yet implemented. "
            f"Model={model}, device={device}, quantization={quantization}. "
            f"Please configure a remote backend (google_api or groq_api) "
            f"or implement _call_local()."
        )

    # ------------------------------------------------------------------
    # Groq Cloud (placeholder)
    # ------------------------------------------------------------------

    def _call_groq(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Placeholder for Groq Cloud API inference.

        In production this would:
            1. Initialize the ``groq.Groq`` client with the API key.
            2. Call ``client.chat.completions.create()``.
            3. Return the first choice's message content.

        Raises:
            NotImplementedError: Always, until Groq integration is wired up.
        """
        logger.warning(
            "Groq API inference requested (model=%s) but the backend is "
            "not yet implemented.",
            model,
        )
        raise NotImplementedError(
            f"Groq Cloud inference is not yet implemented. "
            f"Model={model}. Install the 'groq' package and set GROQ_API_KEY, "
            f"then implement _call_groq()."
        )
