"""
DermArbiter Model Router — Multi-Backend LLM Dispatch

Routes LLM calls to the appropriate backend based on per-agent configuration:
    • Google Gemini API  (via langchain_google_genai)
    • Local HuggingFace  (transformers with BitsAndBytes quantization)
    • Groq Cloud API     (groq SDK)

Includes automatic fallback: if the primary backend fails, the router
retries on the next configured backend before raising.
"""

from __future__ import annotations

import logging
import os
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

        # Forward local_hf-specific settings from agent config
        local_kwargs: dict[str, Any] = {
            "device": getattr(agent_cfg, "device", "cpu"),
            "quantization": getattr(agent_cfg, "quantization", None),
        }
        local_kwargs.update(kwargs)

        # Try primary backend, then fallbacks
        errors: list[str] = []
        backends_to_try = [backend] + [
            b for b in _FALLBACK_ORDER if b != backend
        ]

        for be in backends_to_try:
            if be not in self._initialized_backends:
                continue
            try:
                return self._dispatch(be, messages, model, temp, max_tok, **local_kwargs)
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
        json_mode = bool(kwargs.get("json_mode", False))
        if backend == ModelBackend.GOOGLE_API:
            return self._call_gemini(messages, model, temperature, max_tokens,
                                      json_mode=json_mode)
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
        json_mode: bool = False,
    ) -> str:
        """
        Call Google Gemini via ``langchain_google_genai.ChatGoogleGenerativeAI``.

        Caches the LLM object per (model, temperature, json_mode) to avoid
        repeated initialization overhead. ``json_mode=True`` sets
        ``response_mime_type='application/json'`` on the Gemini request so
        the model is required to emit a parseable JSON payload (no
        markdown commentary, no fenced code blocks) — the agent brief
        path uses this so extract_json() never has to guess.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        cache_key = f"{model}:{temperature}:{int(json_mode)}"
        if cache_key not in self._gemini_llm_cache:
            init_kwargs: dict[str, Any] = {
                "model": model,
                "google_api_key": self._config.google_api_key,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if json_mode:
                # langchain_google_genai forwards model_kwargs to the
                # underlying google-generativeai client's generation_config.
                init_kwargs["model_kwargs"] = {
                    "response_mime_type": "application/json",
                }
            self._gemini_llm_cache[cache_key] = ChatGoogleGenerativeAI(**init_kwargs)
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
    # Local HuggingFace (Transformers)
    # ------------------------------------------------------------------

    def _call_local(
        self,
        messages: list[dict[str, str]],
        model: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
    ) -> str:
        """
        Run inference on a local HuggingFace Transformers model.

        Supports ``4bit`` and ``8bit`` quantization via bitsandbytes when
        running on CUDA.  Models and tokenizers are cached in
        ``_local_model_cache`` keyed by ``(model, quantization)`` so
        subsequent calls skip the expensive load.

        Args:
            messages: Chat-style ``[{"role": ..., "content": ...}]`` dicts.
            model: HuggingFace model identifier (e.g. ``Qwen/Qwen3-8B-Instruct``).
            device: Target device (``"cpu"`` or ``"cuda"``).
            quantization: ``"4bit"`` / ``"8bit"`` / ``None``.

        Returns:
            The generated response text.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_key = f"{model}:{quantization}"

        # Lazily initialize the model cache
        if not hasattr(self, "_local_model_cache"):
            self._local_model_cache: dict[str, tuple[Any, Any]] = {}

        if cache_key not in self._local_model_cache:
            logger.info(
                "Loading local model %s (device=%s, quant=%s) ...",
                model, device, quantization,
            )
            hf_token = os.environ.get("HF_TOKEN")

            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "token": hf_token,
            }

            if quantization in ("4bit", "4") and device != "cpu":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
            elif quantization in ("8bit", "8") and device != "cpu":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = "auto" if device == "cuda" else {"": device}

            tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
                token=hf_token,
            )
            lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
            lm.eval()
            self._local_model_cache[cache_key] = (tokenizer, lm)
            logger.info("Local model %s loaded successfully.", model)

        tokenizer, lm = self._local_model_cache[cache_key]

        # Apply chat template (most modern models support this)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: manual concatenation if no chat template defined
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|{role}|>\n{content}\n"
            prompt += "<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        # Move inputs to model device
        model_device = next(lm.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        start = time.monotonic()
        with torch.inference_mode():
            output_ids = lm.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )
        elapsed = time.monotonic() - start
        logger.debug("Local inference (%s) completed in %.2fs", model, elapsed)

        # Decode only the generated tokens (skip the prompt)
        answer = tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        return answer

    # ------------------------------------------------------------------
    # Groq Cloud API
    # ------------------------------------------------------------------

    def _call_groq(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call the Groq Cloud API for fast inference on hosted open-source models.

        Uses the ``groq`` Python SDK to call ``chat.completions.create()``.
        Requires ``GROQ_API_KEY`` to be set in the environment or config.

        Args:
            messages: Chat-style ``[{"role": ..., "content": ...}]`` dicts.
            model: Groq model identifier (e.g. ``llama-3.3-70b-versatile``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The LLM's response text.
        """
        import groq as groq_sdk

        if not hasattr(self, "_groq_client"):
            self._groq_client = groq_sdk.Groq(
                api_key=self._config.groq_api_key,
            )

        start = time.monotonic()
        response = self._groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.monotonic() - start
        logger.debug(
            "Groq call (%s, T=%.2f) completed in %.2fs",
            model, temperature, elapsed,
        )

        return response.choices[0].message.content
