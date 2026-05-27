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
import random
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

    # Default retry parameters for transient API failures (rate limits,
    # network timeouts).  Overridable via config.extra["router_*"] keys.
    _DEFAULT_MAX_RETRIES: int = 3
    _DEFAULT_BASE_DELAY_S: float = 1.0

    def __init__(self, config: DermArbiterConfig) -> None:
        self._config = config
        self._agent_configs: dict[str, AgentConfig] = config.agents
        self._gemini_llm_cache: dict[str, Any] = {}
        self._initialized_backends: set[ModelBackend] = set()

        # Retry settings — configurable via top-level config extra keys
        # (e.g. ``router_max_retries: 5`` in default.yaml).
        self._max_retries: int = self._DEFAULT_MAX_RETRIES
        self._base_delay_s: float = self._DEFAULT_BASE_DELAY_S

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

        # Try primary backend, then fallbacks.
        # Model IDs are backend-specific (Qwen/Qwen3-8B only resolves on
        # HuggingFace, gemini-2.5-flash only on Google API). When a local_hf
        # model fails to load, falling back to google_api with the SAME
        # model id sent a request to
        # ``generativelanguage.googleapis.com/.../Qwen/Qwen3-8B`` → 404
        # NotFound. Per-backend fallback model: use the agent's primary
        # model for its own backend, and ``cfg.default_model`` for any
        # cross-backend fallback to Gemini.
        errors: list[str] = []
        backends_to_try = [backend] + [
            b for b in _FALLBACK_ORDER if b != backend
        ]

        for be in backends_to_try:
            if be not in self._initialized_backends:
                continue
            # If fallback backend is incompatible with the primary's model
            # id (e.g. cross-family), substitute a backend-appropriate model.
            be_model = model
            if be != backend:
                if be == ModelBackend.GOOGLE_API:
                    be_model = self._config.default_model
                elif be == ModelBackend.GROQ_API:
                    # Conservative Groq default; bail if no Groq pref is set.
                    be_model = getattr(self._config, "default_groq_model",
                                       "llama-3.3-70b-versatile")
                elif be == ModelBackend.LOCAL_HF:
                    # No safe local fallback for an arbitrary Gemini model.
                    msg = (f"Skipping cross-backend fallback to local_hf "
                           f"for model {model!r} (no compatible local).")
                    logger.info(msg)
                    errors.append(msg)
                    continue
                logger.info(
                    "Fallback %s → %s: substituting model %r for %r",
                    backend.value, be.value, be_model, model,
                )
            try:
                return self._call_with_retry(
                    be, messages, be_model, temp, max_tok, **local_kwargs,
                )
            except Exception as exc:
                msg = f"Backend {be.value} ({be_model}) failed after retries: {exc}"
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
    # Retry logic
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        backend: ModelBackend,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Call ``_dispatch`` with exponential-backoff retry.

        Retries up to ``self._max_retries`` times on transient failures
        (rate-limit 429s, network timeouts, server 5xx errors).  Each
        retry doubles the wait time with jitter to avoid thundering-herd
        effects across concurrent agents.

        Non-retryable errors (e.g. ``ValueError``, ``ImportError``) are
        raised immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._dispatch(
                    backend, messages, model, temperature, max_tokens, **kwargs,
                )
            except (ValueError, ImportError, TypeError):
                # Not transient — reraise immediately
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._base_delay_s * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.25)
                    total_delay = delay + jitter
                    logger.warning(
                        "Backend %s attempt %d/%d failed: %s — "
                        "retrying in %.1fs",
                        backend.value,
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        total_delay,
                    )
                    time.sleep(total_delay)

        # All retries exhausted
        raise last_exc  # type: ignore[misc]

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

        cache_key = f"{model}:{temperature}"
        if cache_key not in self._gemini_llm_cache:
            self._gemini_llm_cache[cache_key] = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self._config.google_api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        llm = self._gemini_llm_cache[cache_key]

        # JSON mode: bind per-call instead of via model_kwargs (which langchain
        # warns about and silently routes to an inapplicable slot). The .bind()
        # path sets generation_config.response_mime_type on the actual
        # google-generativeai request — verified by langchain_google_genai's
        # _generate() path. Effect: Gemini's API is *required* to emit a JSON
        # payload (no markdown fences, no preamble), so extract_json never has
        # to guess.
        if json_mode:
            llm = llm.bind(response_mime_type="application/json")

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

    # Model-type registry — `model_type` strings reported by AutoConfig for
    # multimodal architectures we want to route through the image-text-to-text
    # API. Detection is primarily via ``hasattr(cfg, "vision_config")`` so
    # this list is a backstop only.
    _MULTIMODAL_MODEL_TYPES = frozenset({
        "gemma3", "gemma3_text",
        "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
        "llava", "llava_next", "llava_onevision",
        "idefics", "idefics2", "idefics3",
        "internvl_chat", "phi3_v", "fuyu", "molmo", "paligemma",
    })

    def _is_multimodal(self, model: str, hf_token: Optional[str]) -> bool:
        """Decide whether ``model`` should be loaded as image-text-to-text.

        Prefers a generic structural check (``cfg.vision_config`` present)
        so future multimodal architectures Just Work; falls back to a
        registry of known model_type strings.
        """
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(
                model, trust_remote_code=True, token=hf_token,
            )
        except Exception as exc:
            logger.warning(
                "AutoConfig probe failed for %s — assuming text-only. (%s)",
                model, exc,
            )
            return False

        if hasattr(cfg, "vision_config") and cfg.vision_config is not None:
            return True
        return getattr(cfg, "model_type", "") in self._MULTIMODAL_MODEL_TYPES

    def _call_local(
        self,
        messages: list[dict[str, str]],
        model: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
    ) -> str:
        """Run inference on a local HuggingFace Transformers model.

        Auto-detects multimodal models (e.g. MedGemma-4B-it / Gemma3,
        Qwen-VL family) and routes them to the image-text-to-text API
        which respects the multimodal chat template. Text-only models
        (Qwen3-8B, Llama, etc.) keep the existing AutoModelForCausalLM
        path. Both paths cache by ``(model, quantization, kind)`` so
        the heavy load happens only once per agent.

        Args:
            messages: Chat-style ``[{"role": ..., "content": ...}]`` dicts.
            model: HuggingFace model identifier.
            device: ``"cpu"`` / ``"cuda"`` / ``"auto"``.
            quantization: ``"4bit"`` / ``"8bit"`` / ``None``.
        """
        # Lazily initialize the model cache
        if not hasattr(self, "_local_model_cache"):
            self._local_model_cache: dict[str, tuple[Any, Any]] = {}

        hf_token = os.environ.get("HF_TOKEN")

        if self._is_multimodal(model, hf_token):
            return self._call_local_multimodal(
                messages, model, device, quantization, hf_token,
            )
        return self._call_local_text(
            messages, model, device, quantization, hf_token,
        )

    # ----- shared bnb config -----

    def _purge_cuda_cache(self) -> None:
        """Free up VRAM aggressively before loading a heavy model.

        T4 (16 GB) is borderline with two 4-bit local LLMs after Phase 1's
        tool churn. Even if the previous tool called ``.unload()``,
        PyTorch's allocator may still hold the freed blocks in its cache;
        the next ``from_pretrained`` then sees only the *currently allocated*
        free pool and trips bitsandbytes' "Some modules are dispatched on
        the CPU or the disk" error during quantization.
        """
        import gc
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("CUDA cache purge skipped: %s", exc)

    def _local_load_kwargs(
        self, device: str, quantization: Optional[str], hf_token: Optional[str],
    ) -> dict[str, Any]:
        """Common ``from_pretrained(**kwargs)`` for text + multimodal paths.

        CPU offload tradeoff: enabling ``llm_int8_enable_fp32_cpu_offload``
        let the T4 (16 GB) load models that didn't fit purely on GPU, but
        the offloaded layers ran 5-10× slower on CPU — a 50-case test
        averaged 50 min/case instead of the expected 3 min. With A100
        (40 GB) or L4 (24 GB) there's no need for CPU spill; the model
        fits entirely on GPU and inference is fast. We auto-detect: if
        usable VRAM ≥ 20 GB, keep everything on GPU. If less, fall back
        to CPU offload as a last resort — but the cross-backend fallback
        chain in ``call()`` will route to Gemini before that path is
        exercised on cramped GPUs.
        """
        import torch

        # Detect available GPU VRAM (header decides offload policy).
        offload_to_cpu = False
        try:
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                total_gb = total_mem / (1024 ** 3)
                offload_to_cpu = total_gb < 20.0  # T4 needs offload, A100/L4 don't
                logger.info(
                    "GPU detected: %.1f GB total — CPU offload %s",
                    total_gb,
                    "ENABLED (small VRAM)" if offload_to_cpu else "DISABLED (ample VRAM)",
                )
        except Exception:
            pass

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "token": hf_token,
        }
        if quantization in ("4bit", "4") and device != "cpu":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=offload_to_cpu,
            )
            kwargs["device_map"] = "auto"
        elif quantization in ("8bit", "8") and device != "cpu":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=offload_to_cpu,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "auto" if device == "cuda" else {"": device}
        return kwargs

    # ----- text-only path (Qwen3-8B, Llama, …) -----

    def _call_local_text(
        self,
        messages: list[dict[str, str]],
        model: str,
        device: str,
        quantization: Optional[str],
        hf_token: Optional[str],
    ) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_key = f"{model}:{quantization}:text"

        if cache_key not in self._local_model_cache:
            logger.info(
                "Loading local TEXT model %s (device=%s, quant=%s)…",
                model, device, quantization,
            )
            self._purge_cuda_cache()   # reclaim VRAM from any prior tool
            tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True, token=hf_token,
            )
            lm = AutoModelForCausalLM.from_pretrained(
                model, **self._local_load_kwargs(device, quantization, hf_token),
            )
            lm.eval()
            self._local_model_cache[cache_key] = (tokenizer, lm)
            logger.info("Local TEXT model %s loaded.", model)

        tokenizer, lm = self._local_model_cache[cache_key]

        # Apply chat template (most modern models ship one).
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: rudimentary <|role|> concatenation.
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|{role}|>\n{content}\n"
            prompt += "<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        model_device = next(lm.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        start = time.monotonic()
        with torch.inference_mode():
            output_ids = lm.generate(
                **inputs, max_new_tokens=4096, do_sample=False,
            )
        elapsed = time.monotonic() - start
        logger.debug("Local TEXT inference (%s) completed in %.2fs",
                     model, elapsed)

        return tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True,
        ).strip()

    # ----- multimodal path (MedGemma-4B / Gemma3, Qwen-VL, Llava, …) -----

    def _call_local_multimodal(
        self,
        messages: list[dict[str, str]],
        model: str,
        device: str,
        quantization: Optional[str],
        hf_token: Optional[str],
    ) -> str:
        """Text-only invocation of a multimodal model.

        The plain ``AutoModelForCausalLM`` path produces empty output on
        Gemma3-style architectures because the model's chat template
        expects vision tokens to be reachable through the processor.
        We instead use ``AutoModelForImageTextToText`` (or its older
        Vision2Seq alias) and feed messages in the ``[{"type": "text",
        "text": …}]`` shape the processor's apply_chat_template wants.
        No image is attached — the model still produces the assistant
        turn from the textual context.

        Critically, Gemma3 family (incl. MedGemma-4B-it) emits NaN logits
        when loaded in float16 — the official HF example uses bfloat16.
        We override torch_dtype here regardless of what the text path
        prefers, otherwise generate() returns 0 new tokens silently.
        """
        import torch
        from transformers import AutoProcessor

        cache_key = f"{model}:{quantization}:mm"

        if cache_key not in self._local_model_cache:
            logger.info(
                "Loading local MULTIMODAL model %s (device=%s, quant=%s)…",
                model, device, quantization,
            )
            self._purge_cuda_cache()   # reclaim VRAM from any prior tool
            load_kwargs = self._local_load_kwargs(device, quantization, hf_token)
            # Force bfloat16 for multimodal models — float16 produces NaN
            # logits on Gemma3 / MedGemma and silently yields empty output.
            load_kwargs["torch_dtype"] = torch.bfloat16
            if "quantization_config" in load_kwargs:
                # bnb 4-bit compute dtype must match the model dtype to
                # avoid the same NaN issue at the quantized-matmul boundary.
                load_kwargs["quantization_config"].bnb_4bit_compute_dtype = torch.bfloat16

            # Try ImageTextToText (newest API) → Vision2Seq → CausalLM.
            lm = None
            last_err: Exception | None = None
            for AutoClsName in ("AutoModelForImageTextToText",
                                "AutoModelForVision2Seq",
                                "AutoModelForCausalLM"):
                try:
                    import transformers as _t
                    AutoCls = getattr(_t, AutoClsName)
                    lm = AutoCls.from_pretrained(model, **load_kwargs)
                    logger.info("Loaded %s via %s", model, AutoClsName)
                    break
                except (AttributeError, ValueError) as exc:
                    last_err = exc
                    continue
            if lm is None:
                raise RuntimeError(
                    f"Could not load {model} with any AutoModel class. "
                    f"Last error: {last_err}"
                )

            processor = AutoProcessor.from_pretrained(
                model, trust_remote_code=True, token=hf_token,
            )
            lm.eval()
            self._local_model_cache[cache_key] = (processor, lm)
            logger.info("Local MULTIMODAL model %s loaded.", model)

        processor, lm = self._local_model_cache[cache_key]

        # Re-shape dict messages into the multimodal content format the
        # multimodal processors expect. We only attach text — no image.
        mm_messages: list[dict[str, Any]] = []
        for msg in messages:
            mm_messages.append({
                "role": msg.get("role", "user"),
                "content": [{"type": "text", "text": msg.get("content", "")}],
            })

        # Tokenize inline; same one-shot pattern as medgemma_tool.py.
        try:
            inputs = processor.apply_chat_template(
                mm_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            # Some older processors expose this only on .tokenizer; try that.
            logger.warning(
                "processor.apply_chat_template failed for %s (%s) — "
                "falling back to tokenizer.apply_chat_template",
                model, exc,
            )
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor.tokenizer(prompt, return_tensors="pt")

        model_device = next(lm.parameters()).device
        inputs = {
            k: v.to(model_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]

        # Generation config: smaller cap (1024 is more than enough for the
        # AgentBrief schema), explicit pad/eos so the model doesn't see
        # `pad_token_id is None` and bail out, and min_new_tokens guards
        # against a degenerate first-step EOS.
        tokenizer = getattr(processor, "tokenizer", None) or processor
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            pad_id = eos_id

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 1024,
            "min_new_tokens": 16,
            "do_sample": False,
        }
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        start = time.monotonic()
        with torch.inference_mode():
            output_ids = lm.generate(**inputs, **gen_kwargs)
        elapsed = time.monotonic() - start
        new_token_count = output_ids.shape[-1] - input_len
        logger.info(
            "Local MULTIMODAL inference (%s) → %d new tokens in %.2fs",
            model, new_token_count, elapsed,
        )

        # processor exposes .decode in newer transformers; fall back to
        # the underlying tokenizer otherwise.
        decode = getattr(processor, "decode", None) or processor.tokenizer.decode
        return decode(
            output_ids[0][input_len:], skip_special_tokens=True,
        ).strip()

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
